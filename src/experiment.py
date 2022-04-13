import logging
import math
import datetime
import os
import json

import torch

from sklearn.metrics import mean_squared_error
from transformers import set_seed, default_data_collator, get_constant_schedule_with_warmup
from torch.utils.data import DataLoader

from .document_model import DocumentModel
from .json_document_dataset import JsonDocumentDataset
from .config import Config
from .metrics import Metrics

logger = logging.getLogger(__name__)


class Experiment:
    config: Config
    model: DocumentModel
    train_dataset: JsonDocumentDataset = None
    eval_dataset: JsonDocumentDataset = None
    train_dataloader: DataLoader = None
    eval_dataloader: DataLoader = None
    device: torch.device = None
    experiment_id: str = datetime.datetime.now().isoformat()
    experiment_folder: os.path = None

    def __init__(self, config_filepath, train_data_filepath=None, eval_data_filepath=None):
        # load in config
        self.config = Config.from_json(config_filepath)

        # set seed for reproducible behaviour
        set_seed(self.config.seed)

        # setup the model
        self.model = DocumentModel(self.config)

        # setup the datasets
        if train_data_filepath is not None:
            self.train_dataset = JsonDocumentDataset(train_data_filepath, self.config)
            logger.info(f"Example processed dataset: {self.train_dataset[0]}")

        if eval_data_filepath is not None:
            self.eval_dataset = JsonDocumentDataset(eval_data_filepath, self.config)

        # Create collator for DataLoaders
        self.data_collator = JsonDocumentDataset.own_default_collator

        # Data loaders
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(self.train_dataset, collate_fn=self.data_collator,
                                               batch_size=self.config.train_batch_size)

        if self.eval_dataset is not None:
            self.eval_dataloader = DataLoader(self.eval_dataset, collate_fn=self.data_collator,
                                              batch_size=self.config.eval_batch_size)

        # Move model to our chosen device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        self.model.to(self.device)

        # Setup experiment folder
        self.experiment_folder = os.path.join(os.getcwd(), "runs", self.experiment_id)
        os.mkdir(self.experiment_folder)
        self.save_config()

    def train(self):
        assert self.train_dataset is not None

        # setup the optimiser
        if self.config.optimiser == "adam":
            optimiser = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, eps=self.config.opt_eps)
        else:
            raise Exception("Unknown optimiser")

        # setup lr scheduler (for warmup)
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        total_train_steps = self.config.epochs * num_update_steps_per_epoch
        warmup_steps = total_train_steps * self.config.warmup_ratio

        lr_scheduler = get_constant_schedule_with_warmup(optimiser, warmup_steps)

        # double check the model is moved to self.device, if not, move it
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # training loop
        prev_eval_loss = None
        early_stop_cnt = 0

        for epoch in range(self.config.epochs):
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                # move to device
                moved_batch = {}
                for k, v in batch.items():
                    moved_batch[k] = v.to(self.device)

                cls_logit, token_outputs = self.model(**moved_batch)
                # TODO: token outputs can be smaller than batch["label_ids"]
                #  -> right-pad with 0s for tokens where batch["label_ids"] != -100
                #  (-100 is padding to ensure all batch label_ids have the same length)
                loss = self.__calculate_loss(cls_logit, moved_batch["label"], token_outputs, moved_batch["label_ids"])
                loss = loss / self.config.gradient_accumulation_steps

                loss.backward()

                # if gradient fully accumulated, update parameters
                if step % self.config.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    optimiser.step()
                    lr_scheduler.step()
                    optimiser.zero_grad()

                # free up GPU
                keys = moved_batch.keys()
                for k in list(keys):
                    del moved_batch[k]
                del cls_logit
                del token_outputs
                torch.cuda.empty_cache()

            # Evaluate at the end of each epoch
            train_performance = self.eval(self.train_dataloader)
            eval_performance = self.eval()

            print()
            logger.info(f"Finished epoch {epoch + 1} out of {self.config.epochs}")
            logger.info(f"Training dataset performance: {train_performance.to_json()}")
            logger.info(f"Eval dataset performance: {eval_performance.to_json()}")
            print()

            if self.config.stop_if_no_improvement_n_epochs != -1:
                if prev_eval_loss is None:
                    # first epoch
                    prev_eval_loss = eval_performance.loss
                    early_stop_cnt = 0
                elif prev_eval_loss > eval_performance.loss:
                    # loss decreasing
                    early_stop_cnt = 0
                    prev_eval_loss = eval_performance.loss
                else:
                    # loss not improving
                    early_stop_cnt += 1

                if early_stop_cnt >= self.config.stop_if_no_improvement_n_epochs:
                    # stop if no improvement
                    logger.info(f"No improvement of eval loss after {early_stop_cnt} epochs, early stopping...")
                    return

    def eval(self, data_loader=None):
        if data_loader is None:
            assert self.eval_dataset is not None
            data_loader = self.eval_dataloader

        # double check the model is moved to self.device, if not, move it
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # Evaluation loop
        self.model.eval()

        document_predictions = []
        true_document_labels = []
        token_predictions = []
        true_token_labels = []

        total_loss = torch.zeros(1, dtype=torch.float)

        for step, batch in enumerate(data_loader):
            # move to device
            moved_batch = {}
            for k, v in batch.items():
                moved_batch[k] = v.to(self.device)

            with torch.no_grad():
                cls_probs, token_outputs = self.model(**moved_batch)

            total_loss += len(moved_batch["label"]) * self.__calculate_loss(cls_probs, moved_batch["label"],
                                                                            token_outputs,
                                                                            moved_batch["label_ids"])
            document_predictions += cls_probs.detach().cpu().tolist()
            true_document_labels += moved_batch["label"].detach().cpu().tolist()

            # free up GPU
            keys = moved_batch.keys()
            for k in list(keys):
                del moved_batch[k]

            del cls_probs
            del token_outputs
            torch.cuda.empty_cache()

        return Metrics(torch.tensor(document_predictions), torch.tensor(true_document_labels),
                       loss=total_loss.item() / len(data_loader))

    def __calculate_loss(self, cls_logit, cls_targets, token_outputs, token_targets):
        """
        Calculate the loss function given the document and token predictions. TOOD: add token-based loss
        :param cls_logit:
        :param cls_targets:
        :param token_outputs:
        :param token_targets:
        :return:
        """
        assert len(set(cls_targets.tolist())) <= 2  # only support binary for now
        criterion = torch.nn.CrossEntropyLoss()
        loss = criterion(cls_logit, cls_targets)
        return loss

    def save_model(self, path: str):
        """
        Save the trained model
        :param path:
        :return:
        """
        torch.save(self.model.state_dict(), os.path.join(self.experiment_folder, path))

    def save_results(self, metrics: Metrics, path: str):
        """
        Save the given metrics
        :param metrics:
        :param path:
        :return:
        """
        with open(os.path.join(self.experiment_folder, path), 'w') as f:
            json.dump(metrics.to_json(), f)

    def save_config(self):
        """
        Dump the config
        :return:
        """
        self.config.to_json(os.path.join(self.experiment_folder, "config.json"))
