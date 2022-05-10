import logging
import math
import gc
import datetime
import os
import json
import time
import random

from collections import OrderedDict
from copy import deepcopy

import torch

from transformers import set_seed, get_constant_schedule_with_warmup, get_linear_schedule_with_warmup
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
                                               batch_size=self.config.train_batch_size, shuffle=True)

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
        elif self.config.optimiser == "adamW":
            optimiser = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr, eps=self.config.opt_eps)
        else:
            raise Exception("Unknown optimiser")

        # setup lr scheduler (for warmup)
        num_update_steps_per_epoch = math.ceil(len(self.train_dataloader) / self.config.gradient_accumulation_steps)
        total_train_steps = self.config.epochs * num_update_steps_per_epoch
        warmup_steps = total_train_steps * self.config.warmup_ratio

        lr_scheduler = get_linear_schedule_with_warmup(optimiser, num_warmup_steps=warmup_steps,
                                                       num_training_steps=total_train_steps)

        # double check the model is moved to self.device, if not, move it
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # training loop
        best_eval_loss = None
        early_stop_cnt = 0
        best_model_state_dict = {k: deepcopy(v.to('cpu')) for k, v in self.model.state_dict().items()}
        best_model_state_dict = OrderedDict(best_model_state_dict)
        weights = self.train_dataset.get_weights().to(
            self.device) if self.config.num_labels > 1 else None  # for loss function

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1} Learning Rate: {lr_scheduler.get_last_lr()}")
            self.model.train()

            for step, batch in enumerate(self.train_dataloader):
                # move to device
                moved_batch = {}
                moved_batch["input_ids"] = batch["input_ids"].to(self.device)
                moved_batch["attention_mask"] = batch["attention_mask"].to(self.device)

                document_logits, token_outputs = self.model(**moved_batch)

                # move labels to calculate loss
                moved_batch["label_ids"] = batch["label_ids"].to(self.device)
                moved_batch["label"] = batch["label"].to(self.device)

                # TODO: token outputs can be smaller than batch["label_ids"]
                #  -> right-pad with 0s for tokens where batch["label_ids"] != -100
                #  (-100 is padding to ensure all batch label_ids have the same length)
                loss = self.model.loss(moved_batch["label"], weights=weights)
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

                del document_logits
                del token_outputs
                del moved_batch
                del loss
                gc.collect()
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
                if best_eval_loss is None:
                    # first epoch
                    best_eval_loss = eval_performance.loss
                    early_stop_cnt = 0

                    # move best model to CPU and cache
                    best_model_state_dict = {k: deepcopy(v.to('cpu')) for k, v in self.model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

                elif best_eval_loss < eval_performance.loss:
                    # loss improving
                    early_stop_cnt = 0
                    best_eval_loss = eval_performance.loss

                    # move best model to CPU and cache
                    best_model_state_dict = {k: deepcopy(v.to('cpu')) for k, v in self.model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

                else:
                    # loss not improving on eval
                    early_stop_cnt += 1

                if early_stop_cnt >= self.config.stop_if_no_improvement_n_epochs and epoch > self.config.min_epochs:
                    # stop if no improvement
                    logger.info(f"No improvement of eval loss after {early_stop_cnt} epochs, early stopping...")
                    # restore best model
                    self.model.load_state_dict(best_model_state_dict)

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

        total_len = 0
        total_loss = torch.zeros(1, dtype=torch.float)

        for step, batch in enumerate(data_loader):
            total_len += len(batch["label"])
            # move to device
            moved_batch = {}
            moved_batch["input_ids"] = batch["input_ids"].to(self.device)
            moved_batch["attention_mask"] = batch["attention_mask"].to(self.device)

            with torch.no_grad():
                document_logits, token_outputs = self.model(**moved_batch)

                # move labels to calculate loss
                moved_batch["label_ids"] = batch["label_ids"].to(self.device)
                moved_batch["label"] = batch["label"].to(self.device)

                total_loss += len(moved_batch["label"]) * self.model.loss(moved_batch["label"]).detach().cpu()

            document_predictions += document_logits.detach().cpu().tolist()
            true_document_labels += moved_batch["label"].detach().cpu().tolist()

            if token_outputs is not None:
                token_predictions += self.__convert_token_preds_to_words(
                    token_outputs.detach().cpu().tolist(), batch)  # convert token preds to word preds by taking max
                # TODO:
                #  1) zero-out tru labels_ids for negative documents;
                #  2) skip all negative documents
                #  3) evaluate as-is
                true_token_labels += moved_batch["label_ids"].detach().cpu().tolist()

            # free up GPU
            del document_logits
            del token_outputs
            del moved_batch

            torch.cuda.empty_cache()

        if true_token_labels != []:
            rand_idx = random.randint(0, len(true_token_labels))
            logger.info(f"Sample Idx = {rand_idx}")
            logger.info(f"token predictions: {token_predictions[rand_idx]}")
            logger.info(f"true token labels: {true_token_labels[rand_idx]}")

        return Metrics(torch.tensor(document_predictions), torch.tensor(true_document_labels),
                       loss=total_loss.item() / total_len, token_true=true_token_labels,
                       token_preds=token_predictions)

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

    def __convert_token_preds_to_words(self, token_preds, dataset):
        """
        Convert token level predictions to predictions on words by taking maximum of each word score
        (split across many tokens)
        :param token_preds:
        :param dataset:
        :return:
        """
        new_token_preds = []

        # Go through the batch
        for i, doc in enumerate(token_preds):
            # score for each word initially is 0
            word_preds = torch.zeros(max(dataset['word_ids'][i]) + 1, dtype=torch.float)

            # Iterate through tokens
            for j in range(len(doc)):
                if dataset['word_ids'][i][j] == -1:
                    # skip special tokens that do not map to words
                    continue

                # score for word is the maximum of prev max and current token score
                word_preds[dataset['word_ids'][i][j]] = max(
                    word_preds[dataset['word_ids'][i][j]], doc[j])

            # assign the same score to the tokens everywhere in the same word
            new_token_preds.append([word_preds[dataset['word_ids'][i][j]] if
                                    dataset['word_ids'][i][j] != -1 else -100 for j in range(len(doc))])
        return new_token_preds
