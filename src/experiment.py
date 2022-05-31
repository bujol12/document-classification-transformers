import logging
import math
import datetime
import os
import json

from collections import OrderedDict
from copy import deepcopy

import torch

from transformers import set_seed, get_linear_schedule_with_warmup, PretrainedConfig, \
    AutoConfig
from torch.utils.data import DataLoader

from src.compositional.compositional_model import CompositionalModel
from .document_model import DocumentModel
from .json_document_dataset import JsonDocumentDataset
from .config import Config
from .metrics import Metrics

logger = logging.getLogger(__name__)


class Experiment:
    config: Config
    model: torch.nn.Module
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
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set seed for reproducible behaviour
        set_seed(self.config.seed)

        self.lm_config = self.get_transformers_config(self.config)

        # setup the model
        if self.config.compose_sentence_representations:
            self.model = CompositionalModel(self.config, self.lm_config, self.device)
        else:
            self.model = DocumentModel(self.config, self.lm_config)

        # setup the datasets
        if train_data_filepath is not None:
            self.train_dataset = JsonDocumentDataset(train_data_filepath, self.config)
            logger.info(f"Example processed dataset: {self.train_dataset[0]}")

        if eval_data_filepath is not None:
            self.eval_dataset = JsonDocumentDataset(eval_data_filepath, self.config)

        # Data loaders
        if self.train_dataset is not None:
            self.train_dataloader = DataLoader(self.train_dataset,
                                               collate_fn=self.train_dataset.own_default_collator if not self.config.compose_sentence_representations else self.train_dataset.compositional_collator,
                                               batch_size=self.config.train_batch_size, shuffle=True)

        # Move model to our chosen device
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
            self.device) if self.config.num_labels > 1 and self.config.weighted_loss else None  # for loss function

        for epoch in range(self.config.epochs):
            logger.info(f"Epoch {epoch + 1} Learning Rate: {lr_scheduler.get_last_lr()}")
            self.model.train()

            for step, orig_batch in enumerate(self.train_dataloader):
                if not self.config.compose_sentence_representations:
                    # move to device
                    batch = self.__move_batch(orig_batch, self.device)
                    # move later for compositional approach
                else:
                    batch = orig_batch

                document_logits, token_outputs = self.model(**batch)
                # assign predictions to the dataset
                self.train_dataset.add_preds(indices=orig_batch["dataset_idx"],
                                             document_preds=document_logits.detach().cpu().tolist(),
                                             token_preds=self.__convert_token_preds_to_words(
                                                 token_outputs, orig_batch))

                loss = self.model.loss(batch["label"].to(self.device), weights=weights)
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()

                # if gradient fully accumulated, update parameters
                if step % self.config.gradient_accumulation_steps == 0 or step == len(self.train_dataloader) - 1:
                    optimiser.step()
                    lr_scheduler.step()
                    optimiser.zero_grad()

                # free up GPU
                del document_logits
                del token_outputs
                del batch
                del loss

                torch.cuda.empty_cache()

            # Evaluate at the end of each epoch
            train_performance = self.eval(self.train_dataset)
            eval_performance = self.eval(self.eval_dataset)

            print()
            logger.info(f"Finished epoch {epoch + 1} out of {self.config.epochs}")
            logger.info(f"Training dataset performance: {train_performance.to_json()}")
            logger.info(f"Eval dataset performance: {eval_performance.to_json()}")
            print()

            # early stopping check
            if self.config.stop_if_no_improvement_n_epochs != -1:
                if best_eval_loss is None:
                    # first epoch
                    best_eval_loss = eval_performance.loss
                    early_stop_cnt = 0

                    # move best model to CPU and cache
                    best_model_state_dict = {k: deepcopy(v.to('cpu')) for k, v in self.model.state_dict().items()}
                    best_model_state_dict = OrderedDict(best_model_state_dict)

                elif best_eval_loss > eval_performance.loss:
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

    def eval(self, eval_dataset):
        data_loader = DataLoader(eval_dataset, collate_fn=eval_dataset.own_default_collator if not self.config.compose_sentence_representations else eval_dataset.compositional_collator, batch_size=self.config.eval_batch_size)

        # double check the model is moved to self.device, if not, move it
        if next(self.model.parameters()).device != self.device:
            self.model.to(self.device)

        # Evaluation loop
        self.model.eval()

        total_len = 0
        total_loss = torch.zeros(1, dtype=torch.float)

        for step, orig_batch in enumerate(data_loader):
            total_len += len(orig_batch["label"])
            if not self.config.compose_sentence_representations:
                # move to device
                batch = self.__move_batch(orig_batch, self.device)
                # move later for compositional approach
            else:
                batch = orig_batch

            with torch.no_grad():
                document_logits, token_outputs = self.model(**batch)

                # assign predictions to the dataset
                eval_dataset.add_preds(indices=orig_batch["dataset_idx"],
                                       document_preds=document_logits.detach().cpu().tolist(),
                                       token_preds=self.__convert_token_preds_to_words(
                                           token_outputs, orig_batch))

                total_loss += len(batch["label"]) * self.model.loss(batch["label"].to(self.device)).detach().cpu()

            # free up GPU
            del document_logits
            del token_outputs
            del batch

            torch.cuda.empty_cache()

        # rand_idx = random.randint(0, len(eval_dataset) - 1)
        # logger.info(f"Sample Idx = {rand_idx}")
        # logger.info(eval_dataset.pretty_print(idx=rand_idx))

        return Metrics(eval_dataset, loss=total_loss.item() / total_len)

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

    def save_predictions(self, dataset: JsonDocumentDataset, path: str):
        dataset.save_predictions(os.path.join(self.experiment_folder, path))

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
        for i, doc_preds in enumerate(token_preds):
            # score for each word initially is 0

            if self.config.compose_sentence_representations:
                # tokens nested into sentences
                word_preds = [[0.0 for _ in range(max(words_sent) + 1)] for words_sent in dataset['word_ids'][i]]
                for j, sent_preds in enumerate(doc_preds):  # sentences
                    for k, token_pred in enumerate(sent_preds):  # tokens
                        if dataset['word_ids'][i][j][k] == -1:
                            # skip special tokens that do not map to words
                            continue

                        # score for word is the maximum of prev max and current token score
                        word_preds[j][dataset['word_ids'][i][j][k]] = max(
                            word_preds[j][dataset['word_ids'][i][j][k]], token_pred)

                    sent_token_preds = [word_preds[j][dataset['word_ids'][i][j][k]] if
                                        dataset['word_ids'][i][j][k] != -1 else -100 for k in range(len(sent_preds))]
                    new_token_preds.append(sent_token_preds)

            else:
                word_preds = [0.0 for _ in range(max(dataset['word_ids'][i]) + 1)]

                # Iterate through tokens
                for j in range(len(dataset['input_ids'][i])):
                    if dataset['word_ids'][i][j] == -1:
                        # skip special tokens that do not map to words
                        continue

                    # score for word is the maximum of prev max and current token score
                    # print(dataset['word_ids'][i][j], len(word_preds))
                    # print(word_preds[dataset['word_ids'][i][j]])
                    # print(doc_preds, j)
                    # print(doc_preds[j])
                    word_preds[dataset['word_ids'][i][j]] = max(
                        word_preds[dataset['word_ids'][i][j]], doc_preds[j])

                # assign the same score to the tokens everywhere in the same word
                new_token_preds.append([word_preds[dataset['word_ids'][i][j]] if
                                        dataset['word_ids'][i][j] != -1 else -100 for j in range(len(dataset['input_ids'][i]))])
        return new_token_preds

    def __move_batch(self, batch, device):
        """
        Move the batch (dict of tensors) to a given device
        :param batch:
        :param device:
        :return: moved batch
        """
        moved_batch = {}
        for k, v in batch.items():
            if k in ["input_ids", "attention_mask", "label"]:  # only move what's needed
                moved_batch[k] = v.to(device)

        return moved_batch

    @staticmethod
    def get_transformers_config(config: Config) -> PretrainedConfig:
        """
        Get the config for the given transformer model
        + do modifications we would like
        :param config:
        :return: Transformers configs
        """
        return AutoConfig.from_pretrained(config.transformers_model_name_or_path, **config.transformers_override)
