import json
import itertools
import logging
from typing import List, Any, Dict
from copy import deepcopy

import torch
from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer, BatchEncoding

from .config import Config

logger = logging.getLogger(__name__)


class JsonDocumentDataset(Dataset):
    """
    Class representing the dataset to be used for training and evaluation of the model
    Handles tokenisation using the chosen model's tokeniser.
    """
    tokeniser: PreTrainedTokenizer
    input_tokens: List[List[List[str]]]
    document_labels: List[int]
    token_labels: List[List[List[int]]]
    sentence_labels: List[List[int]]
    tokenised_input: BatchEncoding

    config: Config

    def __init__(self, filepath: str, config: Config):
        """
        Initialise the Dataset from the given file, which contains a dataset
        of json format, of the following hierarchy:

        {"documents": [{"tokens" : [[]], "document_label": ?, "sentence_labels": [], "token_labels": [[]]}]}

        :param filepath: filepath to the input JSON file
        :param config: object containing the whole configuration available
        """
        self.config = config

        logger.info(f"Loading a dataset in from: {filepath}")

        with open(filepath) as f:
            data_dict = json.load(f)

        # Check generic format of the input
        keys = set(data_dict["documents"][0].keys())
        logger.info(f"columns: {keys}")
        assert "documents" in data_dict.keys()
        assert {"tokens", "document_label", "sentence_labels", "token_labels", "id"} == set(
            data_dict["documents"][0].keys())

        self.input_tokens = []
        self.document_labels = []
        self.token_labels = []
        self.sentence_labels = []

        # Process each document and add to the list
        for doc in data_dict["documents"]:
            # Tokens
            self.input_tokens.append(doc["tokens"])
            self.token_labels.append(doc["token_labels"])

            # Sentences
            sentence_labels = doc["sentence_labels"]
            if doc["sentence_labels"] is None and config.infer_labels:
                # infer labels from tokens
                assert doc["token_labels"] is not None
                sentence_labels = [max(sent) for sent in doc["token_labels"]]
            self.sentence_labels.append(sentence_labels)

            # Document
            if doc["document_label"] is None and config.infer_labels:
                self.document_labels.append(max(self.sentence_labels[-1]))
            else:
                self.document_labels.append(doc["document_label"])

        # Add Transformer tokenizer
        if "roberta" in config.transformers_model_name_or_path.lower() or "longformer" in config.transformers_model_name_or_path.lower():
            self.tokeniser = AutoTokenizer.from_pretrained(config.transformers_model_name_or_path,
                                                           add_prefix_space=True)
        else:
            self.tokeniser = AutoTokenizer.from_pretrained(config.transformers_model_name_or_path)

        self.__tokenise_and_align()

    def get_weights(self, level="document"):
        """
        Get weights representing proportions of each label
        :return: torch.tensor summing up to 1
        """
        counts = {}
        if level == "document":
            for label in self.document_labels:
                counts[label] = counts.get(label, 0) + 1
        else:
            raise Exception("Undefined behaviour")

        weights = torch.zeros(max(counts.keys()) + 1)
        total = sum(counts.values())
        for k, v in counts.items():
            weights[k] = v / total
        return weights

    def __len__(self) -> int:
        return len(self.input_tokens)  # number of documents

    def __getitem__(self, idx):
        """
        Get idx-th example from the tokenised dataset
        :param idx:
        :return: dict {'attention_mask': [], 'input_ids': [], 'label_ids': [], 'label': ?}
        """
        res = {}
        for k, v in self.tokenised_input.items():
            res[k] = v[idx]
        return res

    def __tokenise_and_align(self):
        """
        Tokenise the input using the transformer tokeniser for the model
        and align the labels to only start at the beginning of each word
        """
        logger.info("Tokenising the dataset....")

        if self.config.compose_sentence_representations:
            # tokenise each sentence separately
            # TODO: add labels for each sentence
            raise Exception("Composition of sentence representations not implemented")
        else:
            # tokenise each document separately

            # flatten the sentences into a single list representing tokens in document
            flat_input_tokens = []
            for doc in self.input_tokens:
                flat_input_tokens.append(list(itertools.chain(*doc)))

            # tokenise
            self.tokenised_input = self.tokeniser(flat_input_tokens, is_split_into_words=True,
                                                  max_length=self.config.max_transformer_input_len,
                                                  padding="max_length",
                                                  truncation=True)

            self.tokenised_input["label_ids"] = self.__generate_tokenised_labels()
            self.tokenised_input["label"] = self.document_labels

    def __generate_tokenised_labels(self):
        """
        iterate through examples and get labels for each token, setting labels for special tokens == -100
        taken from HuggingFace example
        https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py#L416
        :return: tokenised labels
        """
        processed_labels = []
        for i, labels in enumerate(self.token_labels):
            if type(labels[0]) == list:
                flat_labels = list(itertools.chain(*labels))
            else:
                flat_labels = deepcopy(labels)
            word_ids = self.tokenised_input.word_ids(batch_index=i)

            previous_word_idx = None
            label_ids = []

            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)

                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(flat_labels[word_idx])
                # For the other tokens in a word, we set the label to either the current label or -100, depending on
                # the label_all_tokens flag.
                else:
                    if self.config.label_all_tokens:
                        label_ids.append(flat_labels[word_idx])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            truncated_labels = flat_labels[max([word for word in word_ids if word is not None]) + 1:]
            label_ids += truncated_labels  # add labels of truncated tokens
            processed_labels.append(label_ids)
        return processed_labels

    @staticmethod
    def own_default_collator(features: List[Any]) -> Dict[str, Any]:
        """
        Similar to HuggingFace default_data_collator, but does not have special handling for labels_ids etc
        Truncate input that's too long
        :return:
        """

        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["label"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        if "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["label_ids"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["label_ids"] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f["label_ids"], dtype=dtype) for f in features], batch_first=True, padding_value=-100)

                # pad all label ids to be of the same length across the batch, -100 shows end of original label_ids
                # batch["label_ids"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                else:
                    batch[k] = torch.tensor([f[k] for f in features])
        return batch
