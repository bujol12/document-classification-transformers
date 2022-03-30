import json
import itertools

from torch.utils.data.dataset import Dataset
from transformers import AutoTokenizer

from dataclasses import dataclass
from typing import List

from config import Config


class JsonDocumentDataset(Dataset):
    """
    Class representing the dataset to be used for training and evaluation of the model
    Handles tokenisation using the chosen model's tokeniser.
    """

    def __init__(self, filepath: str, config: Config):
        """
        Initialise the Dataset from the given file, which contains a dataset
        of json format, of the following hierarchy:

        {"documents": [{"tokens" : [[]], "document_label": ?, "sentence_labels": [], "token_labels": [[]]}]}

        :param filepath: filepath to the input JSON file
        :param config: object containing the whole configuration available
        """
        with open(filepath) as f:
            data_dict = json.load(f)

        # Check generic format of the input
        assert "documents" in data_dict.keys()
        assert {"tokens", "document_label", "sentence_labels", "token_labels"} == set(data_dict["documents"][0].keys())

        self.input_tokens: List[List[List[str]]] = []
        self.document_labels: List[int] = []
        self.token_labels: List[List[List[int]]] = []
        self.sentence_labels: List[List[int]] = []

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
        self.tokeniser = AutoTokenizer.from_pretrained(config.model_name, add_prefix_space=True)

    def __len__(self) -> int:
        return len(self.input_tokens)

    def __getitem__(self, idx):
        input_tokens = self.input_tokens[idx]
        # flatten the list to input to the tokeniser
        # returns dict{'input_ids': [], 'attention_mask': []}
        tokens = self.tokeniser(list(itertools.chain(*input_tokens)), is_split_into_words=True)

        # TODO: split+flatten of sentence and token labels
        return tokens, self.document_labels[idx]
