import json
import itertools
import logging
from typing import List, Any, Dict, Tuple
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

    tokenised_input: List

    config: Config

    def __init__(self, filepath: str, config: Config):
        """
        Initialise the Dataset from the given file, which contains a dataset
        of json format, of the following hierarchy:

        {"documents": [{"tokens" : [[]], "document_label": ?, "sentence_labels": [], "token_labels": [[]]}]}

        :param filepath: filepath to the input JSON file
        :param config: object containing the whole configuration available
        """
        JsonDocumentDataset.config = config

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
            JsonDocumentDataset.tokeniser = AutoTokenizer.from_pretrained(config.transformers_model_name_or_path,
                                                           add_prefix_space=True)
        else:
            JsonDocumentDataset.tokeniser = AutoTokenizer.from_pretrained(config.transformers_model_name_or_path)

        self.tokenised_input = [None for _ in range(len(self.document_labels))]
        #self.__tokenise_and_align()

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
        :return: dict {'attention_mask': [], 'input_ids': [], 'label_ids': [], 'label': ?, 'word_ids': []}
        """
        # res = {}
        # for k, v in self.tokenised_input.items():
        #     res[k] = v[idx]
        # return res, idx

        res = {
            "input_tokens": self.input_tokens[idx],
            "document_labels": self.document_labels[idx],
            "token_labels": self.token_labels[idx],
            "sentence_labels": self.sentence_labels[idx]
        }

        return res, idx

    @classmethod
    def __tokenise_and_align(cls, input_tokens, document_labels, token_labels, sentence_labels):
        """
        Tokenise the input using the transformer tokeniser for the model
        and align the labels to only start at the beginning of each word
        """

        if cls.config.compose_sentence_representations:
            # tokenise each sentence separately
            tokenised_input = None

            for doc_id, doc_tokens in enumerate(input_tokens):
                # tokenise each sentence in document separately
                tokenised_document = cls.tokenise_flat_input(doc_tokens, token_labels[doc_id])

                # assign document-level labels
                tokenised_document["label"] = document_labels[doc_id]

                # assign sentence-level labels
                tokenised_document["sentence_labels"] = sentence_labels[doc_id]
                tokenised_document["sentence_preds"] = [None for _ in range(len(tokenised_document["sentence_labels"]))]
                tokenised_document["pred"] = None

                tokenised_document["words"] = input_tokens[doc_id]
                tokenised_document["word_labels"] = token_labels[doc_id]
                tokenised_document["word_preds"] = [None for _ in range(len(tokenised_document["word_labels"]))]

                tokenised_input = cls.__extend_tokenised_input(tokenised_input, tokenised_document)
        else:
            # tokenise each document separately

            # flatten the sentences into a single list representing tokens in document
            flat_input_tokens = []
            flat_input_token_labels = []
            for doc, labels in zip(input_tokens, token_labels):
                flat_input_tokens.append(list(itertools.chain(*doc)))
                flat_input_token_labels.append(list(itertools.chain(*labels)))

            # tokenise
            tokenised_input = cls.tokenise_flat_input(flat_input_tokens, token_labels)

            # assign document-level labels
            tokenised_input["label"] = document_labels

            tokenised_input["sentence_labels"] = [None for _ in range(len(tokenised_input["label"]))]
            tokenised_input["sentence_preds"] = [None for _ in range(len(tokenised_input["label"]))]

            tokenised_input["pred"] = [None for _ in range(len(tokenised_input["label"]))]

            tokenised_input["words"] = flat_input_tokens
            tokenised_input["word_labels"] = flat_input_token_labels
            tokenised_input["word_preds"] = [None for _ in range(len(flat_input_token_labels))]

        return tokenised_input

    @classmethod
    def tokenise_flat_input(cls, flat_input_tokens, token_labels):
        """
        Tokenise a dataset consisting of 2-d array of input tokens.
        Assign token-level labels and word ids for each token
        :param flat_input_tokens:
        :return:
        """

        # tokenise
        tokenised_input = cls.tokeniser(flat_input_tokens, is_split_into_words=True,
                                         #max_length=cls.config.max_transformer_input_len,
                                         padding="longest",
                                         truncation=True)

        # assign token-level labels
        tokenised_input["label_ids"] = cls.__generate_tokenised_labels(token_labels, tokenised_input)

        # copy over word ids
        tokenised_input["word_ids"] = [
            [word_idx if word_idx is not None else -1 for word_idx in tokenised_input.word_ids(batch_index=i)]
            for i in
            range(len(tokenised_input["label_ids"]))]

        # add token scores placeholder
        tokenised_input["token_preds"] = [None for _ in range(len(tokenised_input["input_ids"]))]

        # add original tokens
        tokenised_input["tokens"] = [cls.tokeniser.convert_ids_to_tokens(sent) for sent in
                                     tokenised_input["input_ids"]]

        return tokenised_input

    @classmethod
    def __generate_tokenised_labels(cls, token_labels, tokenised_input):
        """
        iterate through examples and get labels for each token, setting labels for special tokens == -100
        taken from HuggingFace example
        https://github.com/huggingface/transformers/blob/main/examples/pytorch/token-classification/run_ner_no_trainer.py#L416
        :return: tokenised labels
        """
        processed_labels = []
        for i, labels in enumerate(token_labels):
            if type(labels[0]) == list:
                flat_labels = list(itertools.chain(*labels))
            else:
                flat_labels = deepcopy(labels)
            word_ids = tokenised_input.word_ids(batch_index=i)

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
                    if cls.config.label_all_tokens:
                        label_ids.append(flat_labels[word_idx])
                    else:
                        label_ids.append(-100)
                previous_word_idx = word_idx

            truncated_labels = flat_labels[max([word for word in word_ids if word is not None]) + 1:]
            label_ids += truncated_labels  # add labels of truncated tokens
            processed_labels.append(label_ids)
        return processed_labels

    def own_default_collator(self, features_with_idx: List[Tuple[Dict, int]]) -> Dict[str, torch.Tensor]:
        """
        Similar to HuggingFace default_data_collator, but does not have special handling for labels_ids etc
        Truncate input that's too long, but leave labels of original length

        Returns a Dictionary containing BATCHED inputs, each being a torch tensor
        Usually used for collating batches of flattened documents (no sentence-depth)
        :return:
        """

        # split features and idx lists
        features = {}
        for feat, _ in features_with_idx:
            for k, v in feat.items():
                features[k] = features.get(k, []) + [v]
        idx = [idx for _, idx in features_with_idx]

        # tokenise the batch
        features = self.__tokenise_and_align(**features)

        for idx_val in idx:
            self.tokenised_input[idx_val] = {k: v[idx_val] for k, v in features.items()}

        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in features and features["label"][0] is not None:
            label = features["label"][0].item() if isinstance(features["label"][0], torch.Tensor) else features["label"][0]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["label"] = torch.tensor([f for f in features["label"]], dtype=dtype)
        if "label_ids" in features and features["label_ids"][0] is not None:
            if isinstance(features["label_ids"][0], torch.Tensor):
                batch["label_ids"] = torch.stack([f for f in features["label_ids"]])
            else:
                dtype = torch.long if type(features["label_ids"][0][0]) is int else torch.float
                batch["label_ids"] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f, dtype=dtype) for f in features["label_ids"]], batch_first=True, padding_value=-100)

                # pad all label ids to be of the same length across the batch, -100 shows end of original label_ids
                # batch["label_ids"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in features.items():
            if k not in ("label", "label_ids") and v[0] is not None and not isinstance(v[0], str):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack([f for f in features[k]])
                if k in ("word_labels"): # word labels aren't of the same shape for each (different number of words)
                    batch[k] = [f for f in features[k]]
                elif not isinstance(v[0], str) and (not any([isinstance(e, str) for e in v[0]]) if isinstance(v[0],
                                                                                                        list) else True):  # skip strings as those cannot be converted to tensors
                    batch[k] = torch.tensor([f for f in features[k]])

        batch["dataset_idx"] = torch.tensor(idx)
        return batch

    def compositional_collator(self, features_with_idx: List[Tuple[Dict, int]]) -> Dict[str, Any]:
        """
        Similar to HuggingFace default_data_collator, but does not have special handling for labels_ids etc
        Truncate input that's too long, but leave labels of original length

        Return a list (instead of tensor) that contains a variable number of sentences,

        Usually used to collate batches of documents, where each document consists of a variable number of sentences,
        where each sentence is padded to the same length.

        :return:
        """
        # split features and idx lists
        features = {}
        for feat, _ in features_with_idx:
            for k, v in feat.items():
                features[k] = features.get(k, []) + [v]
        idx = [idx for _, idx in features_with_idx]

        # tokenise the batch
        features = self.__tokenise_and_align(**features)

        for idx_val in idx:
            self.tokenised_input[idx_val] = {k: v[idx_val] for k, v in features.items()}

        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in features and features["label"][0] is not None:
            label = features["label"][0].item() if isinstance(features["label"][0], torch.Tensor) else features["label"][0]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["label"] = torch.tensor([f for f in features["label"]], dtype=dtype)
        if "label_ids" in features and features["label_ids"][0] is not None:
            if isinstance(features["label_ids"][0], torch.Tensor):
                batch["label_ids"] = torch.stack([f for f in features["label_ids"]])
            else:
                dtype = torch.long if type(features["label_ids"][0][0]) is int else torch.float
                batch["label_ids"] = [  # pad each document individually
                    torch.nn.utils.rnn.pad_sequence([torch.tensor(sent_label_ids) for sent_label_ids in f],
                                                    batch_first=True, padding_value=-100)
                    for f in features["label_ids"]]

                # pad all label ids to be of the same length across the batch, -100 shows end of original label_ids
                # batch["label_ids"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in features.items():
            if k not in ("label", "label_ids") and v[0] is not None and not isinstance(v[0], str):
                if isinstance(v[0], torch.Tensor):
                    batch[k] = torch.stack([f for f in features[k]])
                else:
                    batch[k] = [f for f in features[k]]

        batch["dataset_idx"] = torch.tensor(idx)
        return batch

    @staticmethod
    def __extend_tokenised_input(tokenised_input: BatchEncoding, tokenised_document: BatchEncoding):
        """
        Key-wise extend the given input with the newly processed document
        :param tokenised_input:
        :param tokenised_document:
        :return:
        """

        if tokenised_input is None:
            # if not input, simply return the part that's extending it
            tokenised_input = tokenised_document
            for k, v in tokenised_document.items():
                tokenised_input[k] = [v]
        else:
            for k, v in tokenised_document.items():
                tokenised_input[k].append(v)

        return tokenised_input

    def save_predictions(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.tokenised_input, f)

    def add_preds(self, indices: List[int], document_preds: List[float], token_preds: List[Any]):
        """
        Add the document and token-level predictions to the dataset

        :param document_preds:
        :param token_preds: can be a 1-d or 2-d list (depending if input flattened or not)
        :return:
        """
        for i, idx in enumerate(indices):
            self.tokenised_input[idx]["pred"] = document_preds[i]
            if token_preds != []:
                self.tokenised_input[idx]["token_preds"] = token_preds[i]
            else:
                self.tokenised_input[idx]["token_preds"] = None

    def pretty_print(self, idx: int) -> Dict:
        """
        Get columns the most important for inclusion in the print
        :param idx:
        :return:
        """
        res, _ = self[idx]
        final_dct = {}
        for k, v in res.items():
            if k in ['attention_mask', 'input_ids']:
                continue
            final_dct[k] = v
        return final_dct
