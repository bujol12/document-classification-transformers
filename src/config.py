import json
import logging

from dataclasses import dataclass, field, asdict
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Config:
    transformers_model_name_or_path: str  # name of the transformer model to be used as base
    train_batch_size: int  # batch size for training
    eval_batch_size: int  # batch size for evaluation
    num_labels: int  # number of labels to be predicted
    epochs: int  # number of training epochs
    stop_if_no_improvement_n_epochs: int  # number of epochs after which if no improvement on eval, stop (or -1)
    gradient_accumulation_steps: int  # number of steps over which gradient is accumulated
    dataset_name: str

    # Transformers related config
    transformers_override: dict[str, Any] = field(default_factory=lambda: {
        "output_hidden_states": True,
        "output_attentions": True
    })  # specify overrides to transformers config
    max_transformer_input_len: int = 512  # maximum len of tokenised input to transformer
    predict_document_label_from_cls: bool = True  # use the prediction of document-level label from the CLS token

    # Soft Attention Config
    soft_attention: bool = False  # apply the standard soft attention layer from Bujel 2021
    soft_attention_dropout: float = 0.10  # dropout to apply in soft attention layer
    soft_attention_evidence_size: int = 100  # size of attention evidence layer
    soft_attention_hidden_size: int = 300

    # Compositional Approach Config
    compose_sentence_representations: bool = False  # if set true, apply transformer to each sentence separately. If false, apply transformer to the whole document
    compositional_sentence_batch_size: int = 4 # how many sentence to feed into BERT at once
    compositional_model_max_token_len: int = 4096  # same as Longformer

    # Misc
    infer_labels: bool = False  # take max of labels for sentence and document if not present
    seed: int = 22
    label_all_tokens: bool = False  # if False, only first part of token has a label, with rest = -100
    min_epochs: int = 10  # min epochs to run for, ignoring early stopping

    # Hyperparameters
    optimiser: str = "adamW"  # name of the optimiser to be used
    lr: float = 2e-5  # learning rate
    opt_eps: float = 1e-7  # eps of the optimiser
    warmup_ratio: float = 0.06  # % number of steps for the optimiser to warmup the learning rate
    dropout: float = 0.10  # dropout outside of the transformer
    initializer_name: str = "normal"  # how to initialise new layers
    weighted_loss: bool = False  # use weighted loss fucntion to tackle class imbalance
    token_loss_gamma: float = 0.0  # gamma parameter of the weight of the loss function for token-level calculations
    regularisation_loss_gamma: float = 0.0  # regularisation of the token attention scores
    mean_token_loss_gamma: float = 0.0  # mean of token values should be close to document label
    min_max_token_loss_gamma: float = 0.0  # min/max should be close to document label
    top_k_pct: float = 0.10  # pct of tokens to be optimised as providing evidence
    top_k_token_loss_gamma: float = 0.0  # importance of top-k optimisation on the token level
    top_k_bottom_k: bool = False  # if False, optimise all tokens outside of top k to be 0

    @classmethod
    def from_json(cls, filepath):
        """
        Load the config from json
        :param filepath: path to the file
        """
        logger.info(f"Reading config file from: {filepath}")

        with open(filepath) as f:
            config = json.load(f)
        return cls(**config)

    def to_json(self, filepath):
        """
        Save the config to the given filepath
        :param filepath:
        """

        with open(filepath, 'w') as f:
            json.dump(asdict(self), f)
