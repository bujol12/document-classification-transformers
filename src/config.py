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

    infer_labels: bool = False  # take max of labels for sentence and document if not present
    transformers_override: dict[str, Any] = field(default_factory=lambda: {
        "output_hidden_states": True,
        "output_attentions": True
    })  # specify overrides to transformers config
    seed: int = 22
    max_transformer_input_len: int = 512  # maximum len of tokenised input to transformer
    compose_sentence_representations: bool = False  # if set true, apply transformer to each sentence separately. If false, apply transformer to the whole document
    label_all_tokens: bool = False  # if False, only first part of token has a label, with rest = -100
    optimiser: str = "adamW"  # name of the optimiser to be used
    lr: float = 2e-5  # learning rate
    opt_eps: float = 1e-7  # eps of the optimiser
    warmup_ratio: float = 0.10  # number of steps for the optimiser to warmup the learning rate
    dropout: float = 0.10  # dropout outside of the transformer
    initializer_name: str = "xavier" # how to initialise new layers

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
