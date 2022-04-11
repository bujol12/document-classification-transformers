import torch
import logging

from transformers import AutoModel, PretrainedConfig, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .config import Config

logger = logging.getLogger(__name__)


class DocumentModel(torch.nn.Module):
    """
    Represent a document-level modelm with a given Transformer model
    as base and custom additional layers on top.
    """

    config: Config
    language_model: PreTrainedModel
    lm_config: PretrainedConfig
    lm_outputs: BaseModelOutput

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Base transformer
        self.lm_config = self.get_transformers_config(self.config)

        logger.info("Setting up a transformer model with Config")
        logger.info(self.lm_config)

        self.language_model = AutoModel.from_pretrained(
            self.config.transformers_model_name_or_path,
            config=self.lm_config,
        )

        self.cls_logit_layer = torch.nn.Linear(self.lm_config.hidden_size, self.config.num_labels)
        self.cls_softmax = torch.nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, **kwargs):
        """
        Call each component of the model
        :param kwargs:
        :return:
        """
        self.lm_outputs = self.language_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # last_hidden_state, hidden_states, attentions

        cls_probs = self.cls_softmax(self.cls_logit_layer(self.lm_outputs.last_hidden_state[:, 0]))
        token_outputs = self.lm_outputs.last_hidden_state[:, 1:]
        return cls_probs, token_outputs

    @staticmethod
    def get_transformers_config(config: Config) -> PretrainedConfig:
        """
        Get the config for the given transformer model
        + do modifications we would like
        :param config:
        :return: Transformers configs
        """
        return AutoConfig.from_pretrained(config.transformers_model_name_or_path, **config.transformers_override)
