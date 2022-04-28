import torch
import logging

from transformers import AutoModel, PretrainedConfig, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from config import Config

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

        self.cls_dropout = torch.nn.Dropout(p=self.config.dropout)
        self.cls_logit_layer = torch.nn.Linear(self.lm_config.hidden_size, self.config.num_labels)

        self.__init_weights(self.cls_dropout)
        self.__init_weights(self.cls_logit_layer)

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

        logits = self.cls_logit_layer(self.cls_dropout(self.lm_outputs.pooler_output))  # last_hidden_state[:, 0])
        token_outputs = self.lm_outputs.last_hidden_state[:, 1:]

        return logits, token_outputs

    @staticmethod
    def get_transformers_config(config: Config) -> PretrainedConfig:
        """
        Get the config for the given transformer model
        + do modifications we would like
        :param config:
        :return: Transformers configs
        """
        return AutoConfig.from_pretrained(config.transformers_model_name_or_path, **config.transformers_override)

    def __init_weights(self, m):
        if self.config.initializer_name == "normal":
            self.initializer = torch.nn.init.normal_
        elif self.config.initializer_name == "glorot":
            self.initializer = torch.nn.init.xavier_normal_
        elif self.config.initializer_name == "xavier":
            self.initializer = torch.nn.init.xavier_uniform_

        if isinstance(m, torch.nn.Linear):
            self.initializer(m.weight)
            torch.nn.init.zeros_(m.bias)
