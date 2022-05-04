import torch
import logging

from transformers import AutoModel, PretrainedConfig, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .config import Config
from .soft_attention_layer import SoftAttentionLayer

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

        # Transformer CLS token prediction (document-level)
        self.cls_dropout = torch.nn.Dropout(p=self.config.dropout)
        self.cls_logit_layer = torch.nn.Linear(self.lm_config.hidden_size, self.config.num_labels)

        # Soft Attention layer
        self.soft_attention_layer = SoftAttentionLayer(self.config, self.lm_config)

        # Initialise layers
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

        token_transformer_outputs = self.lm_outputs.last_hidden_state[:, 1:]

        # Apply soft attention methods
        if self.config.soft_attention:
            self.document_logits, token_outputs = self.soft_attention_layer(token_transformer_outputs,
                                                                            attention_mask[:, 1:])

        # CLS document prediction
        if self.config.predict_document_label_from_cls:
            self.document_logits = self.cls_logit_layer(
                self.cls_dropout(self.lm_outputs.pooler_output))  # last_hidden_state[:, 0])

        return self.document_logits, token_outputs

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

    def loss(self, document_targets, weights=None):
        assert len(set(document_targets.tolist())) <= 2  # only support binary for now

        # Calculate loss on the document-level prediction
        document_logits = self.document_logits

        if self.config.num_labels == 1:
            # MSE
            criterion = torch.nn.MSELoss()
            document_targets = document_targets.to(torch.float32)[:, None]
            document_logits = torch.nn.Sigmoid()(self.document_logits)
        elif self.config.num_labels == 2:
            criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
            document_targets = torch.nn.functional.one_hot(document_targets, num_classes=self.config.num_labels).to(
                torch.float32).to(document_targets.device)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)

        document_loss = criterion(document_logits, document_targets)

        # Calculate loss on the token-level prediction
        token_loss = 0.0
        if self.config.token_loss_gamma != 0.0:
            if self.config.soft_attention:
                token_loss += self.soft_attention_layer.loss(document_targets)
            else:
                raise Exception("don't support token loss without soft attention")

        loss = document_loss + self.config.token_loss_gamma * token_loss
        return loss
