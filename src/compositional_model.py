import torch

import logging

from transformers import AutoModel, PretrainedConfig, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput

from .config import Config
from .soft_attention_layer import SoftAttentionLayer

logger = logging.getLogger(__name__)


class CompositionalModel(torch.nn.Module):
    """
    Implement a compositional Document Classifier, where a Transformer is applied invidividually to each sentence
    and then the document and token labels are inferred.
    """

    config: Config
    language_model: PreTrainedModel
    lm_config: PretrainedConfig

    def __init__(self, config, lm_config):
        super().__init__()
        self.config = config
        self.lm_config = lm_config

        logger.info("Setting up a transformer model with Config")
        logger.info(self.lm_config)

        self.language_model = AutoModel.from_pretrained(
            self.config.transformers_model_name_or_path,
            config=self.lm_config,
        )

        # Soft Attention directly on tokens
        self.soft_attention_tokens = SoftAttentionLayer(self.config, self.lm_config.hidden_size)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, **kwargs):
        """
        Call each component of the model
        :param input_ids: tensor -> batch_size x num_sentences x max_word_len
        :return:
        """
        print(input_ids)
