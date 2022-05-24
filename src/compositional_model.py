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

    def __init__(self, config, lm_config, device):
        super().__init__()
        self.config = config
        self.lm_config = lm_config
        self.device = device

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
        :param input_ids: list -> batch_size x num_sentences x max_word_len
        :return:
        """
        batch_size = len(input_ids)

        # apply transformers to the whole batch per sentence, until run out of sentences
        self.token_outputs = []
        self.document_outputs = []

        for batch_id in range(batch_size):
            print(batch_id)
            batch_token_outputs = []
            batch_attention_masks = []

            for i in range(len(input_ids[batch_id])):
                lm_outputs = self.language_model(
                    torch.tensor([input_ids[batch_id][i]]).to(self.device),
                    attention_mask=torch.tensor([attention_mask[batch_id][i]]).to(self.device),
                )

                # get last unmasked token in the sentence
                last_token_idx = torch.tensor(attention_mask[batch_id][i]).nonzero(as_tuple=True)[0][-1] + 1

                batch_attention_masks.append(torch.tensor(attention_mask[batch_id][i])[1:last_token_idx])

                # append token embeddings to the document representation
                batch_token_outputs.append(lm_outputs.last_hidden_state[0, 1:last_token_idx])

            batch_token_outputs_tensor = torch.cat(batch_token_outputs).unsqueeze_(0)
            batch_attention_masks_tensor = torch.cat(batch_attention_masks).unsqueeze_(0)


            # pad inputs to be the same length
            batch_token_outputs_tensor = torch.nn.functional.pad(batch_token_outputs_tensor, (
                0, 0, 0, self.config.compositional_model_max_token_len - batch_token_outputs_tensor.shape[1], 0, 0),
                                                                 value=-100)
            batch_attention_masks_tensor = torch.nn.functional.pad(batch_attention_masks_tensor, (
                0, self.config.compositional_model_max_token_len - batch_attention_masks_tensor.shape[1], 0, 0),
                                                                   value=0)
            print(batch_token_outputs_tensor.shape, batch_attention_masks_tensor.shape)
            document_logits, token_outputs = self.soft_attention_tokens(
                batch_token_outputs_tensor.to(self.device), batch_attention_masks_tensor.to(self.device))

            self.token_outputs.append(token_outputs)
            self.document_outputs.append(document_logits)

        self.document_logits = torch.vstack(self.document_outputs).to(self.device)
        self.token_outputs = torch.vstack(self.token_outputs).to(self.device)

        return self.document_logits, self.token_outputs

    def loss(self, document_targets, weights=None):
        assert len(set(document_targets.tolist())) <= 2  # only support binary for now

        # Calculate loss on the document-level prediction
        document_logits = self.document_logits

        if self.config.num_labels == 1:
            # MSE
            criterion = torch.nn.MSELoss()
            document_targets_loss = document_targets.to(torch.float32)[:, None]
            document_logits = torch.nn.Sigmoid()(self.document_logits)
        elif self.config.num_labels == 2:
            criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
            document_targets_loss = torch.nn.functional.one_hot(document_targets,
                                                                num_classes=self.config.num_labels).to(
                torch.float32).to(document_targets.device)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
            document_targets_loss = document_targets

        document_loss = criterion(document_logits, document_targets_loss)

        return document_loss
