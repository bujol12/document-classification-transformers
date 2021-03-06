import torch

import logging

from transformers import AutoModel, PretrainedConfig, AutoConfig, PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from torch.utils.data import DataLoader

from src.config import Config
from src.soft_attention_layer import SoftAttentionLayer
from src.compositional.sentence_dataset import SentenceDataset
from src.json_document_dataset import JsonDocumentDataset

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

        # sigmoid to convert document logits to probs
        self.document_probs_layer = torch.nn.Sigmoid() if self.config.num_labels == 1 else torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_mask, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, **kwargs):
        """
        Call each component of the model
        :param input_ids: list -> batch_size x num_sentences x max_word_len
        :return:
        """
        batch_size = len(input_ids)
        #assert batch_size == 1

        # apply transformers to the whole batch per sentence, until run out of sentences
        self.token_outputs = []
        self.document_outputs = []
        sent_boundaries = []

        token_embeddings = []  # torch.zeros((batch_size, self.config.compositional_model_max_token_len, self.lm_config.hidden_size))
        token_attentions = []  # torch.zeros((batch_size, self.config.compositional_model_max_token_len))
        sent_boundaries = []

        # batch_id = 0
        # prepare data loader
        for batch_id in range(batch_size):
            document_dataset = SentenceDataset(input_ids[batch_id], attention_mask[batch_id])
            data_loader = DataLoader(document_dataset, collate_fn=document_dataset.own_default_collator,
                                     batch_size=self.config.compositional_sentence_batch_size,
                                     shuffle=False)

            document_sent_attn_masks = []
            document_sent_token_outputs = []

            for batch_sentences in data_loader:
                input_ids_batch, attention_mask_batch = batch_sentences['input_ids'].to(self.device), batch_sentences[
                    'attention_mask'].to(self.device)

                lm_outputs = self.language_model(input_ids_batch, attention_mask=attention_mask_batch)

                last_token_idx = attention_mask_batch.shape[1] - torch.argmax(torch.fliplr(attention_mask_batch),
                                                                              dim=1)  # find first element beyond the last attn == 1

                document_sent_attn_masks += [sent[1:last_token_idx[i]] for i, sent in
                                             enumerate(attention_mask_batch)]
                document_sent_token_outputs += [sent[1:last_token_idx[i]] for i, sent in
                                                enumerate(lm_outputs.last_hidden_state)]

                del input_ids_batch
                del attention_mask_batch
                del last_token_idx
                del lm_outputs

                torch.cuda.empty_cache()

            # flatten the per sentence output

            batch_token_outputs_tensor = torch.cat(document_sent_token_outputs).unsqueeze_(0)[0]
            batch_attention_masks_tensor = torch.cat(document_sent_attn_masks).unsqueeze_(0)[0]

            token_embeddings.append(batch_token_outputs_tensor)
            token_attentions.append(batch_attention_masks_tensor)

            # calculate sentence boundaries
            doc_sent_boundaries = [0]
            curr_sent_boundary = 0

            for tokens in document_sent_token_outputs:
                curr_sent_boundary += tokens.shape[0]
                doc_sent_boundaries.append(curr_sent_boundary)

            sent_boundaries.append(doc_sent_boundaries)

        # pad all outputs to be of the same size
        token_attentions = torch.nn.utils.rnn.pad_sequence(token_attentions, batch_first=True, padding_value=0)
        max_doc_tokens_len = token_attentions.shape[-1]
        for doc_id in range(batch_size):
            pad_size = max_doc_tokens_len - token_embeddings[doc_id].shape[0]
            token_embeddings[doc_id] = torch.unsqueeze(torch.vstack(
                [token_embeddings[doc_id], torch.zeros((pad_size, self.lm_config.hidden_size)).to(self.device)]), dim=0)
        token_embeddings = torch.cat(token_embeddings)

        self.document_logits, self.token_outputs = self.soft_attention_tokens(
            token_embeddings.to(self.device), token_attentions.to(self.device))

        nested_token_outputs = []
        for doc_id, doc_sent_boundaries in enumerate(sent_boundaries):
            doc_token_outputs = []
            for sent_id in range(len(doc_sent_boundaries) - 1):
                start_idx, end_idx = doc_sent_boundaries[sent_id], doc_sent_boundaries[sent_id + 1]
                doc_token_outputs.append(self.token_outputs[doc_id, start_idx:end_idx].detach().cpu().tolist())
            nested_token_outputs.append(doc_token_outputs)

        self.document_probs = self.document_probs_layer(self.document_logits)
        if self.config.num_labels == 1:
            document_preds = torch.round(self.document_probs[:, 0])
        else:
            document_preds = torch.argmax(self.document_probs, dim=1)

        return document_preds, nested_token_outputs

    def loss(self, document_targets, weights=None):
        assert len(set(document_targets.tolist())) <= 2  # only support binary for now

        # Calculate loss on the document-level prediction
        document_logits = self.document_logits

        if self.config.num_labels == 1:
            # MSE
            criterion = torch.nn.MSELoss()
            document_targets_loss = document_targets.to(torch.float32)[:, None]
            document_logits = self.document_probs
        elif self.config.num_labels == 2:
            criterion = torch.nn.BCEWithLogitsLoss(weight=weights)
            document_targets_loss = torch.nn.functional.one_hot(document_targets,
                                                                num_classes=self.config.num_labels).to(
                torch.float32).to(document_targets.device)
        else:
            criterion = torch.nn.CrossEntropyLoss(weight=weights)
            document_targets_loss = document_targets

        document_loss = criterion(document_logits, document_targets_loss)

        token_loss = self.soft_attention_tokens.loss(document_targets)

        return document_loss + self.config.token_loss_gamma * token_loss
