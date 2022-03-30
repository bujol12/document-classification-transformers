import torch

from transformers import AutoModel


class DocumentModel(torch.nn.Module):
    """
    Represent a document-level modelm with a given Transformer model
    as base and custom additional layers on top.
    """
    def __init__(self):
        super().__init__()

        # Base transformer
        self.transformer = AutoModel.from_pretrained(
            self.config_dict["model_name"],
            from_tf=bool(".ckpt" in self.config_dict["model_name"]),
            config=model_config,
        )

    def forward(self):
        outputs = self.language_model(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )  # last hidden states, pooler output, hidden states, attentions
