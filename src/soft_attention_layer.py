import torch

from .config import Config


class SoftAttentionLayer(torch.nn.Module):
    """
    A module containing all components of the Soft Attention Architecture
    """

    def __init__(self, config: Config, lm_config):
        super().__init__()
        self.config = config
        self.lm_config = lm_config

        # Token-level attention
        self.soft_attention_dropout_layer = torch.nn.Dropout(p=self.config.soft_attention_dropout)
        self.soft_attention_evidence_layer = torch.nn.Linear(self.lm_config.hidden_size,
                                                             self.config.soft_attention_evidence_size)
        # used for predicting token-level scores
        self.soft_attention_scores_layer = torch.nn.Linear(self.config.soft_attention_evidence_size, 1)
        self.soft_attention_activation = torch.sigmoid

        # Compose to build Document-level representation
        self.document_repr_hidden_layer = torch.nn.Linear(self.lm_config.hidden_size,
                                                          self.config.soft_attention_hidden_size)
        self.document_preds_layer = torch.nn.Linear(self.config.soft_attention_hidden_size, self.config.num_labels)

        # Initialise weights
        self.__init_weights(self.soft_attention_dropout_layer)
        self.__init_weights(self.soft_attention_evidence_layer)
        self.__init_weights(self.soft_attention_scores_layer)
        self.__init_weights(self.document_repr_hidden_layer)
        self.__init_weights(self.document_preds_layer)

    def forward(self, transformer_token_outputs, attention_mask):
        """
        Apply the soft attention layer to the transformer token outputs (provided without CLS output)

        :param transformer_token_outputs: bxnxh output of transformer model
        :param attention_mask: bxn show which tokens to attend to
        :return: (sentence_logit, token_attention_scores)
        """
        self.transformers_attention_mask = attention_mask
        # calculate lengths of inputs, used for masking out later
        self.inp_lengths = (self.transformers_attention_mask != 0).sum(dim=1)

        transfomers_token_outputs_after_dropout = self.soft_attention_dropout_layer(transformer_token_outputs)

        # e_i = tanh(W_e * h_i + b_e)
        attention_evidence = torch.tanh(
            self.soft_attention_evidence_layer(transfomers_token_outputs_after_dropout))

        # \tilde{e_i} = W_{\tilde{e}}*e_i + b_{\tilde{e}}; \tilde{a_i} = \sigma{\tilde{e_i}}
        self.attention_scores = self.soft_attention_activation(
            self.soft_attention_scores_layer(attention_evidence).view(transformer_token_outputs.size()[:2]))

        # mask out scores after end of the input
        self.attention_scores = torch.where(
            self.__sequence_mask(self.inp_lengths, maxlen=self.transformers_attention_mask.shape[1]),
            self.attention_scores,
            torch.zeros_like(self.attention_scores)
        )

        # quadratic attention + a_i (normalise scores)
        attention_weights = torch.square(self.attention_scores)
        attention_weights = attention_weights / torch.sum(attention_weights, dim=1, keepdim=True)

        ###### Document Representation Building #####

        # apply attention to the post-dropout transformer token outputs
        post_attention_document_representation = torch.bmm(
            transfomers_token_outputs_after_dropout.transpose(1, 2), attention_weights.unsqueeze(2)
        ).squeeze(dim=2)

        # obtain the hidden layer from weighted representation
        document_hidden_layer_outputs = torch.tanh(
            self.document_repr_hidden_layer(post_attention_document_representation))

        # Obtain final document scores
        self.document_logits = self.document_preds_layer(document_hidden_layer_outputs)
        self.document_logits = self.document_logits.view(
            [transformer_token_outputs.shape[0], self.config.num_labels]
        )

        return self.document_logits, self.attention_scores

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

    def __sequence_mask(self, lengths, maxlen=None, dtype=torch.bool):
        """
        Provide a mask where all entries after length are 0
        :param lengths:
        :param maxlen:
        :param dtype:
        :return: torch.tensor(lengths.shape[0], maxlen)
        """
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1).to(next(self.parameters()).device)
        matrix = torch.unsqueeze(lengths, dim=-1)

        mask = row_vector < matrix

        mask.type(dtype)
        return mask

    def loss(self, document_targets):
        """
        :param document_targets: batch_size labels of the documents
        :return:
        """
        # encourage the model to focus on some, but not all tokens by optimising
        # minimum attention score to be close to 0

        min_attentions, _ = torch.min(
            torch.where(
                self.__sequence_mask(self.inp_lengths, maxlen=self.transformers_attention_mask.shape[1]),
                self.attention_scores,
                torch.zeros_like(self.attention_scores) + 1e6,
            ),
            dim=1,
        )
        l2 = torch.mean(torch.square(min_attentions.view(-1)))

        # encourage the model to have some positive (=1) token labels if overall document label is positive
        max_attentions, _ = torch.max(
            torch.where(
                self.__sequence_mask(self.inp_lengths, maxlen=self.transformers_attention_mask.shape[1]),
                self.attention_scores,
                torch.zeros_like(self.attention_scores) - 1e6,
            ),
            dim=1,
        )

        l3 = torch.mean(
            torch.square(max_attentions.view(-1) - document_targets.view(-1)))

        #### Positive sentences should have means closer to 1 than 0
        l4 = torch.mean(
            torch.square(torch.sum(self.attention_scores, dim=1) / self.inp_lengths - document_targets.view(-1)))

        # l2 regularisation
        # calculate mean squared values per document and average
        l5 = torch.mean(torch.sum(torch.square(self.attention_scores), dim=1) / self.inp_lengths)

        l6 = self.__top_k_loss(document_targets)

        return self.config.min_max_token_loss_gamma * (l2 + l3) \
               + self.config.mean_token_loss_gamma * l4 \
               + self.config.regularisation_loss_gamma * l5 \
               + self.config.top_k_token_loss_gamma * l6

    def __top_k_loss(self, document_targets):
        """
        Calculate loss based on top k% of tokens in the document
        Optimise top-k% to be close to the label of the document, while the rest close to 0.

        :param document_targets:
        :return:
        """

        ### Part 1: optimise top k to be the same as the document label

        # number of tokens to be optimised for each document
        k_vals = torch.round(self.inp_lengths * self.config.top_k_pct)

        # sort the tokens by max values
        sorted_attentions, _ = torch.sort(
            torch.where(
                self.__sequence_mask(self.inp_lengths, maxlen=self.transformers_attention_mask.shape[1]),
                self.attention_scores,
                torch.zeros_like(self.attention_scores) - 1e6,
            ),
            dim=1,
        )

        # only take top k tokens for each document, zero the rest out
        masked_attns = torch.where(
            self.__sequence_mask(k_vals, maxlen=self.transformers_attention_mask.shape[1]),
            torch.fliplr(sorted_attentions) - document_targets[:, None],  # flip to have highest labels first
            # want the difference to be 0 <> top attens be close to 0/1 dep. on the doc. label. This is MSE^
            torch.zeros_like(sorted_attentions)
        )

        loss = torch.mean(torch.square(masked_attns))  # average the mean sqaured error

        #### Part 2: optimise all other tokens
        k_vals = torch.round(self.inp_lengths * (1 - self.config.top_k_pct))

        # sort the tokens by max values (mask out tokens outside of the input)
        sorted_attentions, _ = torch.sort(
            torch.where(
                self.__sequence_mask(self.inp_lengths, maxlen=self.transformers_attention_mask.shape[1]),
                self.attention_scores,
                torch.zeros_like(self.attention_scores) + 1e6,
                # now add a large number to get only token-based atttentions first
            ),
            dim=1,
        )

        # zero out the rest
        masked_attns = torch.where(
            self.__sequence_mask(k_vals, maxlen=self.transformers_attention_mask.shape[1]),
            sorted_attentions,  # want the rest of attentions to be close to 0
            torch.zeros_like(sorted_attentions)
        )

        loss += torch.mean(torch.square(masked_attns))  # average the mean sqaured error

        return loss
