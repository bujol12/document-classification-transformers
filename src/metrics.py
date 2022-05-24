import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score


from .json_document_dataset import JsonDocumentDataset

class Metrics:
    token_acc = None
    token_map = None
    token_f1 = None
    token_p = None
    token_r = None

    def __init__(self, dataset: JsonDocumentDataset, loss):
        """
            Compute metrics for the (flattened) given predictions
            :param dataset:
            :param loss:
            :return: metric class
        """
        self.y_preds = torch.tensor(dataset.tokenised_input["pred"])
        y_true = torch.tensor(dataset.tokenised_input["label"])

        self.num_labels = self.y_preds.shape[-1]

        if self.num_labels == 1:
            # single values
            self.y_true = y_true.to(torch.float32)
            y_pred_labels = torch.round(self.y_preds[:, 0])
        else:
            self.y_true = y_true
            y_pred_labels = self.y_preds

        # document-level metrics
        self.acc = accuracy_score(self.y_true, y_pred_labels)
        self.f1 = f1_score(self.y_true, y_pred_labels)
        self.p = precision_score(self.y_true, y_pred_labels)
        self.r = recall_score(self.y_true, y_pred_labels)

        # loss
        self.loss = loss

        # token level, if present
        token_preds = dataset.tokenised_input["token_preds"]
        token_true = dataset.tokenised_input["label_ids"]

        if token_preds != [] and token_true != []:
            # pad all true tokens to be of the same length (with -100)
            token_true = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor(labels) for labels in token_true], batch_first=True, padding_value=-100)

            token_preds = torch.tensor(token_preds)

            # always 0-1 only, padding tokens and/or split ones have labels set to -100 while processing dataset
            self.token_true = token_true.view(-1)

            if token_true.shape[-1] - token_preds.shape[-1] > 0:
                # pad with 0s for when the sequence length was longer than the maximum model length
                token_preds = torch.nn.functional.pad(token_preds,
                                                      (0, token_true.shape[-1] - token_preds.shape[-1]))

            self.token_preds = token_preds.view(-1)

            assert len(self.token_true) == len(self.token_preds)

            # ignore tokens with -100
            token_true_lst = []
            token_preds_lst = []
            for i in range(len(self.token_true)):
                if self.token_preds[i] == -100 or self.token_true[i] == -100:
                    continue
                token_true_lst.append(self.token_true[i].item())
                token_preds_lst.append(self.token_preds[i].item())

            token_pred_labels = torch.round(torch.tensor(token_preds_lst))

            self.token_acc = accuracy_score(token_true_lst, token_pred_labels)
            self.token_map = self.__get_map(token_true, token_preds)
            self.token_f1 = f1_score(token_true_lst, token_pred_labels)
            self.token_p = precision_score(token_true_lst, token_pred_labels)
            self.token_r = recall_score(token_true_lst, token_pred_labels)

    def to_json(self):
        return {'loss': self.loss, 'doc_acc': self.acc, 'doc_f1': self.f1, 'doc_p': self.p, 'doc_r': self.r,
                'tok_acc': self.token_acc, 'tok_f1': self.token_f1, 'tok_p': self.token_p, 'tok_r': self.token_r,
                'tok_map': self.token_map}

    def __get_map(self, token_true, token_preds):
        """
        Calculate mean average precision on 2-D predictions (batch_size x pred_size)
        :param token_true: true labels (batch_size x pred_size)
        :param token_preds: predictions
        :return: MAP score
        """
        map_sum = 0.0
        cnt = 0

        for sample_idx in range(len(token_true)):
            true = []
            preds = []

            # skip all -100s
            for idx in range(len(token_true[sample_idx])):
                if token_true[sample_idx][idx] == -100 or token_preds[sample_idx][idx] == -100:
                    # skip tokens to be ignored
                    continue
                true.append(token_true[sample_idx][idx])
                preds.append(token_preds[sample_idx][idx])
            if max(true) > 0.0:
                map_sum += average_precision_score(true, preds)
                cnt += 1

        return map_sum / cnt
