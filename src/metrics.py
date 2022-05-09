import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score


class Metrics:
    token_acc = None
    token_map = None
    token_f1 = None
    token_p = None
    token_r = None

    def __init__(self, y_preds: torch.tensor, y_true: torch.tensor, loss, token_preds: torch.tensor = None,
                 token_true: torch.tensor = None):
        """
            Compute metrics for the (flattened) given predictions
            :param preds: n_samples x n_classes np array containing probabilities for each sample of each class
            :param labels: n_samples np array
            :param loss:
            :param token_preds: n_samples x n_classes
            :param token_true: n_samples x length
            :return: metric class
        """
        self.y_preds = y_preds
        self.num_labels = self.y_preds.shape[1]

        if self.num_labels == 1:
            # single values
            self.y_true = y_true.to(torch.float32)
            y_true_probs = self.y_true
            self.y_preds = torch.nn.Sigmoid()(self.y_preds)
            y_pred_labels = torch.round(self.y_preds[:, 0])
        else:
            self.y_true = y_true
            y_true_probs = F.one_hot(self.y_true, num_classes=self.num_labels)
            y_pred_labels = torch.argmax(self.y_preds, dim=1)

        # document-level metrics
        self.acc = accuracy_score(self.y_true, y_pred_labels)
        self.f1 = f1_score(self.y_true, y_pred_labels)
        self.p = precision_score(self.y_true, y_pred_labels)
        self.r = recall_score(self.y_true, y_pred_labels)

        # loss
        self.loss = loss

        # token level, if present
        if token_preds is not None and token_true is not None:
            # always 0-1 only, padding tokens and/or split ones have labels set to -100 while processing dataset
            self.token_true = token_true.view(-1)

            # pad with 0s for when the sequence length was longer than the maximum model length
            self.token_preds = torch.nn.functional.pad(token_preds,
                                                       (0, token_true.shape[-1] - token_preds.shape[-1])).view(-1)

            # ignore tokens with -100
            assert len(self.token_true) == len(self.token_preds)
            token_true = []
            token_preds = []
            for i in range(len(self.token_true)):
                if self.token_preds[i] == -100 or self.token_true[i] == -100:
                    continue
                token_true.append(self.token_true[i].item())
                token_preds.append(self.token_preds[i].item())

            print(token_preds)
            token_pred_labels = torch.round(torch.tensor(token_preds))

            self.token_acc = accuracy_score(token_true, token_pred_labels)
            self.token_map = average_precision_score(token_true, token_preds)
            self.token_f1 = f1_score(token_true, token_pred_labels)
            self.token_p = precision_score(token_true, token_pred_labels)
            self.token_r = recall_score(token_true, token_pred_labels)

    def to_json(self):
        return {'loss': self.loss, 'doc_acc': self.acc, 'doc_f1': self.f1, 'doc_p': self.p, 'doc_r': self.r,
                'tok_acc': self.token_acc, 'tok_f1': self.token_f1, 'tok_p': self.token_p, 'tok_r': self.token_r,
                'tok_map': self.token_map}
