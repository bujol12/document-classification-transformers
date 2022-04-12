import torch
import torch.nn.functional as F

from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, accuracy_score


class Metrics:
    def __init__(self, y_preds: torch.tensor, y_true: torch.tensor, loss):
        """
            Compute metrics for the (flattened) given predictions
            :param preds: n_samples x n_classes np array containing probabilities for each sample of each class
            :param labels: n_samples np array
            :return: metric class
        """
        self.y_preds = y_preds
        self.y_true = y_true
        print(self.y_preds)
        print(self.y_true)
        y_true_probs = F.one_hot(self.y_true)

        self.acc = accuracy_score(self.y_true, torch.argmax(self.y_preds, dim=1))
        self.map = average_precision_score(y_true_probs, self.y_preds)
        self.f1 = f1_score(self.y_true, torch.argmax(self.y_preds, dim=1))
        self.p = precision_score(self.y_true, torch.argmax(self.y_preds, dim=1))
        self.r = recall_score(self.y_true, torch.argmax(self.y_preds, dim=1))
        self.loss = loss

    def to_json(self):
        return {'loss': self.loss, 'acc': self.acc, 'map': self.map, 'f1': self.f1, 'p': self.p, 'r': self.r}
