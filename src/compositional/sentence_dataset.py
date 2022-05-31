from typing import List, Dict, Tuple

import torch

from torch.utils.data import Dataset
from transformers import BatchEncoding

class SentenceDataset(Dataset):

    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx], 'attention_mask': self.attention_mask[idx]}, idx

    def __len__(self):
        return len(self.input_ids)

    @staticmethod
    def own_default_collator(features_with_idx: List[Tuple[Dict, int]]) -> Dict[str, torch.Tensor]:
        """
        Similar to HuggingFace default_data_collator, but does not have special handling for labels_ids etc
        Truncate input that's too long, but leave labels of original length
        Returns a Dictionary containing BATCHED inputs, each being a torch tensor
        Usually used for collating batches of flattened documents (no sentence-depth)
        :return:
        """
        features = [feat for feat, _ in features_with_idx]
        idx = [idx for _, idx in features_with_idx]

        if not isinstance(features[0], (dict, BatchEncoding)):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)
        if "label" in first and first["label"] is not None:
            label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
            dtype = torch.long if isinstance(label, int) else torch.float
            batch["label"] = torch.tensor([f["label"] for f in features], dtype=dtype)
        if "label_ids" in first and first["label_ids"] is not None:
            if isinstance(first["label_ids"], torch.Tensor):
                batch["label_ids"] = torch.stack([f["label_ids"] for f in features])
            else:
                dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
                batch["label_ids"] = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(f["label_ids"], dtype=dtype) for f in features], batch_first=True, padding_value=-100)

                # pad all label ids to be of the same length across the batch, -100 shows end of original label_ids
                # batch["label_ids"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which key/values are not None for this model.
        for k, v in first.items():
            if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                if k in ("word_labels"):  # word labels aren't of the same shape for each (different number of words)
                    batch[k] = [f[k] for f in features]
                elif not isinstance(v, str) and (not any([isinstance(e, str) for e in v]) if isinstance(v,
                                                                                                        list) else True):  # skip strings as those cannot be converted to tensors
                    batch[k] = torch.tensor([f[k] for f in features])

        batch["dataset_idx"] = torch.tensor(idx)
        return batch


