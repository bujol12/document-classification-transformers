from dataclasses import dataclass


@dataclass
class Config:
    model_name: str
    infer_labels: bool = False
