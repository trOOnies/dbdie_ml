"""Training light classes."""

from dataclasses import asdict, dataclass
from dbdie_classes.base import FullModelType
from pydantic import BaseModel, field_validator


@dataclass
class TrainingParams:
    """Helper dataclass for training params."""
    epochs: int
    batch_size: int
    adam_lr: float

    def __post_init__(self):
        self.epochs = int(self.epochs)
        self.batch_size = int(self.batch_size)
        self.adam_lr = float(self.adam_lr)

    def typed_dict(self) -> dict[str, int | float]:
        return {k: v for k, v in asdict(self).items()}

    def dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}


class TrainModel(BaseModel):
    model_id: int
    total_classes: int


class TrainExtractor(BaseModel):
    id: int
    name: str
    full_model_types: dict[FullModelType, TrainModel]

    @field_validator("full_model_types")
    @classmethod
    def fmt_not_empty(cls, fmt: dict) -> dict[FullModelType, TrainModel]:
        assert fmt, "full_model_types cannot be empty."
        return fmt
