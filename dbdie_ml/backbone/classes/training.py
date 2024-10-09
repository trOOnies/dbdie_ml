"""Training light classes."""

from dataclasses import asdict, dataclass
from dbdie_classes.base import FullModelType
from dbdie_classes.version import DBDVersionRange
from pydantic import BaseModel, field_validator
from typing import Any


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
    fmt: FullModelType
    total_classes: int
    cps_name: str
    trained_model: Any | None


class TrainExtractor(BaseModel):
    id: int
    name: str
    cps_name: str
    fmts: dict[FullModelType, TrainModel]
    custom_dbdvr: DBDVersionRange | None

    @field_validator("fmts")
    @classmethod
    def fmt_not_empty(cls, fmts: dict) -> dict[FullModelType, TrainModel]:
        assert fmts, "fmts cannot be empty."
        return fmts
