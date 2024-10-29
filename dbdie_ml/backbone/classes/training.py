"""Training light classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pydantic import BaseModel, field_validator

from dbdie_classes.base import FullModelType
from dbdie_classes.schemas.helpers import DBDVersionRange


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
    id: int
    fmt: FullModelType
    total_classes: int
    cps_name: str

    @classmethod
    def from_model(cls, iem) -> TrainModel:
        return cls(
            id=iem.id,
            fmt=iem.fmt,
            total_classes=iem.total_classes,
            cps_name=iem.cps_name,
        )


class TrainExtractor(BaseModel):
    id: int
    name: str
    cps_name: str
    fmts: dict[FullModelType, TrainModel]
    stratify_fallback: bool  # TODO: change name to another one
    custom_dbdvr: DBDVersionRange | None  # TODO: Not implemented

    @field_validator("fmts")
    @classmethod
    def fmt_not_empty(cls, fmts: dict) -> dict[FullModelType, TrainModel]:
        assert fmts, "fmts cannot be empty."
        return fmts

    @property
    def models_ids(self) -> dict[FullModelType, int]:
        return {fmt: mcfg.id for fmt, mcfg in self.fmts.items()}
