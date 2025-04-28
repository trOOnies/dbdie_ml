"""Training light classes."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from dbdie_classes.base import FullModelType
from dbdie_classes.options import FMT
from pydantic import BaseModel, field_validator
from yaml import safe_load

from backbone.classes.register import get_cropper_swarm_mpath, get_model_mpath


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
    name: str
    fmt: FullModelType
    total_classes: int
    cps_name: str
    pretrained: bool = False

    @classmethod
    def from_pretrained(cls, id: int) -> TrainModel:
        mpath = get_model_mpath(id=id)
        with open(mpath, "r") as f:
            metadata = safe_load(f)
        metadata["pretrained"] = True
        return cls(**metadata)


class TrainExtractor(BaseModel):
    name: str
    cps_name: str
    stratify_fallback: bool  # TODO: change name to another one
    pretrained_models_ids: dict[FullModelType, int | None] | None = None
    dbdv_min_id: int | None  # ! NOT MEANT TO BE SET
    dbdv_max_id: int | None  # ! NOT MEANT TO BE SET
    id: int | None = None  # ! NOT MEANT TO BE SET

    @field_validator("pretrained_models_ids")
    @classmethod
    def fmt_not_empty(
        cls,
        pretrained_models_ids: dict,
    ) -> dict[FullModelType, int | None]:
        models_ids = {fmt: None for fmt in sorted(FMT.KILLER + FMT.SURV)}
        if pretrained_models_ids is not None:
            assert pretrained_models_ids, "Pretrained models ids cannot be empty."
            models_ids = models_ids | pretrained_models_ids
        return models_ids

    @field_validator("dbdv_min_id")
    @classmethod
    def set_dbdv_min_id(
        cls,
        dbdv_min_id: int | None,
        _context,
    ) -> int:
        assert dbdv_min_id is None
        mpath = get_cropper_swarm_mpath(_context.data["cps_name"])
        with open(mpath, "r") as f:
            cps_metadata = safe_load(f)
        return cps_metadata["dbdv_min_id"]

    @field_validator("dbdv_max_id")
    @classmethod
    def set_dbdv_max_id(
        cls,
        dbdv_max_id: int | None,
        _context,
    ) -> int | None:
        assert dbdv_max_id is None
        mpath = get_cropper_swarm_mpath(_context.data["cps_name"])
        with open(mpath, "r") as f:
            cps_metadata = safe_load(f)
        return cps_metadata["dbdv_max_id"]
