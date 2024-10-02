"""Endpoint for training related processes."""

from dbdie_classes.base import FullModelType
from dbdie_classes.options.FMT import from_fmts, PredictableTypes
from fastapi import APIRouter, status
from pydantic import BaseModel, field_validator

from backbone.code.extraction import (
    get_label_refs, get_raw_dataset, save_label_refs, split_and_save_dataset,
)
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


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


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_train(extr_config: TrainExtractor):
    fmts = list(extr_config.full_model_types.keys())
    mts, _, ifks = from_fmts(fmts)
    pred_types = PredictableTypes(fmts=fmts, mts=mts, ifks=ifks)

    ie = InfoExtractor(extr_config.id, extr_config.name)
    try:
        ie.init_extractor(
            {fmt: d.model_id for fmt, d in extr_config.full_model_types.items()},
            {fmt: d.total_classes for fmt, d in extr_config.full_model_types.items()},
        )

        raw_dataset = get_raw_dataset(pred_types, target_mckd=True)
        paths_dict = split_and_save_dataset(raw_dataset, pred_types, split_data=True)
        del raw_dataset

        label_refs = get_label_refs(pred_types)
        label_ref_paths = save_label_refs(label_refs)
        del label_refs

        ie.train(label_ref_paths, paths_dict["train"], paths_dict["val"])
        ie.save(f"extractors/{ie.name}")
    finally:
        ie.flush()
        del ie

    return status.HTTP_201_CREATED
