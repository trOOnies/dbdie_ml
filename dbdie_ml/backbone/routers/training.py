"""Endpoint for training related processes."""

from dbdie_classes.base import FullModelType
from dbdie_classes.groupings import PredictableTuples
from dbdie_classes.schemas.objects import ExtractorOut, ModelOut
from fastapi import APIRouter, status
import pandas as pd
import requests

from backbone.classes.training import TrainExtractor
from backbone.code.extraction import (
    get_label_refs,
    get_raw_dataset,
    save_label_refs,
    split_and_save_dataset,
)
from backbone.endpoints import bendp, parse_or_raise
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


@router.post(
    "/batch",
    status_code=status.HTTP_201_CREATED,
    response_model=dict[str, ExtractorOut | dict[FullModelType, ModelOut]],
)
def batch_train(extr_config: TrainExtractor):
    fmts = list(extr_config.fmts.keys())
    pred_tuples = PredictableTuples.from_fmts(fmts)

    ie = None
    try:
        ie = InfoExtractor.from_train_config(extr_config)

        matches = parse_or_raise(
            requests.get(bendp("/matches"), params={"limit": 300_000})
        )
        matches = ie.filter_matches_with_dbdv(matches)
        assert matches, "No matches intersect with the extractor's DBDVersionRange."
        matches = pd.DataFrame(
                [
                {"match_id": m["id"], "filename": m["filename"]}
                for m in matches
            ]
        )

        raw_dataset = get_raw_dataset(matches, pred_tuples, target_mckd=True)
        paths_dict = split_and_save_dataset(raw_dataset, pred_tuples, split_data=True)
        del raw_dataset

        label_refs = get_label_refs(pred_tuples)
        label_ref_paths = save_label_refs(label_refs)
        del label_refs

        ie.train(label_ref_paths, paths_dict["train"], paths_dict["val"])
        ie.save(f"extractors/{ie.name}")

        ie_out = ie.to_schema()
        models_out = ie.models_to_schemas()
    finally:
        if ie is not None:
            ie.flush()
        del ie

    return {
        "extractor": ie_out,
        "models": models_out,
    }
