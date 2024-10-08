"""Endpoint for training related processes."""

from dbdie_classes.groupings import PredictableTuples
from fastapi import APIRouter, status
import pandas as pd
import requests

from backbone.classes.training import TrainExtractor
from backbone.code.extraction import (
    get_label_refs, get_raw_dataset, save_label_refs, split_and_save_dataset,
)
from backbone.endpoints import bendp, parse_or_raise
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_train(extr_config: TrainExtractor):
    fmts = list(extr_config.full_model_types.keys())
    pred_tuples = PredictableTuples.from_fmts(fmts)

    ie = None
    try:
        ie = InfoExtractor.from_train_config(extr_config)
        ie.dbdv_min_id = 306  # TODO: parametrize and put in instantiation
        ie.dbdv_max_id = 327  # TODO: parametrize and put in instantiation
        ie.cropper_swarm_id = 1  # TODO: parametrize and put in instantiation

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
    finally:
        if ie is not None:
            ie.flush()
        del ie

    return status.HTTP_201_CREATED
