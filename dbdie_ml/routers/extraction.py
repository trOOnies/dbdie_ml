"""Endpoint for extraction related processes."""

from copy import deepcopy
from dbdie_classes.base import FullModelType, PathToFolder
from dbdie_classes.options.PLAYER_TYPE import extract_mts_and_pts
from dbdie_ml.ml.extractor import InfoExtractor
from fastapi import APIRouter, status

from dbdie_ml.code.extraction import get_raw_dataset

router = APIRouter()


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_extract(
    ie_folder_path: PathToFolder,
    fmts: list[FullModelType] | None = None,
):
    """IMPORTANT. If doing partial uploads, please use 'fmts'."""
    ie = InfoExtractor.from_folder(ie_folder_path)

    try:
        fmts_ = deepcopy(fmts) if fmts is not None else ie.fmts
        mts, pts = extract_mts_and_pts(fmts_)

        raw_dataset = get_raw_dataset(fmts_, mts, pts)
        preds_dict = ie.predict_batch(raw_dataset, fmts_)
    finally:
        ie.flush()
        del ie

    return {
        "preds_dict": preds_dict,
        "match_ids": raw_dataset["match_id"].values,
        "player_ids": raw_dataset["player_id"].values,
    }