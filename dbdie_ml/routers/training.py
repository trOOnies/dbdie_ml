"""Endpoint for training related processes."""

from dbdie_classes.base import FullModelType
from dbdie_classes.options import PLAYER_TYPE as PT
from dbdie_ml.ml.extractor import InfoExtractor
from fastapi import APIRouter, status

from dbdie_ml.code.extraction import get_raw_dataset

router = APIRouter()


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_train(
    extr_name: str,
    fmts_with_counts: dict[FullModelType, int],
):
    fmts = list(fmts_with_counts.keys())
    mts, pts = PT.extract_mts_and_pts(fmts)

    ie = InfoExtractor(extr_name)
    try:
        ie.init_extractor(fmts_with_counts)
        raw_dataset = get_raw_dataset(fmts, mts, pts)
        preds_dict = ie.train(..., ..., ...)  # TODO
    finally:
        ie.flush()
        del ie

    return status.HTTP_201_CREATED
