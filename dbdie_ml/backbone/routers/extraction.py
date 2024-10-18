"""Endpoint for extraction related processes."""

from copy import deepcopy
from dbdie_classes.groupings import PredictableTuples
from dbdie_classes.options.FMT import from_fmts
from fastapi import APIRouter, status

from backbone.code.extraction import get_raw_dataset, split_and_save_dataset
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_extract(
    ie_name: str,
    # fmts: list[FullModelType] | None = None,  # TODO
):
    """IMPORTANT. If doing partial uploads, please use 'fmts'."""
    ie = InfoExtractor.from_folder(ie_name)

    fmts_ = deepcopy(ie.fmts)  # TODO
    mts, _, ifks = from_fmts(fmts_)
    pred_tuples = PredictableTuples.from_lists(fmts_, mts, ifks)

    try:
        raw_dataset = get_raw_dataset(pred_tuples, target_mckd=False)
        paths_dict = split_and_save_dataset(raw_dataset, pred_tuples, split_data=False)

        preds_dict = ie.predict_batch(paths_dict["pred"], fmts_)
        print("PREDS DICT:")
        print(preds_dict)
        raise NotImplementedError
    finally:
        ie.flush()
        del ie

    return {
        "preds_dict": preds_dict,
        "match_ids": raw_dataset["match_id"].values,
        "player_ids": raw_dataset["player_id"].values,
    }
