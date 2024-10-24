"""Endpoint for extraction related processes."""

from copy import deepcopy
from dbdie_classes.groupings import PredictableTuples
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from traceback import print_exc

from backbone.code.extraction import get_raw_dataset, split_and_save_dataset
from backbone.code.routers.training import get_matches
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_extract(
    extr_name: str,
    # fmts: list[FullModelType] | None = None,  # TODO
):
    """IMPORTANT. If doing partial uploads, please use 'fmts'."""
    ie = InfoExtractor.from_folder(extr_name)

    try:
        fmts_ = deepcopy(ie.fmts)  # TODO
        pred_tuples = PredictableTuples.from_fmts(fmts_)

        matches = get_matches(ie)

        raw_dataset = get_raw_dataset(matches, pred_tuples, target_mckd=False)
        paths_dict = split_and_save_dataset(raw_dataset, pred_tuples, split_data=False)

        preds_dict = ie.predict_batch(paths_dict["pred"], fmts_, probas=False)
        preds_dict = {k: {k2: a.tolist() for k2, a in d.items()} for k, d in preds_dict.items()}
    except Exception as e:
        print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    finally:
        ie.flush()
        del ie

    return preds_dict
