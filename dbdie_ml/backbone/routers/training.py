"""Endpoint for training related processes."""

from dbdie_classes.base import FullModelType
from dbdie_classes.options.FMT import from_fmts, PredictableTypes
from fastapi import APIRouter, status

from backbone.code.extraction import get_raw_dataset, save_label_refs, split_and_save_dataset
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_train(
    extr_name: str,
    fmts_with_counts: dict[FullModelType, int],
):
    fmts = list(fmts_with_counts.keys())
    mts, _, ifks = from_fmts(fmts)
    pred_types = PredictableTypes(fmts=fmts, mts=mts, ifks=ifks)

    ie = InfoExtractor(extr_name)
    try:
        ie.init_extractor(fmts_with_counts)
        raw_dataset, label_refs = get_raw_dataset(pred_types)
        train_paths, val_paths = split_and_save_dataset(raw_dataset, pred_types)
        label_ref_paths = save_label_refs(label_refs)
        ie.train(label_ref_paths, train_paths, val_paths)
        ie.save(f"extractors/{ie.name}")
    finally:
        ie.flush()
        del ie

    return status.HTTP_201_CREATED
