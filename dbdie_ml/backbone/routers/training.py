"""Endpoint for training related processes."""

import datetime as dt
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from shutil import rmtree
from traceback import print_exc

from dbdie_classes.base import FullModelType
from dbdie_classes.groupings import PredictableTuples
from dbdie_classes.schemas.objects import ExtractorOut, ModelOut

from backbone.classes.training import TrainExtractor, TrainModel
from backbone.code.extraction import (
    get_label_refs,
    get_raw_dataset,
    save_label_refs,
    split_and_save_dataset,
)
from backbone.code.routers.training import (
    get_matches,
    to_trained_ie_schema,
    to_trained_model_schemas,
)
from backbone.cropping import CropperSwarm
from backbone.endpoints import getr
from backbone.ml.extractor import InfoExtractor

router = APIRouter()


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=dict[str, ExtractorOut | dict[FullModelType, ModelOut]],
)
def batch_train(extr_config: TrainExtractor):
    """Batch train an `InfoExtractor`."""
    fmts = list(extr_config.pretrained_models_ids.keys())
    pred_tuples = PredictableTuples.from_fmts(fmts)

    # Extractor setting
    extr_config.id = getr("/extractor/count", api=True)

    # Models setting
    mask_pretrained = [mid is not None for mid in extr_config.pretrained_models_ids.values()]
    total_non_pretrained = len(mask_pretrained) - sum(mask_pretrained)
    if total_non_pretrained == 0:
        models_cfgs = {
            fmt: TrainModel.from_pretrained(id)
            for fmt, id in extr_config.pretrained_models_ids.items()
        }
    else:
        i = getr("/models/count", api=True)
        models_cfgs = {}
        for fmt, is_pretrained in zip(fmts, mask_pretrained):
            if is_pretrained:
                models_cfgs[fmt] = TrainModel.from_pretrained(id)
            else:
                models_cfgs[fmt] = TrainModel.from_untrained(i, fmt)
                i += 1

    ie = None
    try:
        ie = InfoExtractor.from_train_config(extr_config, models_cfgs)
        matches = get_matches(ie)

        raw_dataset = get_raw_dataset(matches, pred_tuples, target_mckd=True)
        paths_dict = split_and_save_dataset(
            raw_dataset,
            pred_tuples,
            split_data=True,
            stratify_fallback=extr_config.stratify_fallback,
        )
        del raw_dataset

        label_refs = get_label_refs(pred_tuples)
        label_ref_paths = save_label_refs(label_refs)
        del label_refs

        ie.train(label_ref_paths, paths_dict["train"], paths_dict["val"])
        ie.save()

        now = dt.datetime.now()
        today = dt.date.today().strftime("%Y-%m-%d")
        cps_id = CropperSwarm.load_metadata(extr_config.cps_name)["id"]

        ie_out = to_trained_ie_schema(ie, cps_id, now, today)
        models_out = to_trained_model_schemas(ie, cps_id, now, today)
    except Exception as e:
        print_exc()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    finally:
        if ie is not None:
            ie.flush()
        del ie

    return {
        "extractor": ie_out,
        "models": models_out,
    }


@router.delete("")
def delete_extractor(extr_name: str, delete_models: bool):
    if not delete_models:
        raise NotImplementedError  # TODO
    rmtree(f"extractors/{extr_name}")
