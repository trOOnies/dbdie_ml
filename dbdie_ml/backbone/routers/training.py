"""Endpoint for training related processes."""

from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from shutil import rmtree
from traceback import print_exc

from dbdie_classes.base import FullModelType
from dbdie_classes.schemas.objects import ExtractorOut, ModelOut

from backbone.classes.register import safe_pathing
from backbone.classes.training import TrainExtractor
from backbone.ml.extractor import InfoExtractor
from backbone.training.calls import get_models_cfg, process_extr_config
from backbone.training.functions import get_label_ref_paths, get_paths_dict, get_schemas_out

router = APIRouter()


@router.post(
    "",
    status_code=status.HTTP_201_CREATED,
    response_model=dict[str, ExtractorOut | dict[FullModelType, ModelOut]],
)
def batch_train(extr_config: TrainExtractor):
    """Batch train an `InfoExtractor`."""
    pred_tuples, mask_pretrained = process_extr_config(extr_config)
    models_cfgs = get_models_cfg(extr_config, pred_tuples, mask_pretrained)

    label_ref_paths = get_label_ref_paths(pred_tuples)

    ie = None
    try:
        ie = InfoExtractor.from_train_config(extr_config, models_cfgs)
        paths_dict = get_paths_dict(ie, pred_tuples, extr_config)

        ie.train(label_ref_paths, paths_dict["train"], paths_dict["val"])
        ie.save()

        ie_out, models_out = get_schemas_out(ie, extr_config)
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
    safe_pathing(extr_name)
    if delete_models:
        raise NotImplementedError  # TODO
    rmtree(f"extractors/{extr_name}")
