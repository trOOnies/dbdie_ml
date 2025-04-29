"""Training calls."""

from typing import TYPE_CHECKING

from dbdie_classes.groupings import PredictableTuples

from backbone.classes.training import TrainExtractor, TrainModel
from backbone.endpoints import getr

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType


def get_untrained_models(
    extr_config: TrainExtractor,
    pred_tuples: PredictableTuples,
    mask_pretrained: list[bool],
) -> dict["FullModelType", TrainModel]:
    """Get `TrainModels` for when not all models are pretrained."""
    i = getr("/models/count", api=True)

    models_cfgs = {}
    for m_num, (pt, is_pretrained) in enumerate(zip(pred_tuples, mask_pretrained)):
        if is_pretrained:
            models_cfgs[pt.fmt] = TrainModel.from_pretrained(id)
        else:
            tcs = getr(
                f"/{pt.mt}/filter-with-dbdvr/count",
                api=True,
                params={
                    "dbdv_min_id": extr_config.dbdv_min_id,
                    "dbdv_max_id": extr_config.dbdv_max_id,
                },
            )
            models_cfgs[pt.fmt] = TrainModel(
                id=i, name=f"m{m_num}-{extr_config.name}", fmt=pt.fmt,
                total_classes=tcs, cps_name=extr_config.cps_name,
            )
            i += 1

    return models_cfgs


# * Higher level functions


def process_extr_config(extr_config: TrainExtractor) -> tuple[PredictableTuples, list[bool]]:
    """Process the Extractor config."""
    fmts = list(extr_config.pretrained_models_ids.keys())
    pred_tuples = PredictableTuples.from_fmts(fmts)

    # Extractor setting
    extr_config.id = getr("/extractor/count", api=True)

    # Models setting
    mask_pretrained = [mid is not None for mid in extr_config.pretrained_models_ids.values()]

    return pred_tuples, mask_pretrained


def get_models_cfg(
    extr_config: TrainExtractor,
    pred_tuples: PredictableTuples,
    mask_pretrained: list[bool],
) -> dict["FullModelType", TrainModel]:
    """Get `TrainModels`."""
    if all(mask_pretrained):
        return {
            fmt: TrainModel.from_pretrained(id)
            for fmt, id in extr_config.pretrained_models_ids.items()
        }
    else:
        return get_untrained_models(extr_config, pred_tuples, mask_pretrained)
