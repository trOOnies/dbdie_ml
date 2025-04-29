"""Training related functions."""

import datetime as dt
from typing import TYPE_CHECKING

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

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path
    from dbdie_classes.groupings import PredictableTuples

    from backbone.classes.training import TrainExtractor


def get_label_ref_paths(pred_tuples: "PredictableTuples") -> dict["FullModelType", "Path"]:
    label_refs = get_label_refs(pred_tuples)
    label_ref_paths = save_label_refs(label_refs)
    return label_ref_paths


def get_paths_dict(ie, pred_tuples: "PredictableTuples", extr_config: "TrainExtractor"):
    """Get paths dictionary for the training process."""
    matches = get_matches(ie)

    raw_dataset = get_raw_dataset(matches, pred_tuples, target_mckd=True)
    paths_dict = split_and_save_dataset(
        raw_dataset,
        pred_tuples,
        split_data=True,
        stratify_fallback=extr_config.stratify_fallback,
    )
    return paths_dict


def get_schemas_out(ie, extr_config: "TrainExtractor"):
    now = dt.datetime.now()
    today = dt.date.today().strftime("%Y-%m-%d")
    cps_id = CropperSwarm.load_metadata(extr_config.cps_name)["id"]

    ie_out = to_trained_ie_schema(ie, cps_id, now, today)
    models_out = to_trained_model_schemas(ie, cps_id, now, today)

    return ie_out, models_out
