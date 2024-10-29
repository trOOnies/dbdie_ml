"""Extra code for the '/extraction' endpoint."""

import numpy as np
import pandas as pd
import requests
from typing import TYPE_CHECKING

from backbone.code.extraction_funcs import (
    apply_ifk_filter,
    apply_mckd_filter,
    custom_tv_split,
    filter_ifk,
    filter_mckd,
    get_relevant_cols,
    parse_data,
    process_label_ids,
    suffix_filenames,
)
from backbone.endpoints import bendp, getr, parse_or_raise

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path
    from dbdie_classes.groupings import PredictableTuples


def get_raw_dataset(
    matches: pd.DataFrame,
    pred_tuples: "PredictableTuples",
    target_mckd: bool,
) -> pd.DataFrame:
    """Get raw dataset for an `InfoExtractor`."""
    raw_data = parse_or_raise(
        requests.post(
            bendp("/labels/filter-many"),
            params={"ifk": None, "limit": 300_000, "skip": 0},
            json=None,
        ),
        exp_status_code=200,  # OK
    )
    raw_data = [m for m in raw_data if m["match_id"] in matches["match_id"].values]
    assert raw_data, "No labels of selected matches were found."

    data = parse_data(raw_data, pred_tuples.mts)

    data = filter_ifk(data, pred_tuples.ifks)
    data = filter_mckd(data, pred_tuples.mts, target_mckd)

    data = data.merge(matches, how="inner", on="match_id")
    assert not data.empty, (
        f"No {'labeled' if target_mckd else 'unlabeled'} match was found in the 'matches' table."
    )

    return data.sort_values(["match_id", "player_id"], ignore_index=True)


def split_and_save_dataset(
    data: pd.DataFrame,
    pred_tuples: "PredictableTuples",
    split_data: bool,
    stratify_fallback: bool = False,
) -> dict[str, dict["FullModelType", "Path"]]:
    """Split (optionally) and save dataset."""
    splits_fd = "dbdie_ml/backbone/cache/splits"

    if split_data:
        paths_dict = {
            "train": {fmt: f"{splits_fd}/{fmt}_train.csv" for fmt in pred_tuples.fmts},
            "val":   {fmt: f"{splits_fd}/{fmt}_val.csv"   for fmt in pred_tuples.fmts},
        }
    else:
        paths_dict = {
            "pred": {fmt: f"{splits_fd}/{fmt}_pred.csv" for fmt in pred_tuples.fmts},
        }

    for ptup in pred_tuples:
        print(f"Splitting {ptup.fmt}...")
        split = get_relevant_cols(data, ptup.mt)

        split = apply_mckd_filter(split, ptup.mt, training=split_data)
        split = apply_ifk_filter(split, ptup.ifk, training=split_data)

        split = process_label_ids(split, ptup.mt, training=split_data)
        split = suffix_filenames(split, training=split_data)

        if split_data:
            t_split, v_split = custom_tv_split(
                split,
                ptup.fmt,
                stratify_fallback=stratify_fallback,
            )
            del split

            t_split.to_csv(paths_dict["train"][ptup.fmt], index=False)
            v_split.to_csv(paths_dict["val"][ptup.fmt], index=False)
            del t_split, v_split
        else:
            split.to_csv(paths_dict["pred"][ptup.fmt], index=False)
            del split

    return paths_dict


def get_label_refs(
    pred_tuples: "PredictableTuples"
) -> dict["FullModelType", pd.DataFrame]:
    """Get labels reference for selected mts."""
    data = {
        ptup.fmt: getr(
            f"/{ptup.mt}",
            api=True, params={"limit": 300_000, "ifk": ptup.ifk}
        )
        for ptup in pred_tuples
    }
    assert all(len(ds) for ds in data.values()), "ModelType data is empty."
    data = {
        fmt: pd.DataFrame([(d["id"], d["name"]) for d in ds], columns=["id", "name"])
        for fmt, ds in data.items()
    }
    data = {
        fmt: df.sort_values("id", ignore_index=True)
        for fmt, df in data.items()
    }
    for fmt in data:
        data[fmt]["net_id"] = np.arange(data[fmt].shape[0], dtype=int)
    return data


def save_label_refs(
    label_refs: dict["FullModelType", pd.DataFrame],
) -> dict["FullModelType", "Path"]:
    """Save labels reference."""
    lref_fd = "dbdie_ml/backbone/cache/label_ref"
    paths = {
        fmt: f"{lref_fd}/{fmt}.csv"
        for fmt in label_refs
    }

    for fmt, lref in label_refs.items():
        lref.to_csv(paths[fmt], index=False)

    return paths
