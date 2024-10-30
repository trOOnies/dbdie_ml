"""Extra code for the '/extraction' endpoint."""

import numpy as np
import pandas as pd
from typing import TYPE_CHECKING

from backbone.code.extraction_funcs import (
    custom_tv_split,
    data_to_dfs,
    filter_ifk,
    filter_mckd,
    get_mt_data,
    get_paths_dict,
    get_raw_data,
    merge_and_sort_data,
    parse_data,
    processes_presplit,
)

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path
    from dbdie_classes.groupings import PredictableTuples


def get_raw_dataset(
    matches: pd.DataFrame,
    pred_tuples: "PredictableTuples",
    target_mckd: bool,
) -> pd.DataFrame:
    """Get raw dataset for an `InfoExtractor`."""
    raw_data = get_raw_data(matches)
    data = parse_data(raw_data, pred_tuples.mts)

    data = filter_ifk(data, pred_tuples.ifks)
    data = filter_mckd(data, pred_tuples.mts, target_mckd)

    return merge_and_sort_data(data, matches, target_mckd)


def split_and_save_dataset(
    data: pd.DataFrame,
    pred_tuples: "PredictableTuples",
    split_data: bool,
    stratify_fallback: bool = False,
) -> dict[str, dict["FullModelType", "Path"]]:
    """Split (optionally) and save dataset."""
    paths_dict = get_paths_dict(pred_tuples.fmts, split_data)

    for ptup in pred_tuples:
        print(f"Splitting {ptup.fmt}...")
        split = processes_presplit(data, ptup, split_data)

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
    data = get_mt_data(pred_tuples)
    data = data_to_dfs(data)
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
