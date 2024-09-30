"""Extra code for the '/extraction' endpoint."""

from dbdie_classes.options.MODEL_TYPE import (
    ADDONS,
    CHARACTER,
    MULTIPLE_PER_PLAYER,
    PERKS,
    TO_ID_NAMES,
)
import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split
from typing import TYPE_CHECKING

from backbone.endpoints import bendp, parse_or_raise

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, IsForKiller, ModelType, Path
    from dbdie_classes.options.FMT import PredictableTypes


def parse_data(
    data: list[dict],
    mts: list["ModelType"],
) -> pd.DataFrame:
    id_names = {mt: idn for mt, idn in TO_ID_NAMES.items() if mt in mts}
    data = [
        {
            "match_id": d["match_id"],
            "player_id": d["player"]["id"],
            "is_killer": d["player"]["id"] == 4,
        }
        | {idn: d["player"][idn] for idn in id_names.values()}
        | {
            f"{mt}_mckd": v
            for mt, v in d["manual_checks"]["predictables"].items()
            if mt in mts
        }
        for d in data
        if d["manual_checks"]["in_progress"]
    ]
    assert data, "No usable label was found in general."

    return pd.DataFrame(data)


def get_split(data: pd.DataFrame, mt: "ModelType") -> pd.DataFrame:
    split = data.copy()
    return split[
        [
            "match_id",
            "player_id",
            "is_killer",
            TO_ID_NAMES[mt],
            f"{mt}_mckd",
            "filename",
        ]
    ]


def apply_ifk_filter(data: pd.DataFrame, ifk: bool) -> pd.DataFrame:
    mask = data["is_killer"].values.copy()
    if not ifk:
        mask = np.logical_not(mask)
    return data[mask]


def filter_ifk(
    data: pd.DataFrame,
    ifks: list["IsForKiller"],
) -> pd.DataFrame:
    """Filter is_killer if all PTs are not None and identical (i.e. a single PT)."""
    if all(ifk is not None for ifk in ifks):
        unique_ifks = list(set(ifks))
        if len(unique_ifks) == 1:
            ifk = unique_ifks[0]
            data = apply_ifk_filter(data, ifk)
            assert not data.empty, "No ifk related data."
    return data


def filter_mckd(
    data: pd.DataFrame,
    mts: list["ModelType"],
) -> pd.DataFrame:
    """Filter manual checks."""
    mckd_cols = [f"{mt}_mckd" for mt in mts]
    for col in mckd_cols:
        data[col] = data[col].fillna(False)
    data = data[data[mckd_cols].any(axis=1)]
    assert not data.empty, "No usable label was found for this specific extractor."
    return data


def get_matches() -> pd.DataFrame:
    """Get matches' information."""
    matches = parse_or_raise(requests.get(bendp("/matches"), params={"limit": 300_000}))
    assert matches, "No matches could be retrieved."
    matches = [
        {"match_id": m["id"], "filename": m["filename"]}
        for m in matches
    ]
    return pd.DataFrame(matches)


def get_label_refs(pred_types: "PredictableTypes") -> dict["FullModelType", pd.DataFrame]:
    """Get labels reference for selected mts."""
    data = {
        fmt: parse_or_raise(
            requests.get(
                bendp(f"/{mt}"),
                params={
                    "limit": 300_000,
                    ("is_killer" if mt == CHARACTER else "ifk"): ifk
                },
            )
        )
        for fmt, mt, ifk in pred_types
    }
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


def flatten_multiple_mt(split: pd.DataFrame, mt: "ModelType") -> pd.DataFrame:
    if mt == PERKS:
        total_ids = 4
    elif mt == ADDONS:
        total_ids = 2
    else:
        raise NotImplementedError

    split = [split.copy() for _ in range(total_ids)]
    for i in range(total_ids):
        split[i].loc[:, "item_id"] = i
        split[i]["label_id"] = split[i]["label_id"].map(lambda vs: vs[i])

    split = pd.concat(split, axis=0)
    split = split[["match_id", "player_id", "item_id", "label_id", "filename"]]

    return split.sort_values(["match_id", "player_id", "item_id"], ignore_index=True)


def process_label_ids(split: pd.DataFrame, mt: "ModelType") -> pd.DataFrame:
    split = split.rename({TO_ID_NAMES[mt]: "label_id"}, axis=1)
    split = (
        flatten_multiple_mt(split, mt)
        if mt in MULTIPLE_PER_PLAYER
        else split.astype({"label_id": int})
    )
    return split.reset_index(drop=True)


def suffix_filenames(split: pd.DataFrame) -> pd.DataFrame:
    has_item_id = "item_id" in split.columns.values
    split["filename"] = split.apply(
        lambda row: f"{row['filename'][:-4]}_{row['player_id']}_",
        axis=1,
    )
    if has_item_id:
        split["filename"] = (
            split["filename"]
            + split["item_id"].astype(int).map(lambda v: f"{v}.jpg")
        )
    else:
        split["filename"] = split["filename"] + "0.jpg"
    return split


def custom_tv_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Custom train-validation split that takes into account that values
    that aren't repeated in the data should always go to the validation split.
    """
    unique_vals = data["label_id"].value_counts()

    val_size_pc = 0.20
    val_size = int(val_size_pc * data.shape[0])
    random_state = 42

    unique_vals = unique_vals[unique_vals.values == 1]
    if unique_vals.empty:
        df_train, df_val = train_test_split(
            data,
            test_size=val_size,
            random_state=random_state,
            stratify=data["label_id"],
        )
        return (
            df_train.sample(frac=1, random_state=random_state, ignore_index=True),
            df_val.sample(frac=1, random_state=random_state, ignore_index=True),
        )

    unique_vals = unique_vals.index.values

    mask = data["label_id"].isin(unique_vals)
    unique_data = data[mask]
    unique_data_count = unique_data.shape[0]
    target_non_unique_val = val_size - unique_data_count
    assert (
        target_non_unique_val > 0
    ), (
        f"Unique label ids ({unique_data_count}) cannot exceed "
        + f"or equal the validation size ({val_size})."
    )
    data = data[~mask]

    df_train, df_val = train_test_split(
        data,
        test_size=target_non_unique_val,
        random_state=random_state,
        stratify=data["label_id"],
    )
    return (
        df_train.reset_index(drop=True),
        pd.concat(
            (df_val, unique_data),
            axis=0,
        ).sample(frac=1, random_state=random_state, ignore_index=True),
    )


# * Higher level functions


def get_raw_dataset(
    pred_types: "PredictableTypes",
) -> tuple[pd.DataFrame, dict["FullModelType", pd.DataFrame]]:
    """Get raw dataset for the training of an InfoExtractor."""
    raw_data = parse_or_raise(requests.get(bendp("/labels"), params={"limit": 300_000}))
    data = parse_data(raw_data, pred_types.mts)

    data = filter_ifk(data, pred_types.ifks)
    data = filter_mckd(data, pred_types.mts)

    matches = get_matches()
    data = data.merge(matches, how="inner", on="match_id")
    assert not data.empty, "No labeled match was found in the 'matches' table."

    return (
        data.sort_values(["match_id", "player_id"], ignore_index=True),
        get_label_refs(pred_types),
    )


def split_and_save_dataset(
    data: pd.DataFrame,
    pred_types: "PredictableTypes",
) -> tuple[dict["FullModelType", "Path"], dict["FullModelType", "Path"]]:
    splits_fd = "dbdie_ml/backbone/cache/splits"

    train_paths = {fmt: f"{splits_fd}/{fmt}_train.csv" for fmt in pred_types.fmts}
    val_paths   = {fmt: f"{splits_fd}/{fmt}_val.csv"   for fmt in pred_types.fmts}

    for fmt, mt, ifk in pred_types:
        split = get_split(data, mt)

        split = split[split[f"{mt}_mckd"]]
        assert not split.empty
        split = split.drop(f"{mt}_mckd", axis=1)

        if ifk is not None:
            split = apply_ifk_filter(split, ifk)
            assert not split.empty, "No ifk related data."
        split = split.drop("is_killer", axis=1)

        split = process_label_ids(split, mt)
        split = suffix_filenames(split)

        t_split, v_split = custom_tv_split(split)
        del split

        t_split.to_csv(train_paths[fmt], index=False)
        v_split.to_csv(val_paths[fmt], index=False)
        del t_split, v_split

    return train_paths, val_paths


def save_label_refs(
    label_refs: dict["FullModelType", pd.DataFrame],
) -> dict["FullModelType", "Path"]:
    lref_fd = "dbdie_ml/backbone/cache/label_ref"
    paths = {
        fmt: f"{lref_fd}/{fmt}.csv"
        for fmt in label_refs
    }

    for fmt, lref in label_refs.items():
        lref.to_csv(paths[fmt], index=False)

    return paths
