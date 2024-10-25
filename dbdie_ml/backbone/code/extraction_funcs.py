"""Extra code for the '/extraction' endpoint."""

from dbdie_classes.options.MODEL_TYPE import (
    ADDONS,
    MULTIPLE_PER_PLAYER,
    PERKS,
    TO_ID_NAMES,
)
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbdie_classes.base import IsForKiller, ModelType


def parse_data(
    data: list[dict],
    mts: list["ModelType"],
) -> pd.DataFrame:
    id_names = {mt: idn for mt, idn in TO_ID_NAMES.items() if mt in mts}
    data = [
        (
            {
            "match_id": d["match_id"],
            "player_id": d["player"]["id"],
            "ifk": d["player"]["id"] == 4,
            }
            | {idn: d["player"][idn] for idn in id_names.values()}
            | {
                f"{mt}_mckd": v
                for mt, v in d["manual_checks"]["predictables"].items()
                if mt in mts
            }
        )
        for d in data
        if d["manual_checks"]["in_progress"]
    ]
    assert data, "No usable label was found in general."

    return pd.DataFrame(data)


def get_relevant_cols(data: pd.DataFrame, mt: "ModelType") -> pd.DataFrame:
    split = data.copy()
    return split[
        [
            "match_id",
            "player_id",
            "ifk",
            TO_ID_NAMES[mt],
            f"{mt}_mckd",
            "filename",
        ]
    ]


def apply_mckd_filter(
    split: pd.DataFrame,
    mt: "ModelType",
    training: bool,
) -> pd.DataFrame:
    """Apply manually-checked filter."""
    mask = split[f"{mt}_mckd"]
    if not training:
        mask = np.logical_not(mask)

    split = split[mask]
    if training:
        assert not split.empty, "Split can't be empty when training."

    return split.drop(f"{mt}_mckd", axis=1)


def apply_ifk_filter(
    split: pd.DataFrame,
    ifk: bool | None,
    training: bool,
) -> pd.DataFrame:
    """Apply ifk (killer boolean) filter."""
    if ifk is not None:
        mask = split["ifk"].values.copy()
        if not ifk:
            mask = np.logical_not(mask)

        split = split[mask]
        if training:
            assert not split.empty, "No ifk related data."
    return split.drop("ifk", axis=1)


def filter_ifk(
    data: pd.DataFrame,
    ifks: list["IsForKiller"],
) -> pd.DataFrame:
    """Filter ifk if all PTs are not None and identical (i.e. a single PT)."""
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
    target_mckd: bool,
) -> pd.DataFrame:
    """Filter manual checks."""
    mckd_cols = [f"{mt}_mckd" for mt in mts]
    for col in mckd_cols:
        data[col] = data[col].fillna(False)

    data = (
        data[data[mckd_cols].any(axis=1)]  # Any row that's at least partially filled
        if target_mckd
        else data[~data[mckd_cols].all(axis=1)]  # Any row that has at least 1 value not checked
    )
    assert not data.empty, "No usable label was found for this specific extractor."

    return data


def flatten_multiple_mt(split: pd.DataFrame, mt: "ModelType") -> pd.DataFrame:
    if mt == PERKS:
        total_ids = 4
    elif mt == ADDONS:
        total_ids = 2
    else:
        raise NotImplementedError

    cols = ["match_id", "player_id", "item_id", "label_id", "filename"]
    if split.empty:
        return pd.DataFrame(columns=cols)

    split = [split.copy() for _ in range(total_ids)]
    for i in range(total_ids):
        split[i].loc[:, "item_id"] = i
        split[i]["label_id"] = split[i]["label_id"].map(lambda vs: vs[i])

    split = pd.concat(split, axis=0)
    split = split[cols]

    return split.sort_values(["match_id", "player_id", "item_id"], ignore_index=True)


def process_label_ids(
    split: pd.DataFrame,
    mt: "ModelType",
    training: bool,
) -> pd.DataFrame:
    split = split.rename({TO_ID_NAMES[mt]: "label_id"}, axis=1)
    if mt in MULTIPLE_PER_PLAYER:
        split = flatten_multiple_mt(split, mt)
    split = split.astype({"label_id": int if training else float})
    return split.reset_index(drop=True)


def suffix_filenames(split: pd.DataFrame, training: bool) -> pd.DataFrame:
    if split.empty:
        assert not training
        return split

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


def split_unique(
    data: pd.DataFrame,
    unique_vals: pd.Series[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split the data using unique values."""
    unique_vals = unique_vals.index.values
    mask = data["label_id"].isin(unique_vals)
    return data[~mask], data[mask]


def process_unique(unique_data: pd.DataFrame, val_size: int) -> int:
    """Process maximum unique assertion."""
    unique_data_count = unique_data.shape[0]

    target_non_unique_val = val_size - unique_data_count
    cond = target_non_unique_val > 0
    assert cond, (
        f"Unique label ids ({unique_data_count}) cannot exceed "
        + f"or equal the validation size ({val_size})."
    )

    return target_non_unique_val


def custom_tv_split(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Custom train-validation split.
    It takes into account that values that aren't repeated
    in the data should always go to the validation split.
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
    data, unique_data = split_unique(data, unique_vals)
    target_non_unique_val = process_unique(unique_data, val_size)

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
