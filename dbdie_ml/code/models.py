import pandas as pd
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dbdie_ml.classes import Path


def is_str_like(v: Any) -> bool:
    return not (isinstance(v, (int, bool, float)) or v is None)


# * Training


def load_label_ref(path: "Path") -> dict[int, str]:
    label_ref = pd.read_csv(
        path,
        usecols=["label_id", "name"],
        dtype={"label_id": int, "name": str},
    )

    unique_vals = label_ref.label_id.unique()
    assert unique_vals.min() == 0
    assert unique_vals.max() + 1 == label_ref.shape[0]
    assert unique_vals.size == label_ref.shape[0]

    return {
        row["label_id"]: row["name"] for _, row in label_ref.iterrows()
    }
