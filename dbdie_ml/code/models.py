import yaml
import pandas as pd
from typing import TYPE_CHECKING, Any
from torch import save

if TYPE_CHECKING:
    from dbdie_ml.classes.base import Path


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

    return {row["label_id"]: row["name"] for _, row in label_ref.iterrows()}


# * Loading and saving


def save_metadata(model, dst: "Path") -> None:
    assert dst.endswith(".yaml")
    metadata = {
        k: getattr(model, k)
        for k in ["name", "model_type", "is_for_killer", "version"]
    }
    metadata["image_size"] = list(model.image_size)
    metadata.update({k: getattr(model, f"_{k}") for k in ["norm_means", "norm_std"]})
    with open(dst, "w") as f:
        yaml.dump(metadata, f)


def save_model(obj, dst: "Path") -> None:
    assert obj.model_is_trained, "IEModel is not trained"
    assert dst.endswith(".pt")
    save(obj._model, dst)
    # save(obj._model.state_dict(), dst)
