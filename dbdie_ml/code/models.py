"""Extra code for the IEModel models Python file"""

import json
import os
from typing import TYPE_CHECKING, Any

import yaml
from torch import save

from dbdie_ml.cropping.crop_settings import ALL_CS
from dbdie_ml.options import MODEL_TYPES

if TYPE_CHECKING:
    from dbdie_ml.classes.base import LabelRef, Path, PathToFolder


def is_str_like(v: Any) -> bool:
    return not (isinstance(v, (int, bool, float)) or v is None)


# * Loading


def process_metadata(metadata: dict) -> dict:
    """Process IEModel raw metadata dict (straight from the YAML file)."""
    assert metadata["model_type"] in MODEL_TYPES.ALL

    cs = [cs_i for cs_i in ALL_CS if cs_i.name == metadata["cs"]][0]
    metadata["img_size"] = cs.crop_shapes[metadata["crop"]]
    del metadata["cs"], metadata["crop"]

    metadata["version_range"] = cs.version_range

    return metadata


def load_label_ref(model_fd: "PathToFolder") -> "LabelRef":
    """Load label_ref used in training."""
    with open(os.path.join(model_fd, "label_ref.json"), "r") as f:
        label_ref = json.load(f)
    label_ref = {int(k): v for k, v in label_ref.items()}
    return label_ref


# * Saving


def save_metadata(model, dst: "Path") -> None:
    assert dst.endswith(".yaml")
    metadata = {
        k: getattr(model, k) for k in ["name", "model_type", "is_for_killer", "version"]
    }
    metadata["img_size"] = list(model.img_size)
    metadata.update({k: getattr(model, f"_{k}") for k in ["norm_means", "norm_std"]})
    with open(dst, "w") as f:
        yaml.dump(metadata, f)


def save_model(obj, dst: "Path") -> None:
    assert obj.model_is_trained, "IEModel is not trained"
    assert dst.endswith(".pt")
    save(obj._model, dst)
    # save(obj._model.state_dict(), dst)
