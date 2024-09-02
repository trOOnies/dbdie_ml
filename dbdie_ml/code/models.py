import json
import os
from typing import TYPE_CHECKING, Any

import yaml
from torch import load, save

from dbdie_ml.cropping.crop_settings import ALL_CS
from dbdie_ml.options import MODEL_TYPES

if TYPE_CHECKING:
    from dbdie_ml.classes.base import Path, PathToFolder


def is_str_like(v: Any) -> bool:
    return not (isinstance(v, (int, bool, float)) or v is None)


# * Loading


def load_with_trained_model(obj_class, model_fd: "PathToFolder", metadata: dict):
    with open(os.path.join(model_fd, "model.pt"), "rb") as f:
        model = load(f)

    iem = obj_class(model=model, **metadata)
    iem.init_model()
    iem.model_is_trained = True

    return iem


def load_label_ref(model_fd: "PathToFolder"):
    with open(os.path.join(model_fd, "label_ref.json"), "r") as f:
        label_ref = json.load(f)
    label_ref = {int(k): v for k, v in label_ref.items()}
    return label_ref


def process_metadata(metadata: dict) -> dict:
    assert metadata["model_type"] in MODEL_TYPES.ALL
    cs = [cs_i for cs_i in ALL_CS if cs_i.name == metadata["cs"]][0]
    metadata["img_size"] = cs.crop_shapes[metadata["crop"]]
    del metadata["cs"], metadata["crop"]

    metadata["version_range"] = cs.version_range
    metadata["_norm_means"] = metadata["norm_means"]
    metadata["_norm_std"] = metadata["norm_std"]
    del metadata["norm_means"], metadata["norm_std"]

    metadata["training_params"] = metadata["training"]
    del metadata["training"]

    return metadata


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
