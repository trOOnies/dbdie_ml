"""Extra code for the IEModel models Python file."""

import json
import os
import yaml
from typing import TYPE_CHECKING, Any

from torch import load, save
from torch.cuda import mem_get_info

from backbone.cropping.crop_settings import ALL_CS_DICT
from dbdie_classes.options import MODEL_TYPE
from dbdie_classes.version import DBDVersionRange

if TYPE_CHECKING:
    from dbdie_classes.base import LabelRef, Path, PathToFolder


def is_str_like(v: Any) -> bool:
    return not (isinstance(v, (int, bool, float)) or v is None)


# * Loading


def load_metadata_and_model(model_fd: "PathToFolder"):
    with open(os.path.join(model_fd, "metadata.yaml"), "r") as f:
        metadata: dict = yaml.safe_load(f)
    total_classes = metadata["total_classes"]
    del metadata["total_classes"]
    metadata["version_range"] = DBDVersionRange(
        *[str(dbdv) for dbdv in metadata["version_range"]]
    )

    with open(os.path.join(model_fd, "model.pt"), "rb") as f:
        model = load(f)

    return metadata, model, total_classes


def process_metadata(metadata: dict) -> dict:
    """Process IEModel raw metadata dict (straight from the YAML file)."""
    assert metadata["model_type"] in MODEL_TYPE.ALL

    if metadata["is_for_killer"] is not None:
        assert isinstance(metadata["cs"], str)
        assert isinstance(metadata["crop"], str)

        cs = ALL_CS_DICT[metadata["cs"]]
        crop = metadata["crop"]

        metadata["img_size"] = cs.crop_shapes[crop]
    else:
        assert isinstance(metadata["cs"], list)
        assert isinstance(metadata["crop"], list)
        assert len(metadata["cs"]) == 2
        assert len(metadata["crop"]) == 2

        both_cs = [ALL_CS_DICT[cs_str] for cs_str in metadata["cs"]]
        crop_shapes = [
            cs.crop_shapes[crop]
            for cs, crop in zip(both_cs, metadata["crop"])
        ]
        assert crop_shapes[0] == crop_shapes[1]

        metadata["img_size"] = crop_shapes[0]

    del metadata["cs"], metadata["crop"]

    metadata["version_range"] = cs.version_range

    return metadata


def load_label_ref(model_fd: "PathToFolder") -> "LabelRef":
    """Load label_ref used in training."""
    with open(os.path.join(model_fd, "label_ref.json"), "r") as f:
        label_ref = json.load(f)
    label_ref = {int(k): v for k, v in label_ref.items()}
    return label_ref


def print_memory(device) -> None:
    print("MEMORY")
    print(
        "- Free: {:,.2} GiB\n- Total: {:,.2} GiB".format(
            *[v / (2**30) for v in mem_get_info(device)]
        )
    )


# * Saving


def save_metadata(model, dst: "Path") -> None:
    """Save `IEModel` metadata."""
    assert dst.endswith(".yaml")
    metadata = {
        k: getattr(model, k) for k in ["id", "name", "mt", "ifk", "total_classes"]
    } | {"version_range": [model.version_range.id, model.version_range.max_id]}
    metadata["img_size"] = list(model.img_size)
    metadata.update({k: getattr(model, f"_{k}") for k in ["norm_means", "norm_std"]})
    with open(dst, "w") as f:
        yaml.dump(metadata, f)


def save_label_ref(label_ref, dst: "Path") -> None:
    """Save `IEModel` label_ref."""
    with open(dst, "w") as f:
        json.dump(label_ref, f, indent=4)


def save_model(model_is_trained, model, dst: "Path") -> None:
    """Save `IEModel` underlying ML model."""
    assert model_is_trained, "IEModel is not trained"
    assert dst.endswith(".pt")
    save(model, dst)
    # save(model.state_dict(), dst)
