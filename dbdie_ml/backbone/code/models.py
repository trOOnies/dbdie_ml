"""Extra code for the IEModel models Python file."""

import json
import os
from typing import TYPE_CHECKING, Any

from torch import load, save
from torch.cuda import mem_get_info

from backbone.classes.metadata import SavedModelMetadata

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, LabelRef, Path, PathToFolder


def is_str_like(v: Any) -> bool:
    return not (isinstance(v, (int, bool, float)) or v is None)


# * Loading


def load_metadata_and_model(extr_name: str, fmt: "FullModelType", model_fd: str):
    metadata = SavedModelMetadata.load(
        fmt=fmt,
        extr_name=extr_name,
        model_id=None,
        total_classes=None,
        cps_name=None,
    )

    with open(os.path.join(model_fd, "model.pt"), "rb") as f:
        model = load(f)

    return metadata, model, metadata.total_classes


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
    metadata = model.to_metadata()
    metadata.save(dst)


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
