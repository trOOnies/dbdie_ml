"""Extra code for the IEModel models Python file."""

import json
import os
import yaml
from typing import TYPE_CHECKING, Any

from dbdie_classes.version import DBDVersionRange
from torch import load, save
from torch.cuda import mem_get_info

from backbone.classes.metadata import SavedModelMetadata

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
    metadata = SavedModelMetadata.from_model_class(model)
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
