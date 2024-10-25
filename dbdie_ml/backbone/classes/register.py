"""Classes' register related code."""

import os
from typing import TYPE_CHECKING

from dbdie_classes.paths import recursive_dirname

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path

CONFIGS_FD = os.path.join(recursive_dirname(__file__, 2), "configs")
EXTRACTORS_FD = os.path.join(recursive_dirname(__file__, 4), "extractors")


def safe_pathing(path_part: str) -> None:
    assert all(ch not in path_part for ch in [".", "/", "\\"])


def get_extr_mpath(name: str) -> "Path":
    assert name != "models"
    return os.path.join(EXTRACTORS_FD, f"{name}/metadata.yaml")


def get_model_mpath(
    extr_name: str,
    fmt: "FullModelType",
    is_already_trained: bool,
) -> "Path":
    safe_pathing(extr_name)
    safe_pathing(fmt)
    return (
        os.path.join(EXTRACTORS_FD, f"{extr_name}/models/{fmt}/metadata.yaml")
        if is_already_trained
        else os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
    )


def get_cropper_swarm_mpath(name: str) -> "Path":
    safe_pathing(name)
    return os.path.join(CONFIGS_FD, f"cropper_swarms/{name}/metadata.yaml")


def get_crop_settings_mpath(cps_name: str, cs_name: str) -> "Path":
    safe_pathing(cps_name)
    safe_pathing(cs_name)
    return os.path.join(
        CONFIGS_FD,
        f"cropper_swarms/{cps_name}/crop_settings/{cs_name}.yaml",
    )
