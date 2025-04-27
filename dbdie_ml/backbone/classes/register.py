"""Classes' register related code."""

import os
from typing import TYPE_CHECKING, Union

from dbdie_classes.paths import recursive_dirname

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path

BASE_SUPFD = recursive_dirname(__file__, 4)
assert os.path.isdir(BASE_SUPFD), "Base folder does not exist."

CONFIGS_FD = os.path.join(BASE_SUPFD, "configs")
EXTRACTORS_FD = os.path.join(BASE_SUPFD, "extractors")
MODELS_FD = os.path.join(BASE_SUPFD, "models")


def safe_pathing(path_part: str) -> None:
    assert all(ch not in path_part for ch in [".", "/", "\\"])


def get_extr_mpath(id: int) -> "Path":
    assert isinstance(id, int), "id must be an integer."
    return os.path.join(EXTRACTORS_FD, f"{id}/metadata.yaml")


def get_model_mpath(
    fmt: Union["FullModelType", None] = None,
    id: int | None = None,
) -> "Path":
    assert (fmt is None) != (id is None), "Either fmt or id must be provided."

    if fmt is not None:
        safe_pathing(fmt)
        return os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
    else:
        assert isinstance(id, int), "id must be an integer."
        return os.path.join(MODELS_FD, f"{id}/metadata.yaml")


def get_cropper_swarm_mpath(name: str) -> "Path":
    safe_pathing(name)
    return os.path.join(CONFIGS_FD, f"cropper_swarms/{name}/metadata.yaml")


def get_crop_settings_mpath(cps_name: str, cs_name: str) -> "Path":
    safe_pathing(cps_name)
    safe_pathing(cs_name)
    rpath = f"cropper_swarms/{cps_name}/crop_settings/{cs_name}.yaml"
    return os.path.join(CONFIGS_FD, rpath)
