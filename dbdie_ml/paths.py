"""Standard paths for DBDIE.

Holds the instatiation of DBDIEFolderStructure (variable 'dbdie_fs').
"""

from os import environ
from os.path import isdir, join, relpath, dirname
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbdie_ml.classes.base import Path, PathToFolder, RelPath


def absp(rel_path: "RelPath") -> "Path":
    """Convert DBDIE relative path to absolute path"""
    return join(environ["DBDIE_MAIN_FD"], rel_path)


def relp(abs_path: "Path") -> "RelPath":
    """Convert DBDIE absolute path to relative path"""
    return relpath(abs_path, environ["DBDIE_MAIN_FD"])


def recursive_dirname(path: "Path", n: int) -> "PathToFolder":
    """os.path's dirname function but recursive."""
    if n == 1:
        return dirname(path)
    elif n > 1:
        return recursive_dirname(dirname(path), n - 1)
    else:
        raise ValueError(f"n={n} is not a positive integer.")


def validate_rp(rp: "RelPath") -> "RelPath":
    """Validate if the relative path exists and return it if so."""
    assert isdir(absp(rp)), f"Relative path doesn't exist: {rp}"
    return rp


vrp = validate_rp

# * Paths

OLD_VS = "_old_versions"

# * Training

CROPS_MAIN_FD_RP     = vrp("data/crops")
CROPS_VERSIONS_FD_RP = vrp(f"data/crops/{OLD_VS}")

IMG_MAIN_FD_RP         = vrp("data/img")
CROP_PENDING_IMG_FD_RP = vrp("data/img/pending")
CROPPED_IMG_FD_RP      = vrp("data/img/cropped")
IMG_VERSIONS_FD_RP     = vrp(f"data/img/{OLD_VS}")


LABELS_MAIN_FD_RP     = vrp("data/labels")
LABELS_FD_RP          = vrp("data/labels/labels")
LABELS_REF_FD_RP      = vrp("data/labels/label_ref")
LABELS_VERSIONS_FD_RP = vrp(f"data/labels/{OLD_VS}")

# * Inference

INFERENCE_CROPS_MAIN_FD_RP = vrp("inference/crops")

INFERENCE_IMG_MAIN_FD_RP         = vrp("inference/img")
INFERENCE_CROP_PENDING_IMG_FD_RP = vrp("inference/img/pending")
INFERENCE_CROPPED_IMG_FD_RP      = vrp("inference/img/cropped")

INFERENCE_LABELS_MAIN_FD_RP = vrp("inference/labels")
INFERENCE_LABELS_FD_RP      = vrp("inference/labels/labels")
INFERENCE_LABELS_REF_FD_RP  = vrp("inference/labels/label_ref")
