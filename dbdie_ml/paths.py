import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbdie_ml.classes.base import Path, RelPath

OLD_VERSIONS_NAME = "_old_versions"

# * Training

CROPS_MAIN_FD_RP: "RelPath" = "data/crops"
CROPS_VERSIONS_FD_RP: "RelPath" = f"{CROPS_MAIN_FD_RP}/{OLD_VERSIONS_NAME}"

IMG_MAIN_FD_RP: "RelPath" = "data/img"
CROP_PENDING_IMG_FD_RP: "RelPath" = f"{IMG_MAIN_FD_RP}/pending"
CROPPED_IMG_FD_RP: "RelPath" = f"{IMG_MAIN_FD_RP}/cropped"
IN_CVAT_FD_RP: "RelPath" = f"{IMG_MAIN_FD_RP}/in_cvat"
IMG_VERSIONS_FD_RP: "RelPath" = f"{IMG_MAIN_FD_RP}/{OLD_VERSIONS_NAME}"

LABELS_MAIN_FD_RP: "RelPath" = "data/labels"
LABELS_FD_RP: "RelPath" = f"{LABELS_MAIN_FD_RP}/labels"
LABELS_REF_FD_RP: "RelPath" = f"{LABELS_MAIN_FD_RP}/label_ref"
LABELS_VERSIONS_FD_RP: "RelPath" = f"{LABELS_MAIN_FD_RP}/{OLD_VERSIONS_NAME}"

# * Inference

INFERENCE_CROPS_MAIN_FD_RP: "RelPath" = "inference/crops"

INFERENCE_IMG_MAIN_FD_RP: "RelPath" = "inference/img"
INFERENCE_CROP_PENDING_IMG_FD_RP: "RelPath" = f"{INFERENCE_IMG_MAIN_FD_RP}/pending"
INFERENCE_CROPPED_IMG_FD_RP: "RelPath" = f"{INFERENCE_IMG_MAIN_FD_RP}/cropped"

INFERENCE_LABELS_MAIN_FD_RP: "RelPath" = "inference/labels"
INFERENCE_LABELS_FD_RP: "RelPath" = f"{INFERENCE_LABELS_MAIN_FD_RP}/labels"
INFERENCE_LABELS_REF_FD_RP: "RelPath" = f"{INFERENCE_LABELS_MAIN_FD_RP}/label_ref"


def absp(rel_path: "RelPath") -> "Path":
    return os.path.join(os.environ["DBDIE_MAIN_FD"], rel_path)


def relp(abs_path: "Path") -> "RelPath":
    return os.path.relpath(abs_path, os.environ["DBDIE_MAIN_FD"])