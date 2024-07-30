import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbdie_ml.classes import RelPath, Path

OLD_VERSIONS_NAME = "_old_versions"

CROPS_MAIN_FD_RP = "data/crops"
CROPS_VERSIONS_FD_RP = f"{CROPS_MAIN_FD_RP}/{OLD_VERSIONS_NAME}"

IMG_MAIN_FD_RP = "data/img"
CROP_PENDING_IMG_FD_RP = f"{IMG_MAIN_FD_RP}/pending"
CROPPED_IMG_FD_RP = f"{IMG_MAIN_FD_RP}/cropped"
IN_CVAT_FD_RP = f"{IMG_MAIN_FD_RP}/in_cvat"

LABELS_MAIN_FD_RP = "data/labels"
LABELS_FD_RP = f"{LABELS_MAIN_FD_RP}/labels"
LABELS_REF_FD_RP = f"{LABELS_MAIN_FD_RP}/label_ref"


def absp(rel_path: "RelPath") -> "Path":
    return os.path.join(os.environ["DBDIE_MAIN_FD"], rel_path)
