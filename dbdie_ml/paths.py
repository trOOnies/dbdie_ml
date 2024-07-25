import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbdie_ml.classes import RelPath, Path

CROPS_FD = "data/crops"
CROP_PENDING_IMG_FD = "data/img/pending"
CROPPED_IMG_FD = "data/img/cropped"


def absp(rel_path: "RelPath") -> "Path":
    return os.path.join(os.environ["DBDIE_MAIN_FD"], rel_path)
