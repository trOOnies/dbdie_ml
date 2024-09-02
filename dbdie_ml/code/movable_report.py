"""Extra code for the movable_report Python file"""

import os
from typing import TYPE_CHECKING

from dbdie_ml.paths import CROP_PENDING_IMG_FD_RP, CROPPED_IMG_FD_RP, absp

if TYPE_CHECKING:
    from dbdie_ml.classes.base import Filename

MAX_PRINT_LEN = 10


def print_umvi_verdict(umvi: list[str]) -> None:
    """Print whether there are unmovable images or not."""
    if umvi:
        cond = len(umvi) <= MAX_PRINT_LEN
        msg = "These images won't be moved bc of duplicated filenames: "
        msg += str(umvi) if cond else str(umvi[:MAX_PRINT_LEN])
        if not cond:
            msg = f"{msg[:-1]}, ...]"
        print(msg)
    else:
        print("All images can be moved")


def calculate_umvis() -> tuple[list["Filename"], list["Filename"]]:
    """Set UMVIs (unmovable images) (and MVIs), that is,
    images that can't be moved because of duplicated filenames
    when comparing src to dst.
    """
    cropped_fd = absp(CROPPED_IMG_FD_RP)
    fs = os.listdir(absp(CROP_PENDING_IMG_FD_RP))

    list_is_movable = [not os.path.exists(os.path.join(cropped_fd, f)) for f in fs]

    umvi = [f for f, movable in zip(fs, list_is_movable) if not movable]
    print_umvi_verdict(umvi)

    mvi = [f for f, movable in zip(fs, list_is_movable) if movable]
    del fs
    assert mvi, "No image can be moved"

    return mvi, umvi
