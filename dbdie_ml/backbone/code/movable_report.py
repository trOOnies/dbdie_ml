"""Extra code for the MovableReport class file."""

from dbdie_classes.paths import CROP_PENDING_IMG_FD_RP, CROPPED_IMG_FD_RP, absp
import os
from typing import TYPE_CHECKING

from backbone.options.COLORS import get_class_cprint

if TYPE_CHECKING:
    from dbdie_classes.base import Filename

MAX_PRINT_LEN = 10
mr_print = get_class_cprint("MovableReport")


def print_umvi_verdict(umvi: list[str], checking: str) -> None:
    """Print whether there are unmovable images or not."""
    if umvi:
        cond = len(umvi) <= MAX_PRINT_LEN
        msg = f"({checking}) These images won't be moved: "
        msg += str(umvi) if cond else str(umvi[:MAX_PRINT_LEN])
        if not cond:
            msg = f"{msg[:-1]}, ...]"
        mr_print(msg)
    else:
        mr_print(f"({checking}) All images can be moved.")


def calculate_umvis(fs: list["Filename"]) -> tuple[list["Filename"], list["Filename"]]:
    """Set UMVIs (unmovable images) (and MVIs), that is,
    images that can't be moved because of duplicated filenames
    when comparing src to dst.
    """
    cropped_fd = absp(CROPPED_IMG_FD_RP)
    pending_fs = os.listdir(absp(CROP_PENDING_IMG_FD_RP))
    total = len(pending_fs)
    assert total > 0, "There are no images in the pending folder."
    mr_print(f"Total images: {total}")

    # Filtering with our list

    mask = [f in fs for f in pending_fs]
    fs_ = [f for f, cond in zip(pending_fs, mask) if cond]
    assert fs_, "No image in 'fs' was found in the pending folder."

    umvi = [f for f, cond in zip(pending_fs, mask) if not cond]
    print_umvi_verdict(umvi, checking="DBD versions")

    # Filtering by actual movable condition

    list_is_movable = [not os.path.exists(os.path.join(cropped_fd, f)) for f in fs_]

    mvi = [f for f, movable in zip(fs_, list_is_movable) if movable]
    assert mvi, "No image can be moved"

    new_umvi = [f for f, movable in zip(fs_, list_is_movable) if not movable]
    del fs_
    print_umvi_verdict(new_umvi, "Image duplication")
    umvi += new_umvi
    mr_print(f"MVI: {len(mvi)} | UMVI: {len(umvi)}")

    assert len(mvi) + len(umvi) == total
    return mvi, umvi
