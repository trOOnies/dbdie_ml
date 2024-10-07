"""MovableReport class code."""

from dbdie_classes.paths import (
    CROP_PENDING_IMG_FD_RP, CROPPED_IMG_FD_RP, absp, relp
)
import os
from shutil import move
from typing import TYPE_CHECKING

from backbone.code.movable_report import calculate_umvis
from backbone.options.COLORS import OKBLUE, make_cprint_with_header

if TYPE_CHECKING:
    from dbdie_classes.base import Filename, PathToFolder

mr_print = make_cprint_with_header(OKBLUE, "[MovableReport]")


class MovableReport:
    """Images' filenames according to movable status.
    Used in the CropperSwarm code.

    . # ! MovableReport is not intended to be used outside of this library's code

    Take into account the obsolete attribute.
    It's assumed that if MovableReport.move_images() is used,
    the mvi and umvi lists become stale. You should instantiate
    a new MovableReport instead.
    """

    def __init__(self, filenames: list["Filename"]):
        assert filenames, "'filenames' cannot be empty."
        self.mvi, self.umvi = calculate_umvis(filenames)
        self.obsolete = False
        self.mvi_plain_set = set([f[:-4] for f in self.mvi])
        self.umvi_plain_set = set([f[:-4] for f in self.umvi])

    def load_and_filter(self, src: "PathToFolder") -> list["Filename"]:
        """Load and filter images based on UMVI.
        NOTE: These can be the same images that the MovableReport used,
        or player crops, hence the difference in exact and cut match.

        Ex.: base.jpg vs base_N.jpg (where N would be a player index)
        """
        assert not self.obsolete, (
            "You must create a new MovableReport once you've used it for moving images."
        )
        fs = os.listdir(src)
        sfx_cut = 4 if (relp(src) == CROP_PENDING_IMG_FD_RP) else 6  # TODO: Test
        return [f for f in fs if f[:-sfx_cut] not in self.umvi_plain_set]

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        mr_print("Moving images...")
        assert not self.obsolete, (
            "You must create a new MovableReport once you've used it for moving images."
        )

        pending_fd = absp(CROP_PENDING_IMG_FD_RP)
        cropped_fd = absp(CROPPED_IMG_FD_RP)

        self.obsolete = True
        for f in self.mvi:
            move(
                os.path.join(pending_fd, f),
                os.path.join(cropped_fd, f),
            )
        mr_print("Images moved.")
