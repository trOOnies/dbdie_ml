import os
from typing import TYPE_CHECKING
from shutil import move
from dbdie_ml.paths import CROPPED_IMG_FD, CROP_PENDING_IMG_FD
if TYPE_CHECKING:
    from dbdie_ml.classes import Filename, PathToFolder

MAX_PRINT_LEN = 10


class MovableReport:
    """Images' filenames according to movable status"""
    def __init__(self):
        self.mvi, self.umvi = self._calculate_umvis()
        self.obsolete = False
        self.mvi_plain_set = set([f[:-4] for f in self.mvi])
        self.umvi_plain_set = set([f[:-4] for f in self.umvi])

    @staticmethod
    def _calculate_umvis() -> tuple[list["Filename"], list["Filename"]]:
        """Set UMVIs (unmovable images) (and MVIs), that is,
        images that can't be moved because of duplicated filenames
        when comparing src to dst.
        """
        cropped_fd = os.path.join(os.environ["DBDIE_MAIN_FD"], CROPPED_IMG_FD)
        fs = os.listdir(os.environ["DBDIE_MAIN_FD"], CROP_PENDING_IMG_FD)
        list_is_movable = [
            not os.path.exists(os.path.join(cropped_fd, f))
            for f in fs
        ]

        umvi = [
            f for f, movable in zip(fs, list_is_movable)
            if not movable
        ]
        if umvi:
            cond = len(umvi) <= MAX_PRINT_LEN
            msg = f"These images won't be moved bc of duplicated filenames: "
            msg += str(umvi) if cond else str(umvi[:MAX_PRINT_LEN])
            if not cond:
                msg = f"{msg[:-1]}, ...]"
            print(msg)
        else:
            print("All images can be moved")

        mvi = [f for f, movable in zip(fs, list_is_movable) if movable]
        del fs
        assert mvi, "No image can be moved"

        return mvi, umvi


    def load_and_filter(
        self,
        src: "PathToFolder"
    ) -> list["Filename"]:
        """Load and filter images based on UMVI.
        NOTE: These can be the same images that the `MovableReport` used,
        or player crops, hence the difference in exact and cut match.

        Ex.: base.jpg vs base_N.jpg (where N would be a player index)
        """
        assert not self.obsolete
        fs = os.listdir(src)
        sfx_cut = 4 if (src == CROP_PENDING_IMG_FD) else 6
        return [
            f for f in fs
            if f[:-sfx_cut] not in self.umvi_plain_set
        ]

    def move_images(self) -> None:
        """Move movable images to the 'cropped' folder"""
        assert not self.obsolete
        for f in self.mvi:
            move(
                os.path.join(os.environ["DBDIE_MAIN_FD"], CROP_PENDING_IMG_FD, f),
                os.path.join(os.environ["DBDIE_MAIN_FD"], CROPPED_IMG_FD, f)
            )
        self.obsolete = True
        print("Images moved")
