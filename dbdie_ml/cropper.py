from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional, Literal
from shutil import move
from copy import deepcopy
from PIL import Image
from tqdm import tqdm
from crop_settings import (
    IMG_SURV_CS,
    IMG_KILLER_CS,
    PLAYER_SURV_CS,
    PLAYER_KILLER_CS,
    CROPPED_IMG_FD
)
if TYPE_CHECKING:
    from dbdie_ml.classes import Filename, PathToFolder, Boxes, CropSettings

CS_DICT = {
    "surv": IMG_SURV_CS,
    "killer": IMG_KILLER_CS,
    "surv_player": PLAYER_SURV_CS,
    "killer_player": PLAYER_KILLER_CS,
}


class Cropper:
    """Class that crops images in order to have crops model-ready."""
    def __init__(self, settings: "CropSettings") -> None:
        assert os.path.isdir(self.settings["src"])
        assert os.path.isdir(self.settings["dst"])
        self.settings = settings

    # * Instantiate

    @classmethod
    def from_type(
        cls,
        t: Literal["surv", "killer", "surv_player", "killer_player"]
    ) -> Cropper:
        """Loads a certain type of DBDIE Cropper"""
        cpp = Cropper(settings=CS_DICT[t])
        return cpp

    # * Processing

    def _set_umi(self) -> None:
        """Set UMIs (unmovable images) (and MIs), that is,
        images that can't be moved because of duplicated filenames
        when comparing src to dst.
        """
        fs = os.listdir(CS_DICT["surv"].src)  # hardcoded for now
        list_is_movable = [
            not os.path.exists(os.path.join(CROPPED_IMG_FD, f))
            for f in fs
        ]

        self.unmovable_imgs = [
            f for f, movable in zip(fs, list_is_movable)
            if not movable
        ]
        if self.unmovable_imgs:
            print(
                f"These images won't be moved bc of duplicated filenames: {self.unmovable_imgs}"
            )
        else:
            print("All images can be moved")
            self.unmovable_imgs = None

        self.movable_imgs = [f for f, movable in zip(fs, list_is_movable) if movable]
        del fs
        self.movable_imgs = set(self.movable_imgs)

    # * Cropping

    @staticmethod
    def crop_image(
        src_fd: "PathToFolder",
        dst_fd: "PathToFolder",
        filename: "Filename",
        boxes: "Boxes",
        offset: Optional[int]
    ) -> None:
        src = os.path.join(src_fd, filename)
        plain = filename[:-4]

        img = Image.open(src)
        img = img.convert("RGB")

        if isinstance(boxes, list):
            using_boxes = boxes
        else:
            using_boxes = boxes["killer" if plain.endswith("_4") else "surv"]

        o = offset if isinstance(offset, int) else 0
        for i, box in enumerate(using_boxes):
            cropped = img.crop(box)
            cropped.save(os.path.join(dst_fd, f"{plain}_{i+o}.jpg"))

    def _crop_process(
        self,
        full_model_type: str,
        src_filenames: list["Filename"],
        offset: Optional[int]
    ) -> None:
        # TODO: Make a separate graph that initializes the crop folder and the rest if needed
        with tqdm(total=len(src_filenames), desc=full_model_type) as pbar:
            for f in src_filenames:
                self.crop_image(
                    self.settings.src,
                    os.path.join(self.settings.dst, full_model_type),
                    filename=f,
                    boxes=self.settings.crops[full_model_type],
                    offset=offset
                )
                pbar.update(1)

    def _process_umi(
        self,
        src_filenames: list[str],
        use_starting_match: bool
    ) -> list[str]:
        """Process unmovable images (UMI).

        Args:
            src_filenames (list[str])
            use_starting_match (bool): UMIs are compared to images
                as starting match (instead of exact match)
        """
        if use_starting_match:
            # Starting match
            umi_sw = [f[:-4] for f in self.umi]
            sw_len = umi_sw[0]
            assert all(len(f) == sw_len for f in umi_sw)
            umi_sw = set(umi_sw)

            return [
                f for f in src_filenames
                if f[:sw_len] not in umi_sw
            ]
        else:
            # Exact match
            return [
                f for f in src_filenames
                if f not in self.umi
            ]

    def run_crop(
        self,
        crop_only: Optional[list[str]],
        offset: Optional[int] = None,
        use_starting_match: bool = False
    ) -> None:
        """Run crops based on the `settings`.

        Args:
            crop_only (list[str] | None): Crop only certain crop types
            offset (int | None)
            use_starting_match (bool)
        """
        if crop_only is None:
            ks = list(self.settings.crops.keys())  # no filter
        else:
            assert crop_only
            assert len(crop_only) == len(set(crop_only))
            assert all(ct in self.settings.crops for ct in crop_only)
            ks = deepcopy(crop_only)

        src_filenames = os.listdir(self.settings.src)
        if self.unmovable_imgs is not None:
            src_filenames = self._process_umi(
                src_filenames,
                use_starting_match
            )
        if not src_filenames:
            return

        for k in ks:
            self._crop_process(
                k,
                src_filenames=src_filenames,
                offset=offset
            )

    # * Moving images

    def move_images(self) -> None:
        for f in self.movable_imgs:
            move(
                os.path.join(CS_DICT["surv"].src, f),  # hardcoded for now
                os.path.join(CROPPED_IMG_FD, f)
            )
        print("Images moved")
