from __future__ import annotations
import os
from typing import TYPE_CHECKING
from PIL import Image
from dbdie_ml.crop_settings import (
    IMG_SURV_CS,
    IMG_KILLER_CS,
    PLAYER_SURV_CS,
    PLAYER_KILLER_CS
)
from dbdie_ml.utils import pls
if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from dbdie_ml.classes import (
        CropType, CropSettings, FullModelType, Path
    )

CS_DICT: dict["CropType", "CropSettings"] = {
    cs.name: cs
    for cs in [
        IMG_SURV_CS,
        IMG_KILLER_CS,
        PLAYER_SURV_CS,
        PLAYER_KILLER_CS
    ]
}


class Cropper:
    """Class that crops images in order to have crops model-ready."""
    def __init__(self, settings: "CropSettings") -> None:
        settings.make_abs_paths()
        assert os.path.isdir(settings.src)
        assert os.path.isdir(settings.dst)
        self.settings = settings

    def __len__(self) -> int:
        return len(self.settings.crops)

    def __repr__(self) -> str:
        """Cropper('data\crops\player__surv' -> 'data\crops', image_size=(830, 117), 8 crops)"""
        s = (
            "'{src}' -> '{dst}', ".format(
                src=self.settings.get_rel_path("src"),
                dst=self.settings.get_rel_path("dst")
            ) +
            f"img_size={self.settings.img_size}, "
        )
        crops_len = len(self)
        s += pls("crop", crops_len)
        return f"Cropper('{self.settings.name}', {s})"

    def print_crops(self) -> None:
        for k, vs in self.settings.crops.items():
            print(k)
            for v in vs:
                print(f"- {v}")

    @property
    def full_model_types(self) -> list["FullModelType"]:
        return list(self.settings.crops.keys())

    # * Instantiate

    @classmethod
    def from_type(cls, t: "CropType") -> Cropper:
        """Loads a certain type of DBDIE Cropper"""
        cpp = Cropper(settings=CS_DICT[t])
        return cpp

    # * Cropping

    # TODO: Implement again 1 FullModelType apply function

    def apply(
        self,
        img: "PILImage",
        convert_to_rgb: bool = False
    ) -> dict["FullModelType", list["PILImage"]]:
        """Make and return all the `Cropper` crops for a single in-memory image"""
        if convert_to_rgb:
            img = img.convert("RGB")
        return {
            full_mt: [img.crop(box) for box in boxes]
            for full_mt, boxes in self.settings.crops.items()
        }

    def apply_from_path(self, path: "Path") -> dict["FullModelType", list["PILImage"]]:
        """Make and return all the `Cropper` crops for a single in-memory image"""
        img = Image.open(path)
        img = img.convert("RGB")
        return {
            full_mt: [img.crop(box) for box in boxes]
            for full_mt, boxes in self.settings.crops.items()
        }
