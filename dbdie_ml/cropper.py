from __future__ import annotations
import os
from typing import TYPE_CHECKING
from PIL import Image
from dbdie_ml.crop_settings import (
    IMG_SURV_CS,
    IMG_KILLER_CS,
    PLAYER_SURV_CS,
    PLAYER_KILLER_CS,
)
from dbdie_ml.utils import pls, filter_multitype

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage
    from dbdie_ml.classes import CropType, CropSettings, FullModelType, Path

CS_DICT: dict["CropType", "CropSettings"] = {
    cs.name: cs for cs in [IMG_SURV_CS, IMG_KILLER_CS, PLAYER_SURV_CS, PLAYER_KILLER_CS]
}


class Cropper:
    """Class that crops images in order to have crops model-ready.

    Instantiation:
    >>> cpp = Cropper.from_type("surv")
    or
    >>> cpp = Cropper(CropSettings(...))

    Usage:
    >>> cpp.print_crops()
    >>> ans = cpp.apply_from_path("/path/to/img.png")  # * returns output
    >>> ans["player__surv"][3].show()  # shows the 4th surv player snippet
    """

    def __init__(self, settings: "CropSettings") -> None:
        settings.make_abs_paths()
        assert os.path.isdir(settings.src)
        assert os.path.isdir(settings.dst)
        self.settings = settings
        self.name = self.settings.name

        self.full_model_types = list(self.settings.crops.keys())
        self.full_model_types_set = set(self.full_model_types)

    def __len__(self) -> int:
        return len(self.settings.crops)

    def __repr__(self) -> str:
        """Cropper('data/crops/player__surv' -> 'data/crops', version='7.5.0', image_size=(830, 117), 8 crops)"""
        s = (
            "'{src}' -> '{dst}', ".format(
                src=self.settings.get_rel_path("src"),
                dst=self.settings.get_rel_path("dst"),
            )
            + f"version='{self.settings.version_range}', "
            + f"img_size={self.settings.img_size}, "
        )
        crops_len = len(self)
        s += pls("crop", crops_len)
        return f"Cropper('{self.settings.name}', {s})"

    def print_crops(self) -> None:
        """Print all crop boxes from the `Cropper` settings"""
        for k, vs in self.settings.crops.items():
            print(k)
            for v in vs:
                print(f"- {v}")

    # * Instantiate

    @classmethod
    def from_type(cls, t: "CropType") -> Cropper:
        """Loads a certain type of DBDIE Cropper"""
        cpp = Cropper(settings=CS_DICT[t])
        return cpp

    # * Cropping

    def _filter_fmts(
        self, full_model_types: "FullModelType" | list["FullModelType"] | None
    ) -> list["FullModelType"]:
        possible_values = self.full_model_types
        return filter_multitype(
            full_model_types,
            default=possible_values,
            possible_values=possible_values,
        )

    def apply(
        self,
        img: "PILImage",
        convert_to_rgb: bool = False,
        full_model_types: "FullModelType" | list["FullModelType"] | None = None,
    ) -> dict["FullModelType", list["PILImage"]]:
        """Make and return all the `Cropper` crops for a single in-memory image"""
        fmts = self._filter_fmts(full_model_types)
        if convert_to_rgb:
            img = img.convert("RGB")
        return {
            fmt: [img.crop(box) for box in self.settings.crops[fmt]] for fmt in fmts
        }

    def apply_from_path(
        self,
        path: "Path",
        full_model_types: "FullModelType" | list["FullModelType"] | None = None,
    ) -> dict["FullModelType", list["PILImage"]]:
        """Make and return all the `Cropper` crops for a single in-memory image"""
        fmts = self._filter_fmts(full_model_types)
        img = Image.open(path)
        img = img.convert("RGB")
        return {
            fmt: [img.crop(box) for box in self.settings.crops[fmt]] for fmt in fmts
        }
