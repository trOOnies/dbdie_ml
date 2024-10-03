from __future__ import annotations

from typing import TYPE_CHECKING

from PIL import Image

from dbdie_classes.utils import filter_multitype, pls

from backbone.cropping.crop_settings import CropSettings

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

    from dbdie_classes.base import CropType, FullModelType, Path


class Cropper:
    """Crops images in order to have crops model-ready.

    Instantiation:
    >>> cpp = Cropper.from_type("surv")
    or
    >>> cpp = Cropper(CropSettings(...))

    Usage:
    >>> cpp.print_crops()
    >>> ans = cpp.apply_from_path("/path/to/img.png")  # * returns output
    >>> ans["player__surv"][3].show()  # shows the 4th surv player snippet
    """

    def __init__(self, settings: CropSettings) -> None:
        self.settings = settings
        self.name = self.settings.name

        self.full_model_types = list(self.settings.crops.keys())
        self.full_model_types_set = set(self.full_model_types)

    def __len__(self) -> int:
        return len(self.settings.crops)

    def __repr__(self) -> str:
        """Cropper('surv_player', 'crops/player__surv' -> 'crops/...',
        version='>=7.5.0', img_size=(830, 117), 8 crops)
        """
        s = ", ".join(
            [
                "'{src}' -> '{dst}'".format(
                    src=self.settings.src_fd_rp[5:],
                    dst=self.settings.dst_fd_rp[5:] + "/...",
                ),
                f"version='{self.settings.version_range}'",
                f"img_size={self.settings.img_size}",
                pls("crop", len(self)),
            ]
        )
        return f"Cropper('{self.settings.name}', {s})"

    def print_crops(self) -> None:
        """Print all crop boxes from the Cropper settings."""
        for k, vs in self.settings.crops.items():
            print(k)
            for v in vs:
                print(f"- {v}")

    # * Instantiate

    @classmethod
    def from_register(cls, cps_name: str, t: "CropType") -> Cropper:
        """Load a certain type of Cropper."""
        cs_dict = CropSettings.make_cs_dict(cps_name)
        cpp = Cropper(settings=cs_dict[t])
        return cpp

    # * Cropping

    def _filter_fmts(
        self,
        full_model_types: "FullModelType" | list["FullModelType"] | None,
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
        """Make and return all the Cropper crops for a single in-memory image"""
        fmts = self._filter_fmts(full_model_types)
        if convert_to_rgb:
            img = img.convert("RGB")
        return {
            fmt: [img.crop(box.raw()) for box in self.settings.crops[fmt]]
            for fmt in fmts
        }

    def apply_from_path(
        self,
        path: "Path",
        full_model_types: "FullModelType" | list["FullModelType"] | None = None,
    ) -> dict["FullModelType", list["PILImage"]]:
        """Make and return all the Cropper crops for a single in-memory image"""
        fmts = self._filter_fmts(full_model_types)
        img = Image.open(path)
        img = img.convert("RGB")
        return {
            fmt: [img.crop(box.raw()) for box in self.settings.crops[fmt]]
            for fmt in fmts
        }
