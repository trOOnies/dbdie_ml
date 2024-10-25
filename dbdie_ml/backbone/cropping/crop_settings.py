"""CropSettings implementation."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Literal

import yaml

from dbdie_classes.extract import CropCoords
from dbdie_classes.options import CROP_TYPES
from dbdie_classes.paths import absp

from backbone.classes.register import get_crop_settings_mpath
from backbone.code.crop_settings import (
    check_overboard,
    check_overlap,
    check_positivity,
    check_shapes,
    get_dbdvr,
    process_img_size,
)

if TYPE_CHECKING:
    from dbdie_classes.base import (
        CropType, FullModelType, ImgSize, Path, RelPath
    )
    from dbdie_classes.schemas.helpers import DBDVersionRange


class CropSettings:
    """Settings for the cropping of a full screenshot or a previously cropped snippet"""

    def __init__(
        self,
        name: "CropType",
        src_fd_rp: "RelPath",
        dst_fd_rp: "RelPath",
        dbdvr: "DBDVersionRange",
        img_size: "ImgSize",
        crops: dict["FullModelType", list[CropCoords]],
        allow: dict[Literal["overlap", "overboard"], bool],
        offset: int = 0,
    ) -> None:
        self.name = name
        self.src_fd_rp = src_fd_rp
        self.dst_fd_rp = dst_fd_rp
        self.dbdvr = dbdvr
        self.img_size = img_size
        self.crops = crops
        self.allow = allow
        self.offset = offset

        self.src_fd_rp, self.src = self._setup_folder("src")
        self.dst_fd_rp, self.dst = self._setup_folder("dst")
        self._check_crop_shapes()

    def _setup_folder(
        self,
        fd: Literal["src", "dst"],
    ) -> tuple["RelPath", "Path"]:
        """Initial processing of folder's attributes."""
        assert fd in {"src", "dst"}

        rp = getattr(self, f"{fd}_fd_rp")
        rp = rp if rp.startswith("data/") else f"data/{rp}"
        rp = rp[:-1] if rp.endswith("/") else rp

        path = absp(rp)
        assert os.path.isdir(path), f"Folder doesn't exist: {path}"

        return rp, path

    def _check_crop_shapes(self):
        """Sets crop sizes and checks if crop coordinates are feasible."""
        check_overboard(self.name, self.allow, self.img_size, self.crops)

        self.crop_shapes = {
            fmt: crops[0].shape
            for fmt, crops in self.crops.items()
        }
        check_positivity(self.name, self.crop_shapes)
        check_shapes(self.name, self.crops, self.crop_shapes)

        check_overlap(self.name, self.allow, self.crops)

    # * Instantiation

    @classmethod
    def from_register(
        cls,
        cps_name: str,
        cs_name: str,
        depends_on: CropSettings | None,
    ) -> CropSettings:
        """Instantiate CropSettings from a config file."""
        path = get_crop_settings_mpath(cps_name, cs_name)
        with open(path) as f:
            data = yaml.safe_load(f)

        data["dbdvr"] = get_dbdvr(data["dbdvr"])
        data = process_img_size(data, depends_on)
        data["crops"] = {
            fmt: [CropCoords(*c) for c in crops]
            for fmt, crops in data["crops"].items()
        }

        cs = CropSettings(**data)
        return cs

    # * Many CropSettings

    @classmethod
    def make_cs_dict(cls, cps_name: str) -> dict[str, CropSettings]:
        IMG_SURV_CS = cls.from_register(cps_name, "img_surv_cs", depends_on=None)
        IMG_KILLER_CS = cls.from_register(cps_name, "img_killer_cs", depends_on=None)
        PLAYER_SURV_CS = cls.from_register(
            cps_name,
            "player_surv_cs",
            depends_on=IMG_SURV_CS,
        )
        PLAYER_KILLER_CS = cls.from_register(
            cps_name,
            "player_killer_cs",
            depends_on=IMG_KILLER_CS,
        )

        return {
            CROP_TYPES.SURV: IMG_SURV_CS,
            CROP_TYPES.KILLER: IMG_KILLER_CS,
            CROP_TYPES.SURV_PLAYER: PLAYER_SURV_CS,
            CROP_TYPES.KILLER_PLAYER: PLAYER_KILLER_CS,
        }
