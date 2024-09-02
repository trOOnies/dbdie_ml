"""Cropping settings implementation"""

from __future__ import annotations

import os
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import yaml

from dbdie_ml.classes.base import CropCoords
from dbdie_ml.classes.version import DBDVersionRange
from dbdie_ml.paths import absp, recursive_dirname

if TYPE_CHECKING:
    from dbdie_ml.classes.base import CropType, FullModelType, ImgSize, Path, RelPath

CONFIGS_FD = os.path.join(recursive_dirname(__file__, n=2), "configs")


class CropSettings:
    """Settings for the cropping of a full screenshot or a previously cropped snippet"""

    def __init__(
        self,
        name: "CropType",
        src_fd_rp: "RelPath",
        dst_fd_rp: "RelPath",
        version_range: DBDVersionRange,
        img_size: "ImgSize",
        crops: dict["FullModelType", list[CropCoords]],
        allow: dict[Literal["overlap", "overboard"], bool],
        offset: int = 0,
    ) -> None:
        self.name = name
        self.src_fd_rp = src_fd_rp
        self.dst_fd_rp = dst_fd_rp
        self.version_range = version_range
        self.img_size = img_size
        self.crops = crops
        self.allow = allow
        self.offset = offset

        self.src_fd_rp, self.src = self._setup_folder("src")
        self.dst_fd_rp, self.dst = self._setup_folder("dst")
        self._check_crop_shapes()

    def _setup_folder(self, fd: Literal["src", "dst"]) -> tuple["RelPath", "Path"]:
        """Initial processing of folder's attributes."""
        assert fd in {"src", "dst"}

        rp = getattr(self, f"{fd}_fd_rp")
        rp = rp if rp.startswith("data/") else f"data/{rp}"
        rp = rp[:-1] if rp.endswith("/") else rp

        path = absp(rp)
        assert os.path.isdir(path)

        return rp, path

    def _check_crop_shapes(self):
        """Sets crop sizes and checks if crop coordinates are feasible."""
        if not self.allow["overboard"]:
            img_crop = CropCoords(0, 0, self.img_size[0], self.img_size[1])
            assert all(
                crop.is_fully_inside(img_crop)
                for crops in self.crops.values()
                for crop in crops
            ), f"[ct={self.name}] Crop out of bounds"

        self.crop_shapes = {name: crops[0].shape for name, crops in self.crops.items()}
        assert all(
            (cs[0] > 0) and (cs[1] > 0) for cs in self.crop_shapes.values()
        ), f"[ct={self.name}] Coord sizes must be positive"
        assert all(
            c.shape == self.crop_shapes[name]
            for name, crops in self.crops.items()
            for c in crops
        ), f"[ct={self.name}] All crops must have the same shape"

        if not self.allow["overlap"]:
            assert all(
                not c1.check_overlap(c2)
                for crops in self.crops.values()
                for c1, c2 in combinations(crops, 2)
            ), f"[ct={self.name}] Crops cannot overlap"

    # * Instantiation

    @classmethod
    def from_config(
        cls,
        cfg_name: str,
        depends_on=None,
    ) -> CropSettings:
        """Instantiate CropSettings from a config file."""
        path = os.path.join(CONFIGS_FD, "crop_settings", f"{cfg_name}.yaml")
        with open(path) as f:
            data = yaml.safe_load(f)

        data["version_range"] = DBDVersionRange(*data["version_range"])

        if depends_on is not None:
            assert isinstance(data["img_size"], dict)
            assert depends_on.name == data["img_size"]["cs"]
            data["img_size"] = depends_on.crop_shapes[data["img_size"]["crop"]]
        else:
            assert isinstance(data["img_size"], list)
            assert len(data["img_size"]) == 2
            data["img_size"] = tuple(data["img_size"])

        data["crops"] = {
            fmt: [CropCoords(*c) for c in crops] for fmt, crops in data["crops"].items()
        }

        cs = CropSettings(**data)
        return cs


IMG_SURV_CS = CropSettings.from_config("img_surv_cs")
IMG_KILLER_CS = CropSettings.from_config("img_killer_cs")
PLAYER_SURV_CS = CropSettings.from_config(
    "player_surv_cs",
    depends_on=IMG_SURV_CS,
)
PLAYER_KILLER_CS = CropSettings.from_config(
    "player_killer_cs",
    depends_on=IMG_KILLER_CS,
)

ALL_CS = [IMG_SURV_CS, IMG_KILLER_CS, PLAYER_SURV_CS, PLAYER_KILLER_CS]
