"""DBD version related classes"""

import os
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Literal

import yaml

from dbdie_ml.classes.base import CropCoords
from dbdie_ml.paths import absp

if TYPE_CHECKING:
    from dbdie_ml.classes.base import (
        CropType,
        FullModelType,
        ImgSize,
        Path,
        RelPath,
    )

CONFIGS_FD = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs")


@dataclass(frozen=True, eq=True, order=True)
class DBDVersion:
    """DBD game version as named by BHVR (M.m.p)"""

    major: str
    minor: str
    patch: str
    is_not_ptb: bool = True

    def __str__(self) -> str:
        return (
            f"{self.major}.{self.minor}.{self.patch}{'' if self.is_not_ptb else '-ptb'}"
        )

    @classmethod
    def from_str(cls, s: str):
        ss = s.split(".")
        is_ptb = ss[2].endswith("-ptb")
        return DBDVersion(
            ss[0],
            ss[1],
            ss[2][:-4] if is_ptb else ss[2],
            not is_ptb,
        )

    @classmethod
    def from_schema(cls, dbdv):
        return cls.from_str(dbdv.name)


@dataclass
class DBDVersionRange:
    """DBD game version range, first inclusive last exclusive."""

    id: str
    max_id: str | None = None

    def __post_init__(self):
        self.bounded = self.max_id is not None
        self._id = DBDVersion.from_str(self.id)
        self._max_id = DBDVersion.from_str(self.max_id) if self.bounded else None

    def __str__(self) -> str:
        return f">={self._id},<{self._max_id}" if self.bounded else f">={self._id}"

    def __eq__(self, other) -> bool:
        if isinstance(other, DBDVersionRange):
            if self._id != other._id:
                return False
            if not (self.bounded or other.bounded):
                return True
            return (
                (self._max_id == other._max_id)
                if (self.bounded == other.bounded)
                else False
            )
        return False

    def __contains__(self, v: DBDVersion) -> bool:
        return (self._id <= v) and ((not self.bounded) or (v < self._max_id))

    def __and__(self, other):
        """DBDVersions intersection."""
        _id = max(self._id, other._id)
        id = str(_id)

        if not self.bounded:
            max_id = other.max_id
        elif not other.bounded:
            max_id = self.max_id
        else:
            _max_id = min(self._max_id, other._max_id)
            max_id = str(_max_id)

        return DBDVersionRange(id, max_id)


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
    ):
        path = os.path.join(CONFIGS_FD, f"{cfg_name}.yaml")
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
