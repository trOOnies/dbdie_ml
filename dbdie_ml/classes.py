import os
from dataclasses import dataclass
from typing import Literal

import yaml

from dbdie_ml.paths import absp

CONFIGS_FD = os.path.join(os.path.dirname(__file__), "configs")

# General
PlayerId = Literal[0, 1, 2, 3, 4]
ModelType = Literal[
    "character", "perks", "item", "addons", "offering", "status", "points"
]
FullModelType = str  # i.e. character__killer
Probability = float  # 0.0 to 1.0

# Paths
Filename = str
PathToFolder = str
RelPath = str
Path = str

# Crops
Width = int
Height = int
ImgSize = tuple[Width, Height]
SnippetWindow = tuple[
    int, int, int, int
]  # Best estimation (1920x1080): (67,217) to (1015,897)
SnippetCoords = tuple[
    int, int, int, int
]  # Best estimation (1920x1080): from 257 to 842 in intervals of 117
EncodedInfo = tuple[int, int, tuple, int, tuple, int, int]

CropType = Literal["surv", "killer", "surv_player", "killer_player"]


@dataclass(frozen=True, eq=True, order=True)
class DBDVersion:
    """DBD game version as named by BHVR (M.m.p)"""

    major: str
    minor: str
    patch: str

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class DBDVersionRange:
    """DBD game version range, first inclusive last exclusive."""

    id: str
    max_id: str | None = None

    def __post_init__(self):
        self.bounded = self.max_id is not None
        self._id = DBDVersion(*[v for v in self.id.split(".")])
        self._max_id = (
            DBDVersion(*[v for v in self.max_id.split(".")]) if self.bounded else None
        )

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


@dataclass
class PlayerInfo:
    """Integer-encoded DBD information of a player snippet"""

    character_id: int
    perks_ids: tuple[int, int, int, int]
    item_id: int
    addons_ids: tuple[int, int]
    offering_id: int
    status_id: int
    points: int


class CropSettings:
    """Settings for the cropping of a full screenshot or a previously cropped snippet"""

    def __init__(
        self,
        name: CropType,
        src_fd_rp: RelPath,
        dst_fd_rp: RelPath,
        version_range: DBDVersionRange,
        img_size: ImgSize,
        crops: dict[FullModelType, list[SnippetWindow] | list[SnippetCoords]],
        offset: int = 0,
    ) -> None:
        self.name = name
        self.src_fd_rp = src_fd_rp
        self.dst_fd_rp = dst_fd_rp
        self.version_range = version_range
        self.img_size = img_size
        self.crops = crops
        self.offset = offset

        self.src_fd_rp, self.src = self._setup_folder("src")
        self.dst_fd_rp, self.dst = self._setup_folder("dst")
        self._check_crop_sizes()

    def _setup_folder(self, fd: Literal["src", "dst"]) -> tuple[RelPath, Path]:
        """Initial processing of folder's attributes."""
        assert fd in {"src", "dst"}

        rp = getattr(self, f"{fd}_fd_rp")
        rp = rp if rp.startswith("data/") else f"data/{rp}"
        rp = rp[:-1] if rp.endswith("/") else rp

        path = absp(rp)
        print(path)
        assert os.path.isdir(path)

        return rp, path

    def _check_crop_sizes(self):
        """Sets crop sizes and checks if crop coordinates are feasible."""
        assert all(
            (coord >= 0) and (coord <= limit)
            for crops in self.crops.values()
            for crop in crops
            for coord, limit in zip(crop, self.img_size)
        ), f"[ct={self.name}] Crop out of bounds"

        self.crop_sizes = {
            name: (crops[0][2] - crops[0][0], crops[0][3] - crops[0][1])
            for name, crops in self.crops.items()
        }
        assert all(
            (cs[0] > 0) and (cs[1] > 0) for cs in self.crop_sizes.values()
        ), f"[ct={self.name}] Coord sizes must be positive"
        assert all(
            (c[2] - c[0], c[3] - c[1]) == self.crop_sizes[name]
            for name, crops in self.crops.items()
            for c in crops
        ), f"[ct={self.name}] All crops must have the same size"

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

        print(data)

        if depends_on is not None:
            assert isinstance(data["img_size"], dict)
            assert depends_on.name == data["img_size"]["cs"]
            data["img_size"] = depends_on.crop_sizes[data["img_size"]["crop"]]
        else:
            assert isinstance(data["img_size"], list)
            assert len(data["img_size"]) == 2
            data["img_size"] = tuple(data["img_size"])

        data["crops"] = {
            fmt: [tuple(c) for c in crops] for fmt, crops in data["crops"].items()
        }

        cs = CropSettings(**data)
        return cs


PlayersSnippetCoords = dict[PlayerId, SnippetCoords]
PlayersInfoDict = dict[PlayerId, PlayerInfo]