import os
from dataclasses import dataclass
from typing import Literal

from dbdie_ml.paths import absp

PlayerId = Literal[0, 1, 2, 3, 4]
ModelType = Literal[
    "character", "perks", "item", "addons", "offering", "status", "points"
]
FullModelType = str  # i.e. character__killer
Probability = float  # 0.0 to 1.0

Filename = str
PathToFolder = str
RelPath = str
Path = str

Width = int
Height = int
SnippetWindow = tuple[
    int, int, int, int
]  # Best estimation (1920x1080): (67,217) to (1015,897)
SnippetCoords = tuple[
    int, int, int, int
]  # Best estimation (1920x1080): from 257 to 842 in intervals of 117
# SnippetInfo = tuple
EncodedInfo = tuple[int, int, tuple, int, tuple, int, int]

Boxes = list[SnippetCoords] | dict[str, list[SnippetCoords]]

CropType = Literal["surv", "killer", "surv_player", "killer_player"]


@dataclass(eq=True, order=True)
class DBDVersion:
    """DBD game version as named by BHVR (M.m.p)"""
    major: int
    minor: int
    patch: int

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass
class DBDVersionRange:
    """DBD game version range, first inclusive last exclusive."""
    id: str
    max_id: str | None = None

    def __post_init__(self):
        self.bounded = self.max_id is not None
        self._id = DBDVersion(*[int(v) for v in self.id.split(".")])
        self._max_id = (
            DBDVersion(*[int(v) for v in self.max_id.split(".")])
            if self.bounded
            else None
        )

    def __contains__(self, v: DBDVersion) -> bool:
        return (self._id <= v) and ((not self.bounded) or (v < self._max_id))

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


@dataclass
class SnippetInfo:
    """Integer-encoded DBD information of a player snippet"""

    character_id: int
    perks_ids: tuple[int, int, int, int]
    item_id: int
    addons_ids: tuple[int, int]
    offering_id: int
    status_id: int
    points: int


@dataclass
class CropSettings:
    """Settings for the cropping of a full screenshot or a previously cropped snippet"""

    name: CropType
    src_fd_rp: RelPath
    dst_fd_rp: RelPath
    version_range: DBDVersionRange
    img_size: tuple[Width, Height]
    crops: dict[FullModelType, list[SnippetWindow] | list[SnippetCoords]]
    offset: int = 0
    src: PathToFolder = ""
    dst: PathToFolder = ""

    def __post_init__(self):
        self._setup_folder("src")
        self._setup_folder("dst")

    def _setup_folder(self, fd: Literal["src", "dst"]) -> None:
        """Initial processing of folder's attributes."""
        assert fd in {"src", "dst"}

        rp = fd if fd.startswith("data/") else f"data/{fd}"
        rp = rp[:-1] if rp.endswith("/") else rp
        path = absp(rp)
        assert os.path.isdir(path)

        setattr(self, f"{fd}_fd_rp", rp)
        setattr(self, fd, path)


AllSnippetCoords = dict[PlayerId, SnippetCoords]
AllSnippetInfo = dict[PlayerId, SnippetInfo]
