import os
from typing import Literal, Union
from dataclasses import dataclass

PlayerId = Literal[0, 1, 2, 3, 4]
ModelType = Literal["character", "perks", "item", "addons", "offering", "status", "points"]
FullModelType = str  # i.e. character__killer
Probability = float  # 0.0 a 1.0

Filename = str
PathToFolder = str
Path = str

SnippetWindow = tuple[int, int, int, int]  # Best estimation (1920x1080): (67,217) to (1015,897)
SnippetCoords = tuple[int, int, int, int]  # Best estimation (1920x1080): from 257 to 842 in intervals of 117
# SnippetInfo = tuple
EncodedInfo = tuple[int, int, tuple, int, tuple, int, int]

Boxes = Union[list[SnippetCoords], dict[str, list[SnippetCoords]]]

CropType = Literal["surv", "killer", "surv_player", "killer_player"]


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
    name: str
    src: PathToFolder
    dst: PathToFolder
    crops: dict[FullModelType, Union[list[SnippetWindow], list[SnippetCoords]]]
    are_absolute_paths: bool = False

    def make_abs_paths(self) -> None:
        if not self.are_absolute_paths:
            self.src = os.path.join(os.environ["DBDIE_MAIN_FD"], self.src)
            self.dst = os.path.join(os.environ["DBDIE_MAIN_FD"], self.dst)
            self.are_absolute_paths = True

    def get_rel_path(self, fd: Literal["src", "dst"]) -> PathToFolder:
        assert fd in {"src", "dst"}
        if self.are_absolute_paths:
            return os.path.relpath(getattr(self, fd), os.environ["DBDIE_MAIN_FD"])
        else:
            return getattr(self, fd)


AllSnippetCoords = dict[PlayerId, SnippetCoords]
AllSnippetInfo = dict[PlayerId, SnippetInfo]
