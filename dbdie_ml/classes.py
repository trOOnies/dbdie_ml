from typing import TYPE_CHECKING, Any, Literal, Union
from dataclasses import dataclass
if TYPE_CHECKING:
    from PIL import Image

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
CropSettingsKey = Literal["src", "dst", "crops"]
CropSettings = dict[CropSettingsKey, Union[str, Boxes]]


@dataclass
class SnippetInfo:
    character_id: int
    perks_ids: tuple[int, int, int, int]
    item_id: int
    addons_ids: tuple[int, int]
    offering_id: int
    status_id: int
    points: int


@dataclass
class SnippetDetector:
    model: Any

    def predict(self, img: "Image")-> list[SnippetCoords]:
        return self.model.predict(img)


@dataclass
class SnippetProcessor:
    model: Any

    def predict(self, snippet: SnippetCoords)-> SnippetInfo:
        return self.model.predict(snippet)


AllSnippetCoords = dict[PlayerId, SnippetCoords]
AllSnippetInfo = dict[PlayerId, SnippetInfo]
