from typing import TYPE_CHECKING, Tuple, Any, Dict, List, Literal, Union
from dataclasses import dataclass
if TYPE_CHECKING:
    from PIL import Image

PlayerId = Literal[0, 1, 2, 3, 4]
ModelType = Literal["character", "perks", "item", "addons", "offering", "status", "points"]

Filename = str
PathToFolder = str

SnippetWindow = Tuple[int, int, int, int]  # Best estimation (1920x1080): (67,217) to (1015,897)
SnippetCoords = Tuple[int, int, int, int]  # Best estimation (1920x1080): from 257 to 842 in intervals of 117
# SnippetInfo = Tuple
EncodedInfo = Tuple[int, int, tuple, int, tuple, int, int]

Boxes = Union[List[SnippetCoords], Dict[str, List[SnippetCoords]]]
CropSettingsKey = Literal["src", "dst", "crops"]
CropSettings = Dict[CropSettingsKey, Union[str, Boxes]]


@dataclass
class SnippetInfo:
    character_id: int
    perks_ids: Tuple[int, int, int, int]
    item_id: int
    addons_ids: Tuple[int, int]
    offering_id: int
    status_id: int
    points: int


@dataclass
class SnippetDetector:
    model: Any

    def predict(self, img: "Image")-> List[SnippetCoords]:
        return self.model.predict(img)


@dataclass
class SnippetProcessor:
    model: Any

    def predict(self, snippet: SnippetCoords)-> SnippetInfo:
        return self.model.predict(snippet)


AllSnippetCoords = Dict[PlayerId, SnippetCoords]
AllSnippetInfo = Dict[PlayerId, SnippetInfo]
