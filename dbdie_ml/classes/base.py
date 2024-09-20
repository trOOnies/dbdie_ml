"""Base DBDIE classes mainly for typing reasons."""

from typing import Literal

# General
PlayerId = Literal[0, 1, 2, 3, 4]
ModelType = Literal[
    "character", "perks", "item", "addons", "offering", "status", "points", "prestige"
]
PlayerStrict = Literal["killer", "surv"]
PlayerType = PlayerStrict | None
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
CropCoordsRaw = tuple[int, int, int, int]
EncodedInfo = tuple[int, int, tuple, int, tuple, int, int]

CropType = Literal["surv", "killer", "surv_player", "killer_player"]

# Labels
LabelId = int
LabelName = str
LabelRef = dict[LabelId, LabelName]
