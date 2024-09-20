"""DBDIE classes for extract purposes."""

from dataclasses import dataclass

from dbdie_ml.classes.base import CropCoordsRaw, PlayerId


@dataclass(eq=True)
class CropCoords:
    """Crop coordinates in relation to its source image or crop."""

    left: int
    top: int
    right: int
    bottom: int
    index: int = 0

    def raw(self) -> CropCoordsRaw:
        return (self.left, self.top, self.right, self.bottom)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            result = self.raw()[self.index]
        except IndexError:
            raise StopIteration
        self.index += 1
        return result

    @property
    def shape(self) -> tuple[int, int]:
        return (self.right - self.left, self.bottom - self.top)

    @property
    def size(self) -> int:
        return (self.right - self.left) * (self.bottom - self.top)

    def is_fully_inside(self, cc) -> bool:
        """Checks if the crop is fully inside the 'cc' crop."""
        return (
            (cc.left <= self.left)
            and (self.right <= cc.right)
            and (cc.top <= self.top)
            and (self.bottom <= cc.bottom)
        )

    def check_overlap(self, cc) -> int:
        """Checks if 2 crops of the SAME SIZE overlap."""
        return not (
            (cc.right <= self.left)
            or (self.right <= cc.left)
            or (cc.bottom <= self.top)
            or (self.bottom <= cc.top)
        )


@dataclass
class PlayerInfo:
    """Integer-encoded DBD information of a player snippet."""

    character_id: int
    perks_ids: tuple[int, int, int, int]
    item_id: int
    addons_ids: tuple[int, int]
    offering_id: int
    status_id: int
    points: int


PlayersCropCoords = dict[PlayerId, CropCoords]
PlayersInfoDict = dict[PlayerId, PlayerInfo]
