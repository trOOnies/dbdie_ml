from typing import Optional
import datetime as dt

from pydantic import BaseModel

from dbdie_ml.classes.base import PlayerId
from dbdie_ml.classes.version import DBDVersion
from dbdie_ml.code.schemas import (
    check_addons_consistency,
    check_item_consistency,
    check_killer_consistency,
    check_status_consistency,
)
from dbdie_ml.schemas.predictables import (
    Addon,
    Character,
    Item,
    Offering,
    Perk,
    Status,
)

# * Version


class DBDVersionOut(BaseModel):
    id: int
    name: str
    release_date: Optional[dt.date]


# * Players


class PlayerIn(BaseModel):
    character_id: int
    perk_ids: list[int]
    item_id: int
    addon_ids: list[int]
    offering_id: int


class PlayerOut(BaseModel):
    id: PlayerId
    character: Character
    perks: list[Perk]
    item: Item
    addons: list[Addon]
    offering: Offering
    status: Status
    points: int
    is_consistent: Optional[bool] = None

    def model_post_init(self, __context) -> None:
        self.check_consistency()

    @property
    def is_killer(self) -> Optional[bool]:
        return self.character.is_killer

    def check_consistency(self) -> None:
        """Executes all consistency checks.
        It's purposefully separated so that in the future we could have
        customized self healing methods.
        """
        if self.is_killer is None:
            self.is_consistent = False
        elif any(
            not check_killer_consistency(self.is_killer, perk) for perk in self.perks
        ):
            self.is_consistent = False
        elif not check_killer_consistency(self.is_killer, self.offering):
            self.is_consistent = False
        elif not check_item_consistency(self.is_killer, self.item.type_id):
            self.is_consistent = False
        elif not check_addons_consistency(
            self.is_killer, self.addons, self.item.type_id
        ):
            self.is_consistent = False
        elif not check_status_consistency(self.status.character_id, self.is_killer):
            self.is_consistent = False
        else:
            self.is_consistent = True


# * Matches


class MatchCreate(BaseModel):
    filename: str
    match_date: Optional[dt.date]
    dbd_version: Optional[DBDVersion]
    special_mode: Optional[bool]
    user: Optional[str]
    extractor: Optional[str]
    kills: Optional[int]


class MatchOut(MatchCreate):
    id: int
    filename: str
    match_date: Optional[dt.date]
    dbd_version: Optional[DBDVersionOut]
    special_mode: Optional[bool]
    user: Optional[str]
    extractor: Optional[str]
    kills: Optional[int]
    date_created: dt.datetime
    date_modified: dt.datetime


# class LabelsCreate(BaseModel):
#     ...


class FullMatchOut(BaseModel):
    # TODO
    version: DBDVersion
    players: list[PlayerOut]
    kills: Optional[int] = None  # ! do not use
    is_consistent: Optional[bool] = None  # ! do not use

    def model_post_init(self, __context) -> None:
        assert self.kills is None
        assert self.is_consistent is None
        self.check_consistency()
        self.kills = sum(pl.status.is_dead for pl in self.players[:4])

    def check_consistency(self) -> None:
        """Executes all consistency checks."""
        self.is_consistent = all(not pl.character.is_killer for pl in self.players[:4])
        self.is_consistent = self.is_consistent and (
            self.players[4].character.is_killer
        )
        self.is_consistent = self.is_consistent and all(
            pl.is_consistent for pl in self.players
        )
