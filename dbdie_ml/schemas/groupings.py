"""Pydantic schemas for the grouping classes"""

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from dbdie_ml.classes.base import PlayerId
from dbdie_ml.classes.version import DBDVersion
from dbdie_ml.code.schemas import (
    check_addons_consistency,
    check_item_consistency,
    check_killer_consistency,
    check_status_consistency,
)
from dbdie_ml.schemas.predictables import (
    AddonOut,
    CharacterOut,
    ItemOut,
    OfferingOut,
    PerkOut,
    StatusOut,
)

# * Version


class DBDVersionOut(BaseModel):
    id: int
    name: str
    release_date: Optional[dt.date]


# * Players


class PlayerIn(BaseModel):
    """Player to be"""

    id: PlayerId
    character_id: Optional[int] = Field(None, ge=0)
    perk_ids: Optional[list[int]] = None
    item_id: Optional[int] = Field(None, ge=0)
    addon_ids: Optional[list[int]] = None
    offering_id: Optional[int] = Field(None, ge=0)
    status_id: Optional[int] = Field(None, ge=0)
    points: Optional[int] = Field(None, ge=0)

    @classmethod
    def from_labels(cls, labels):
        player = PlayerIn(
            id=labels.player_id,
            character_id=labels.character,
            perk_ids=[labels.perk_0, labels.perk_1, labels.perk_2, labels.perk_3],
            item_id=labels.item,
            addon_ids=[labels.addon_0, labels.addon_1],
            offering_id=labels.offering,
            status_id=labels.status,
            points=labels.points,
        )
        return player

    @field_validator("perk_ids", "addon_ids")
    @classmethod
    def count_is(
        cls,
        v: Optional[list[int]],
        info: ValidationInfo,
    ) -> Optional[list[int]]:
        if v is None:
            return v
        elif info.field_name == "perk_ids":
            assert len(v) == 4, "There can only be 4 perks or None"
            assert all(p >= 0 for p in v), "Perk ids can't be negative"
        elif info.field_name == "addon_ids":
            assert len(v) == 2, "There can only be 2 addons or None"
            assert all(p >= 0 for p in v), "Addon ids can't be negative"
        else:
            raise NotImplementedError
        return v


class PlayerOut(BaseModel):
    id: PlayerId
    character: CharacterOut
    perks: list[PerkOut]
    item: ItemOut
    addons: list[AddonOut]
    offering: OfferingOut
    status: StatusOut
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
    match_date: Optional[dt.date] = None
    dbd_version: Optional[DBDVersion] = None
    special_mode: Optional[bool] = None
    user: Optional[str] = None
    extractor: Optional[str] = None
    kills: Optional[int] = Field(None, ge=0, le=4)


class MatchOut(BaseModel):
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


class LabelsCreate(BaseModel):
    match_id: int
    player: PlayerIn


class LabelsOut(BaseModel):
    match_id: int
    player: PlayerIn
    date_modified: dt.datetime


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