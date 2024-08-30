"""Pydantic schemas for the classes that are to be predicted"""

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dbdie_ml.classes.base import Probability
from dbdie_ml.classes.version import DBDVersion


class DBDVersionCreate(BaseModel):
    name: str
    release_date: Optional[dt.date]


class DBDVersionOut(BaseModel):
    id: int
    name: str
    release_date: Optional[dt.date]


class CharacterCreate(BaseModel):
    name: str
    is_killer: Optional[bool]
    base_char_id: Optional[int] = None
    dbd_version: Optional[str] = None


class CharacterOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    is_killer: Optional[bool]
    base_char_id: Optional[int]
    dbd_version_id: Optional[int]


class PerkCreate(BaseModel):
    name: str
    character_id: int


class PerkOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_for_killer: Optional[bool]


class ItemCreate(BaseModel):
    name: str
    type_id: int


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int


class OfferingCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    type_id: int
    user_id: int


class OfferingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int
    user_id: int
    is_for_killer: Optional[bool]


class AddonCreate(BaseModel):
    name: str
    type_id: int
    user_id: int


class AddonOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int
    user_id: int


class StatusCreate(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    name: str
    character_id: int


class StatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_dead: Optional[bool]


class FullCharacterCreate(BaseModel):
    name: str
    is_killer: bool
    perk_names: list[str]
    dbd_version: DBDVersion
    addon_names: Optional[list[str]]

    @field_validator("perk_names")
    @classmethod
    def perks_must_be_three(cls, perks: list) -> list[str]:
        assert len(perks) == 3, "You must provide exactly 3 perk names"
        return perks

    @model_validator(mode="after")
    def check_total_addons(self):
        if self.is_killer:
            assert (
                self.addon_names is not None and len(self.addon_names) == 20
            ), "You must provide exactly 20 killer addon names"
        else:
            assert (
                self.addon_names is None or not self.addon_names
            ), "Survivors can't have addon names"
        return self


class FullCharacterOut(BaseModel):
    character: CharacterOut
    perks: list[PerkOut]
    addons: list[AddonOut]
