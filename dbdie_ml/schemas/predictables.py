import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict

from dbdie_ml.classes.base import Probability


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
    is_for_killer: Optional[bool]


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
    is_dead: Optional[bool]


class StatusOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_dead: Optional[bool]


class FullCharacterOut(BaseModel):
    character: CharacterOut
    perks: list[PerkOut]
    addons: list[AddonOut]
