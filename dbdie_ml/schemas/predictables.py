from typing import Optional
from pydantic import BaseModel, ConfigDict
import datetime as dt

from dbdie_ml.classes.base import Probability


class DBDVersionOut(BaseModel):
    id: int
    name: str
    release_date: Optional[dt.date]


class CharacterCreate(BaseModel):
    name: str
    is_killer: bool


class Character(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    is_killer: Optional[bool] = None
    base_char_id: Optional[int] = None
    dbd_version_id: Optional[int] = None


class PerkCreate(BaseModel):
    name: str
    character_id: int


class Perk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_for_killer: Optional[bool]


class Item(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int


class Offering(BaseModel):
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


class Addon(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int
    user_id: int


class Status(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_dead: Optional[bool] = None


class FullCharacter(BaseModel):
    character: Character
    perks: list[Perk]
    addons: list[Addon]
