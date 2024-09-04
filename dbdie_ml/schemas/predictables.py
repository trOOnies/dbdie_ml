"""Pydantic schemas for the classes that are to be predicted"""

import datetime as dt
from typing import Optional

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from dbdie_ml.classes.base import Probability
from dbdie_ml.classes.version import DBDVersion


class DBDVersionCreate(BaseModel):
    """DBD game version creation schema"""

    name: str
    release_date: Optional[dt.date]


class DBDVersionOut(BaseModel):
    """DBD game version output schema"""

    id: int
    name: str
    common_name: str | None
    release_date: Optional[dt.date]


class CharacterCreate(BaseModel):
    """Character creation schema"""

    name: str
    is_killer: bool | None
    base_char_id: int | None = None  # Support for legendary outfits
    dbd_version_str: str | None = None
    emoji: str | None = None


class CharacterOut(BaseModel):
    """Character output schema"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    common_name: str | None
    proba: Probability | None = None
    is_killer: bool | None
    base_char_id: int | None
    dbd_version_id: int | None
    emoji: str | None


class PerkCreate(BaseModel):
    """Perk creation schema"""

    name: str
    character_id: int
    dbd_version_str: str | None = None
    emoji: str | None = None


class PerkOut(BaseModel):
    """Perk output schema"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_for_killer: bool | None
    dbd_version_id: int | None
    emoji: str | None


class ItemCreate(BaseModel):
    """Match item creation schema"""

    name: str
    type_id: int


class ItemOut(BaseModel):
    """Match item output schema"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int


class OfferingCreate(BaseModel):
    """Offering creation schema"""

    model_config = ConfigDict(from_attributes=True)

    name: str
    type_id: int
    user_id: int


class OfferingOut(BaseModel):
    """Offering output schema"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int
    user_id: int
    is_for_killer: bool | None


class AddonCreate(BaseModel):
    """Addon creation schema"""

    name: str
    type_id: int
    user_id: int
    dbd_version_str: str | None = None


class AddonOut(BaseModel):
    """Addon output schema"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    type_id: int
    user_id: int
    dbd_version_id: int | None


class StatusCreate(BaseModel):
    """Final player match status creation schema"""

    model_config = ConfigDict(from_attributes=True)

    name: str
    character_id: int
    emoji: str | None = None


class StatusOut(BaseModel):
    """Final player match status output schema"""

    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability | None = None
    character_id: int
    is_dead: bool | None
    emoji: str | None


class FullCharacterCreate(BaseModel):
    """Full character creation schema.
    Includes the creation perks and addons (if addons apply).
    DBD game version must already exist in the database.

    Note: This schema shouldn't be used for creating legendary outfits that
    use base_char_id. Please use CharacterCreate instead.
    """

    name: str
    is_killer: bool
    perk_names: list[str]
    addon_names: Optional[list[str]]
    dbd_version: DBDVersion
    common_name: str
    emoji: str

    @field_validator("perk_names")
    @classmethod
    def perks_must_be_three(cls, perks: list) -> list[str]:
        assert len(perks) == 3, "You must provide exactly 3 perk names"
        return perks

    @field_validator("emoji")
    @classmethod
    def emoji_len_le_4(cls, emoji: str) -> str:
        assert len(emoji) <= 4, "Emoji character-equivalence must be as most 4"
        return emoji

    @model_validator(mode="after")
    def check_total_addons(self):
        if self.is_killer:
            assert (
                self.addon_names is not None and len(self.addon_names) == 20
            ), "You must provide exactly 20 killer addon names"
        else:
            if self.addon_names is not None:
                assert not self.addon_names, "Survivors can't have addon names"
                self.addon_names = None
        return self


class FullCharacterOut(BaseModel):
    """Full character output schema"""

    character: CharacterOut
    perks: list[PerkOut]
    addons: list[AddonOut]
    common_name: str | None
    # proba: Probability | None = None
    is_killer: bool | None
    base_char_id: int | None
    dbd_version_id: int | None
    emoji: str | None

    @field_validator("perks")
    @classmethod
    def perks_must_be_three(cls, perks: list) -> list[PerkOut]:
        assert len(perks) == 3, "You must provide exactly 3 perk names"
        return perks

    @model_validator(mode="after")
    def check_total_addons(self):
        if self.character.is_killer:
            assert len(self.addons) == 20, "There can only be killers with 20 addons"
        elif not self.character.is_killer:
            assert not self.addons, "Survivors can't have addons"
        return self
