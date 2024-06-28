from typing import Optional, Union
from pydantic import BaseModel, ConfigDict
from dbdie_ml.classes import PlayerId, Probability

ALL_CHARS_IDS = {"all": 0, "killer": 1, "surv": 2}
ADDONS_IDS = {"none": 0, "killer": 1, "base": (2, 3, 4, 5, 6)}


class CharacterCreate(BaseModel):
    name: str
    is_killer: bool


class Character(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability = None
    is_killer: Optional[bool]


class PerkCreate(BaseModel):
    name: str
    character_id: int


class Perk(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability = None
    character_id: int
    is_for_killer: Optional[bool]


class Item(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability = None
    type_id: int


class Offering(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability = None
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
    proba: Probability = None
    type_id: int
    user_id: int


class Status(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    proba: Probability = None
    character_id: int
    is_dead: Optional[bool]


class FullCharacter(BaseModel):
    character: Character
    perks: list[Perk]
    addons: list[Addon]


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
        if self.is_killer is None:
            self.is_consistent = False
        elif any(not self._check_killer_consistency(perk) for perk in self.perks):
            self.is_consistent = False
        elif not self._check_killer_consistency(self.offering):
            self.is_consistent = False
        elif not self._check_item_consistency():
            self.is_consistent = False
        elif not self._check_addons_consistency():
            self.is_consistent = False
        elif not self._check_status_consistency():
            self.is_consistent = False
        else:
            self.is_consistent = True

    def _check_killer_consistency(self, obj: Union[Perk, Offering]) -> bool:
        return obj.is_for_killer is None or (obj.is_for_killer == self.is_killer)

    def _check_item_consistency(self) -> bool:
        return self.is_killer == (self.item.type_id == ADDONS_IDS["killer"])

    def _check_addons_consistency(self) -> bool:
        if self.is_killer:
            return all(
                a.type_id == ADDONS_IDS["killer"] or a.type_id == ADDONS_IDS["none"]
                for a in self.addons
            )
        else:
            return all(
                a.type_id == ADDONS_IDS["none"] or (
                    a.type_id in ADDONS_IDS["base"]
                    and a.type_id == self.item.type_id
                ) or a.type_id != ADDONS_IDS["killer"]
                for a in self.addons
            )

    def _check_status_consistency(self) -> bool:
        if self.status.character_id == ALL_CHARS_IDS["all"]:
            return True
        elif (self.status.character_id == ALL_CHARS_IDS["surv"]) == (not self.is_killer):
            return True
        elif (self.status.character_id == ALL_CHARS_IDS["killer"]) == self.is_killer:
            return True
        else:
            return False


class MatchOut(BaseModel):
    version: str
    players: list[PlayerOut]
    kills: Optional[int] = None
    is_consistent: Optional[bool] = None

    def model_post_init(self, __context) -> None:
        assert self.kills is None
        assert self.is_consistent is None
        self.check_consistency()
        self.kills = sum(pl.status.is_dead for pl in self.players[:4])

    def check_consistency(self) -> None:
        self.is_consistent = all(not pl.character.is_killer for pl in self.players[:4])
        self.is_consistent = self.is_consistent and (self.players[4].character.is_killer)
        self.is_consistent = self.is_consistent and all(pl.is_consistent for pl in self.players)
