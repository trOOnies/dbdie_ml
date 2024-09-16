"""Pydantic schemas for the grouping classes"""

from __future__ import annotations

import datetime as dt
from typing import Optional

from pydantic import BaseModel, Field, ValidationInfo, field_validator

from dbdie_ml.classes.base import ModelType, PlayerId
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
    DBDVersionOut,
    ItemOut,
    OfferingOut,
    PerkOut,
    StatusOut,
)
from dbdie_ml.options.MODEL_TYPES import ALL as ALL_MT

# * Players


class PlayerIn(BaseModel):
    """Player input schema to be used for creating labels"""
    id: int
    character_id: int | None = Field(None, ge=0)
    perk_ids:     list[int] | None = None
    item_id:      int | None = Field(None, ge=0)
    addon_ids:    list[int] | None = None
    offering_id:  int | None = Field(None, ge=0)
    status_id:    int | None = Field(None, ge=0)
    points:       int | None = Field(None, ge=0)
    prestige:     int | None = Field(None, ge=0, le=100)

    @classmethod
    def from_labels(cls, labels) -> PlayerIn:
        perks = [labels.perk_0, labels.perk_1, labels.perk_2, labels.perk_3]
        perks = perks if all(p is not None for p in perks) else None

        addons = [labels.addon_0, labels.addon_1]
        addons = addons if all(a is not None for a in addons) else None

        player = PlayerIn(
            id=labels.player_id,
            character_id=labels.character,
            perk_ids=perks,
            item_id=labels.item,
            addon_ids=addons,
            offering_id=labels.offering,
            status_id=labels.status,
            points=labels.points,
            prestige=labels.prestige,
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

    def filled_predictables(self) -> list[str]:
        """Return predictables' names that are filled (fps)."""
        d = dict(self)
        del d["id"]
        return [k for k, v in d.items() if v is not None]

    def to_sqla(self, fps: list[str]) -> dict:
        """To dict for the 'Labels' SQLAlchemy model."""
        sqld = {"player_id": self.id}
        single_ids = {
            "character": self.character_id,
            "item": self.item_id,
            "offering": self.offering_id,
            "status": self.status_id,
        }
        sqld = sqld | {
            k: v for k, v in single_ids.items()
            if getattr(self, f"{k}_id") in fps
        }
        sqld = sqld | {
            f"{k}_mckd": True for k in single_ids
            if getattr(self, f"{k}_id") in fps
        }

        if "points" in fps:
            sqld = sqld | {"points": self.points, "points_mckd": True}
        if "prestige" in fps:
            sqld = sqld | {"prestige": self.prestige, "prestige_mckd": True}
        if "perk_ids" in fps:
            sqld = sqld | {
                "perk_0": self.perk_ids[0] if self.perk_ids is not None else None,
                "perk_1": self.perk_ids[1] if self.perk_ids is not None else None,
                "perk_2": self.perk_ids[2] if self.perk_ids is not None else None,
                "perk_3": self.perk_ids[3] if self.perk_ids is not None else None,
                "perks_mckd": True,
            }
        if "addon_ids" in fps:
            sqld = sqld | {
                "addon_0": self.addon_ids[0] if self.addon_ids is not None else None,
                "addon_1": self.addon_ids[1] if self.addon_ids is not None else None,
                "addons_mckd": True,
            }

        return sqld

    @staticmethod
    def flatten_predictables(info: dict) -> dict:
        new_info = {
            f"{k}_mckd": v
            for k, v in info["manual_checks"]["predictables"].items()
        }
        del info["manual_checks"]
        return info | new_info


class PlayerOut(BaseModel):
    """Player output schema as seen in created labels"""

    id:        PlayerId
    character: CharacterOut
    perks:     list[PerkOut]
    item:      ItemOut
    addons:    list[AddonOut]
    offering:  OfferingOut
    status:    StatusOut
    points:    int
    prestige:  int
    is_consistent: Optional[bool] = None

    def model_post_init(self, __context) -> None:
        self.check_consistency()

    @property
    def is_killer(self) -> Optional[bool]:
        return self.character.is_killer

    def check_consistency(self) -> None:
        """Execute all consistency checks.
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
        elif not check_addons_consistency(self.is_killer, self.addons):
            self.is_consistent = False
        elif not check_status_consistency(self.status.character_id, self.is_killer):
            self.is_consistent = False
        else:
            self.is_consistent = True


# * Matches


class ManualChecksIn(BaseModel):
    """Manual predictables checks input schema."""
    addons    : bool | None = None
    character : bool | None = None
    item      : bool | None = None
    offering  : bool | None = None
    perks     : bool | None = None
    points    : bool | None = None
    prestige  : bool | None = None
    status    : bool | None = None
    is_init     : bool = False  # ! do not use
    in_progress : bool = False  # ! do not use
    completed   : bool = False  # ! do not use

    def model_post_init(self, __context) -> None:
        assert not self.is_init
        assert not self.in_progress
        assert not self.completed

        self.is_init = any(c is not None for c in self.checks)
        conds = [(c is not None and c) for c in self.checks]
        self.in_progress = any(conds)
        self.completed = all(conds)

    @property
    def checks(self) -> list[bool | None]:
        return [
            self.addons,
            self.character,
            self.item,
            self.offering,
            self.perks,
            self.points,
            self.prestige,
            self.status,
        ]

    @staticmethod
    def model_to_cols(labels_model):
        return [
            labels_model.addons_mckd,
            labels_model.character_mckd,
            labels_model.item_mckd,
            labels_model.offering_mckd,
            labels_model.perks_mckd,
            labels_model.points_mckd,
            labels_model.prestige_mckd,
            labels_model.status_mckd,
        ]

    def get_filters_conds(self, labels_model) -> list[tuple]:
        return [
            (col, chk)
            for col, chk in zip(
                self.model_to_cols(labels_model),
                self.checks,
            )
            if chk is not None
        ]


class ManualChecksOut(BaseModel):
    """Manual predictables checks output schema."""
    predictables : dict["ModelType", bool | None]
    is_init      : bool = False  # ! do not use
    in_progress  : bool = False  # ! do not use
    completed    : bool = False  # ! do not use

    @property
    def checks(self) -> list[bool | None]:
        return [self.predictables[mt] for mt in ALL_MT]

    def model_post_init(self, __context) -> None:
        assert not self.is_init
        assert not self.in_progress
        assert not self.completed

        # Dict must have exactly all expected keys
        assert set(mt for mt in ALL_MT) == set(self.predictables.keys())

        self.is_init = any(c is not None for c in self.checks)
        conds = [(c is not None and c) for c in self.checks]
        self.in_progress = any(conds)
        self.completed = all(conds)

    @classmethod
    def from_labels(cls, labels) -> ManualChecksOut:
        mc_out = ManualChecksOut(
            predictables={
                "addons": labels.addons_mckd,
                "character": labels.character_mckd,
                "item": labels.item_mckd,
                "offering": labels.offering_mckd,
                "perks": labels.perks_mckd,
                "prestige": labels.prestige_mckd,
                "points": labels.points_mckd,
                "status": labels.status_mckd,
            }
        )
        return mc_out


class MatchCreate(BaseModel):
    """DBD match creation schema."""

    filename:     str
    match_date:   dt.date | None = None
    dbd_version:  DBDVersion | None = None
    special_mode: bool | None = None
    user_id:      int | None = None
    extractor_id: int | None = None
    kills:        int | None = Field(None, ge=0, le=4)


class MatchOut(BaseModel):
    """DBD match output schema."""

    id:            int
    filename:      str
    match_date:    dt.date | None
    dbd_version:   DBDVersionOut | None
    special_mode:  bool | None
    kills:         int | None
    date_created:  dt.datetime
    date_modified: dt.datetime
    user_id:       int | None
    extractor_id:  int | None


class VersionedFolderUpload(BaseModel):
    """DBD-versioned folder to upload."""

    dbd_version:  DBDVersion
    special_mode: Optional[bool] = None


class VersionedMatchOut(BaseModel):
    """DBD match simplified output schema for DBD-versioned folder upload."""

    id:           int
    filename:     str
    match_date:   Optional[dt.date]
    dbd_version:  DBDVersionOut
    special_mode: Optional[bool]


class LabelsCreate(BaseModel):
    """Labels creation schema."""

    match_id:         int
    player:           PlayerIn
    user_id:          int | None = None
    extractor_id:     int | None = None
    manually_checked: bool | None = None


class LabelsOut(BaseModel):
    """Labels output schema."""

    match_id:       int
    player:         PlayerIn
    date_modified:  dt.datetime
    user_id:        int | None
    extractor_id:   int | None
    manual_checks:  ManualChecksOut

    @classmethod
    def from_labels(cls, labels) -> LabelsOut:
        labels_out = LabelsOut(
            match_id=labels.match_id,
            player=PlayerIn.from_labels(labels),
            date_modified=labels.date_modified,
            user_id=labels.user_id,
            extractor_id=labels.extractor_id,
            manual_checks=ManualChecksOut.from_labels(labels),
        )
        return labels_out


class FullMatchOut(BaseModel):
    """Labeled DBD match output schema."""

    # TODO
    version:       DBDVersion
    players:       list[PlayerOut]
    kills:         int = Field(-1, ge=-1, le=4)  # ! do not use
    is_consistent: bool = True  # ! do not use

    def model_post_init(self, __context) -> None:
        assert self.kills == -1
        assert self.is_consistent
        self.check_consistency()
        self.kills = sum(
            pl.status.is_dead if pl.status.is_dead is not None else 0
            for pl in self.players[:4]
        )

    def check_consistency(self) -> None:
        """Execute all consistency checks."""
        self.is_consistent = all(not pl.character.is_killer for pl in self.players[:4])
        self.is_consistent = (
            self.is_consistent
            and (self.players[4].character.is_killer is not None)
            and self.players[4].character.is_killer
        )
        self.is_consistent = self.is_consistent and all(
            pl.is_consistent for pl in self.players
        )
