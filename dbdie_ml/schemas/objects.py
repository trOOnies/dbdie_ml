"""Pydantic schemas for class objects related to DBDIE."""

import datetime as dt
from pydantic import BaseModel


class UserOut(BaseModel):
    """DBDIE user output schema."""

    id: int
    name: str


class CropperSwarmOut(BaseModel):
    """DBDIE CropperSwarm output schema."""

    id:            int
    name:          str
    user_id:       int
    img_width:     int
    img_height:    int
    dbdv_min_id:   int
    dbdv_max_id:   int | None
    is_for_killer: bool | None


class FullModelTypeOut(BaseModel):
    """DBDIE full model type output schema."""

    id:            int
    name:          str
    model_type:    str
    is_for_killer: bool


class ModelOut(BaseModel):
    """DBDIE IEModel output schema."""

    id:                  int
    name:                str
    user_id:             int
    fmt_id:              int
    cropper_swarm_id:    int
    dbdv_min_id:         int
    dbdv_max_id:         int | None
    special_mode:        bool | None
    date_created:        dt.datetime
    date_modified:       dt.datetime
    date_last_retrained: dt.date


class ExtractorOut(BaseModel):
    """DBDIE InfoExtractor output schema."""

    id:                  int
    name:                str
    user_id:             int
    dbdv_min_id:         int
    dbdv_max_id:         int | None
    special_mode:        bool | None
    cropper_swarm_id:    int
    mid_addons:          int | None
    mid_character:       int | None
    mid_item:            int | None
    mid_offering:        int | None
    mid_perks:           int | None
    mid_points:          int | None
    mid_status:          int | None
    date_created:        dt.datetime
    date_modified:       dt.datetime
    date_last_retrained: dt.date
