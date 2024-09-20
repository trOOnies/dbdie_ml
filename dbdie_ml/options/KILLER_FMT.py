"""Killer FullModelTypes."""

from dbdie_ml.options import MODEL_TYPES as MT
from dbdie_ml.options.PLAYER_TYPE import KILLER

ADDONS    = f"{MT.ADDONS}__{KILLER}"
CHARACTER = f"{MT.CHARACTER}__{KILLER}"
ITEM      = f"{MT.ITEM}__{KILLER}"
OFFERING  = f"{MT.OFFERING}__{KILLER}"
PERKS     = f"{MT.PERKS}__{KILLER}"
POINTS    = f"{MT.POINTS}__{KILLER}"
PRESTIGE  = f"{MT.PRESTIGE}__{KILLER}"
STATUS    = f"{MT.STATUS}__{KILLER}"

ALL = [
    ADDONS,
    CHARACTER,
    ITEM,
    OFFERING,
    PERKS,
    POINTS,
    PRESTIGE,
    STATUS,
]
