"""Survivor FullModelTypes."""

from dbdie_ml.options.PLAYER_TYPE import SURV
from dbdie_ml.options import MODEL_TYPES as MT

ADDONS    = f"{MT.ADDONS}__{SURV}"
CHARACTER = f"{MT.CHARACTER}__{SURV}"
ITEM      = f"{MT.ITEM}__{SURV}"
OFFERING  = f"{MT.OFFERING}__{SURV}"
PERKS     = f"{MT.PERKS}__{SURV}"
POINTS    = f"{MT.POINTS}__{SURV}"
PRESTIGE  = f"{MT.PRESTIGE}__{SURV}"
STATUS    = f"{MT.STATUS}__{SURV}"

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
