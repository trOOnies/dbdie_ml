"""Survivor FullModelTypes"""

from dbdie_ml.options import COMMON_FMT

ADDONS = "addons__surv"
CHARACTER = "character__surv"
ITEM = "item__surv"
OFFERING = "offering__surv"
PERKS = "perks__surv"
POINTS = f"{COMMON_FMT.POINTS}__surv"
PRESTIGE = f"{COMMON_FMT.PRESTIGE}__surv"
STATUS = f"{COMMON_FMT.STATUS}__surv"

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
