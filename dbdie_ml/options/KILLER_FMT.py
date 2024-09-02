"""Killer FullModelTypes"""

from dbdie_ml.options import COMMON_FMT

ADDONS = "addons__killer"
CHARACTER = "character__killer"
ITEM = "item__killer"
OFFERING = "offering__killer"
PERKS = "perks__killer"
POINTS = f"{COMMON_FMT.POINTS}__killer"
PRESTIGE = f"{COMMON_FMT.PRESTIGE}__killer"
STATUS = f"{COMMON_FMT.STATUS}__killer"

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
