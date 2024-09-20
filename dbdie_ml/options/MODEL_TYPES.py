"""Types of predictables that necessarily end up as different models.
They aren't 'full' (FullModelTypes) because they lack the surv / killer suffix.
"""

from dbdie_ml.options import COMMON_FMT

ADDONS    = "addons"
CHARACTER = "character"
ITEM      = "item"
OFFERING  = "offering"
PERKS     = "perks"
POINTS    = COMMON_FMT.POINTS
PRESTIGE  = COMMON_FMT.PRESTIGE
STATUS    = COMMON_FMT.STATUS

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
