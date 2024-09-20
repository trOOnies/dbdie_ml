from dbdie_ml.options import MODEL_TYPES

CHARACTER = ["character"]
PERKS     = ["perk_0", "perk_1", "perk_2", "perk_3"]
ITEM      = ["item"]
ADDONS    = ["addon_0", "addon_1"]
OFFERING  = ["offering"]
STATUS    = ["status"]
POINTS    = ["points"]
PRESTIGE  = ["prestige"]

ALL = [
    CHARACTER,
    PERKS,
    ITEM,
    ADDONS,
    OFFERING,
    STATUS,
    # POINTS,
    # PRESTIGE,
]
MT_TO_COLS = {
    MODEL_TYPES.CHARACTER: CHARACTER,
    MODEL_TYPES.PERKS: PERKS,
    MODEL_TYPES.ITEM: ITEM,
    MODEL_TYPES.ADDONS: ADDONS,
    MODEL_TYPES.OFFERING: OFFERING,
    MODEL_TYPES.STATUS: STATUS,
    # MODEL_TYPES.POINTS: POINTS,
    # MODEL_TYPES.PRESTIGE: PRESTIGE,
}
