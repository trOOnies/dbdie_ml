"""Types of crops made by the Croppers."""

from dbdie_ml.options import PLAYER_TYPE as PT

SURV = PT.SURV
KILLER = PT.KILLER
SURV_PLAYER = f"{PT.SURV}_player"
KILLER_PLAYER = f"{PT.KILLER}_player"

ALL = [SURV, KILLER, SURV_PLAYER, KILLER_PLAYER]
DEFAULT_CROP_TYPES_SEQ = [[SURV, KILLER], [SURV_PLAYER, KILLER_PLAYER]]
