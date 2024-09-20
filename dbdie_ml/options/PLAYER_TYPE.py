from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes.base import PlayerType

SURV: "PlayerType" = "surv"
KILLER: "PlayerType" = "killer"

ALL: list["PlayerType"] = [SURV, KILLER]


def ifk_to_pt(is_for_killer: bool) -> "PlayerType":
    """Killer boolean ('is for killer') to PlayerType."""
    return ALL[int(is_for_killer)]
