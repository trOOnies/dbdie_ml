from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from classes.base import FullModelType, ModelType, PlayerType

SURV: "PlayerType" = "surv"
KILLER: "PlayerType" = "killer"

ALL: list["PlayerType"] = [SURV, KILLER]


def ifk_to_pt(is_for_killer: bool | None) -> "PlayerType":
    """Killer boolean ('is for killer') to PlayerType."""
    return ALL[int(is_for_killer)] if is_for_killer is not None else None


def pt_to_ifk(pt: "PlayerType") -> bool | None:
    """PlayerType to killer boolean ('is for killer')."""
    return (pt == KILLER) if pt is not None else None


def to_fmt(mt: "ModelType", is_for_killer: bool | None) -> "FullModelType":
    """To FullModelType."""
    return mt + ("" if is_for_killer is None else f"__{ifk_to_pt(is_for_killer)}")


def extract_mt_and_pt(fmt: "FullModelType") -> tuple["ModelType", "PlayerType"]:
    ix = fmt.find("__")
    if ix > -1:
        return fmt[:ix], fmt[ix + 2:]
    else:
        return deepcopy(fmt), None


def extract_mts_and_pts(
    fmts: list["FullModelType"]
) -> tuple[list["ModelType"], list["PlayerType"]]:
    mts_and_pts = [extract_mt_and_pt(fmt) for fmt in fmts]
    return (
        [tup[0] for tup in mts_and_pts],
        [tup[1] for tup in mts_and_pts],
    )
