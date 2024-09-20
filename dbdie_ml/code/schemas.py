"""Extra code for the schemas Python file."""

from typing import TYPE_CHECKING, Union

from dbdie_ml.options import PLAYER_TYPE as PT

if TYPE_CHECKING:
    from dbdie_ml.schemas.predictables import AddonOut, OfferingOut, PerkOut

ALL_CHARS_IDS = {"all": 0, PT.KILLER: 1, PT.SURV: 2}
ADDONS_IDS = {"none": 0, PT.KILLER: 1, "base": (2, 3, 4, 5, 6)}

# * PlayerOut


def check_killer_consistency(
    is_killer,
    obj: Union["OfferingOut", "PerkOut"],
) -> bool:
    return obj.ifk is None or (obj.ifk == is_killer)


def check_item_consistency(is_killer: bool, item_type_id: int) -> bool:
    # TODO: Decouple from addons
    return is_killer == (item_type_id == ADDONS_IDS[PT.KILLER])


def check_addons_consistency(
    is_killer: bool,
    addons: list["AddonOut"],
) -> bool:
    return all(
        a.type_id == ADDONS_IDS["none"]
        or ((a.type_id == ADDONS_IDS[PT.KILLER]) == is_killer)
        for a in addons
    )


def check_status_consistency(
    status_character_id: int,
    is_killer: bool,
) -> bool:
    return (
        status_character_id == ALL_CHARS_IDS["all"]
        or ((status_character_id == ALL_CHARS_IDS[PT.SURV]) == (not is_killer))
        or ((status_character_id == ALL_CHARS_IDS[PT.KILLER]) == is_killer)
    )
