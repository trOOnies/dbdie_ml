"""Extra code for the schemas Python file"""

from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dbdie_ml.schemas.predictables import AddonOut, OfferingOut, PerkOut

ALL_CHARS_IDS = {"all": 0, "killer": 1, "surv": 2}
ADDONS_IDS = {"none": 0, "killer": 1, "base": (2, 3, 4, 5, 6)}


# * PlayerOut


def check_killer_consistency(is_killer, obj: Union["OfferingOut", "PerkOut"]) -> bool:
    return obj.is_for_killer is None or (obj.is_for_killer == is_killer)


def check_item_consistency(is_killer: bool, item_type_id: int) -> bool:
    # TODO: Decouple from addons
    return is_killer == (item_type_id == ADDONS_IDS["killer"])


def check_addons_consistency(
    is_killer: bool,
    addons: list["AddonOut"],
) -> bool:
    return all(
        a.type_id == ADDONS_IDS["none"]
        or ((a.type_id == ADDONS_IDS["killer"]) == is_killer)
        for a in addons
    )


def check_status_consistency(status_character_id: int, is_killer: bool) -> bool:
    return (
        status_character_id == ALL_CHARS_IDS["all"]
        or ((status_character_id == ALL_CHARS_IDS["surv"]) == (not is_killer))
        or ((status_character_id == ALL_CHARS_IDS["killer"]) == is_killer)
    )
