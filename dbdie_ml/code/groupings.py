"""Extra code for groupings classes."""

from dbdie_ml.options import MODEL_TYPES as MT
from dbdie_ml.options import COLUMNS as SQL_COLS


def predictables_for_sqld(player, fps: list[str]) -> dict:
    """Predictables to add to SQL dict."""
    sqld = {
        mt: getattr(player, MT.MTS_TO_ID_NAMES[mt])
        for mt in MT.UNIQUE_PER_PLAYER
        if MT.MTS_TO_ID_NAMES[mt] in fps
    }
    sqld = sqld | {f"{mt}_mckd": True for mt in sqld}

    if MT.MTS_TO_ID_NAMES[MT.PERKS] in fps:
        cond = player.perk_ids is not None
        sqld = sqld | {
            f"perk_{i}": pid if cond else None
            for i, pid in enumerate(player.perk_ids)
        } | {"perks_mckd": True}

    if MT.MTS_TO_ID_NAMES[MT.ADDONS] in fps:
        cond = player.addon_ids is not None
        sqld = sqld | {
            f"addon_{i}": aid if cond else None
            for i, aid in enumerate(player.addon_ids)
        } | {"addons_mckd": True}

    return sqld


def check_strict(strict: bool, sqld: dict) -> None:
    if strict:
        all_fps_cols = [
            cols for cols in SQL_COLS.ALL
            if any(c in sqld for c in cols)
        ]
        assert len(all_fps_cols) == 1, "There can't be different model types in strict mode"
