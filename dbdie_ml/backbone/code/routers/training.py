"""Extra code for the training router."""

import pandas as pd

from backbone.endpoints import getr


def get_matches(ie, use_dbdvr: bool = True) -> pd.DataFrame:
    """Get matches for training or extraction."""
    matches = getr("/matches", api=True, params={"limit": 300_000})
    matches = ie.filter_matches_with_dbdv(matches, use_dbdvr=use_dbdvr)
    assert matches, "No matches intersect with the extractor's DBDVersionRange."
    return pd.DataFrame(
            [
            {"match_id": m["id"], "filename": m["filename"]}
            for m in matches
        ]
    )


def to_trained_ie_schema(ie, cps_id, now, today):
    return ie.to_schema(
        {
            "user_id": 1,  # TODO
            "special_mode": False,  # TODO
            "cps_id": cps_id,
            "date_created": now,
            "date_modified": now,
            "date_last_trained": today,
        }
    )


def to_trained_model_schemas(ie, cps_id, now, today):
    fmts_ids = getr("/fmt", api=True, params={"limit": 300})
    fmts_ids = {d["name"]: d["id"] for d in fmts_ids}
    return ie.models_to_schemas(
        {
            fmt: {
                "user_id": 1,  # TODO
                "fmt_id": fmts_ids[fmt],
                "cps_id": cps_id,
                "special_mode": False,  # TODO
                "date_created": now,
                "date_modified": now,
                "date_last_trained": today,
            }
            for fmt in ie._models
        }
    )
