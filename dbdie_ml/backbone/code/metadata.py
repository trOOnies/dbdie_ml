"""Extra code for the metadata classes."""

from typing import TYPE_CHECKING

from backbone.cropping import CropperSwarm, CropSettings

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType


def process_metadata(metadata: dict) -> tuple:
    """Process model metadata when ifk is not None."""
    assert isinstance(metadata["cs_name"], str)
    assert isinstance(metadata["fmt"], str)

    cs_dict = CropSettings.make_cs_dict(metadata["cps_name"])
    cs = cs_dict[metadata["cs_name"]]
    crop = metadata["fmt"]

    metadata["img_size"] = cs.crop_shapes[crop]
    del metadata["fmt"]
    return metadata, cs


def process_metadata_ifk_none(metadata: dict) -> tuple:
    """Process model metadata when ifk is None."""
    assert isinstance(metadata["cs_name"], list)
    assert isinstance(metadata["fmt"], list)
    assert len(metadata["cs_name"]) == 2
    assert len(metadata["fmt"]) == 2

    cs_dict = CropSettings.make_cs_dict(metadata["cps_name"])
    both_cs = [
        cs_dict[cs_str]
        for cs_str in metadata["cs_name"]
    ]
    crop_shapes = [
        cs.crop_shapes[crop]
        for cs, crop in zip(both_cs, metadata["fmt"])
    ]
    assert crop_shapes[0] == crop_shapes[1]

    metadata["img_size"] = crop_shapes[0]
    del metadata["fmt"]

    return metadata, both_cs[0]


def patch_untrained_metadata(
    metadata: dict,
    sdbdv_class,
    fmt: "FullModelType",
    model_id: int,
    cps_name: str,
    total_classes: int,
) -> dict:
    cps = CropperSwarm.from_register(cps_name)
    cs = cps.get_cs_that_contains_fmt(fmt)

    dbdv_min, dbdv_max = sdbdv_class.dbdvr_to_saved_dbdvs(cps.dbdvr)

    return metadata | {
        "id": model_id,
        "total_classes": total_classes,
        "img_size": cs.crop_shapes[fmt],
        "dbdv_max": dbdv_max,
        "dbdv_min": dbdv_min,
        "cps_name": cps_name,
    }
