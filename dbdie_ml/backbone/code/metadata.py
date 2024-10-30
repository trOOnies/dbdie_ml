"""Extra code for the metadata classes."""

from typing import TYPE_CHECKING, Any

from backbone.classes.training import TrainingParams
from backbone.cropping import CropperSwarm, CropSettings

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType


def load_assertions(extr_name: str | None, vars: list[Any | None]) -> bool:
    is_trained_model = extr_name is not None
    assert all(
        is_trained_model == (v is None)
        for v in vars
    ), "Model params should be passed iif the model is being created."
    return is_trained_model


def form_metadata_dict(iem, dbdv_min, dbdv_max) -> dict:
    return (
        {
            k: getattr(iem, k)
            for k in [
                "id", "name", "fmt", "total_classes", "cps_name", "cs_name",
            ]
        }
        | {k: getattr(iem, f"_{k}") for k in ["norm_means", "norm_std"]}
        | {
            "dbdv_max": dbdv_max,
            "dbdv_min": dbdv_min,
            "img_size": list(iem.img_size),
            "training": TrainingParams(**iem.training_params),
        }
    )


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
