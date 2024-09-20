"""Extra code for crop_settings."""

from itertools import combinations

from dbdie_ml.classes.base import CropCoords


def check_overboard(name, allow, img_size, crops) -> None:
    if not allow["overboard"]:
        img_crop = CropCoords(0, 0, img_size[0], img_size[1])
        assert all(
            crop.is_fully_inside(img_crop)
            for crops in crops.values()
            for crop in crops
        ), f"[ct={name}] Crop out of bounds"


def check_positivity(name, crop_shapes) -> None:
    assert all(
        (cs[0] > 0) and (cs[1] > 0) for cs in crop_shapes.values()
    ), f"[ct={name}] Coord sizes must be positive"


def check_shapes(name, crops, crop_shapes) -> None:
    assert all(
        c.shape == crop_shapes[name]
        for name, crops in crops.items()
        for c in crops
    ), f"[ct={name}] All crops must have the same shape"


def check_overlap(name, allow, crops) -> None:
    if not allow["overlap"]:
        assert all(
            not c1.check_overlap(c2)
            for crops in crops.values()
            for c1, c2 in combinations(crops, 2)
        ), f"[ct={name}] Crops cannot overlap"
