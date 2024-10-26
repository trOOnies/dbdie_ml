"""Extra code for crop_settings."""

from itertools import combinations

from dbdie_classes.extract import CropCoords
from dbdie_classes.schemas.helpers import DBDVersionRange

from backbone.endpoints import getr

# * Check crop shapes


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


def process_img_size(data: dict, depends_on) -> dict:
    if depends_on is not None:
        assert isinstance(data["img_size"], dict)
        assert depends_on.name == data["img_size"]["cs"]
        data["img_size"] = depends_on.crop_shapes[data["img_size"]["fmt"]]
    else:
        assert isinstance(data["img_size"], list)
        assert len(data["img_size"]) == 2
        data["img_size"] = tuple(data["img_size"])
    return data


# * Instantiation


def get_dbdvr(dbdv_list: list[str]) -> DBDVersionRange:
    dbdv_min, dbdv_max = tuple(dbdv_list)

    dbdv_min = getr("/dbd-version/id", api=True, params={"dbdv_str": dbdv_min})
    dbdv_min = getr(f"/dbd-version/{dbdv_min}", api=True)

    if dbdv_max is not None:
        dbdv_max = getr("/dbd-version/id", api=True, params={"dbdv_str": dbdv_max})
        dbdv_max = getr(f"/dbd-version/{dbdv_max}", api=True)

    return DBDVersionRange(dbdv_min=dbdv_min, dbdv_max=dbdv_max)
