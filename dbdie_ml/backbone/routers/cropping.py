"""Endpoint for cropping related purposes."""

from dbdie_classes.base import FullModelType
from dbdie_classes.paths import recursive_dirname
from dbdie_classes.schemas.objects import CropperSwarmOut
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
import os
import requests
from traceback import print_exc
import yaml

from backbone.cropping.cropper_swarm import CropperSwarm
from backbone.endpoints import bendp, parse_or_raise

CONFIGS_FD = os.path.join(recursive_dirname(__file__, 2), "configs")

router = APIRouter()


@router.post("/register", status_code=status.HTTP_201_CREATED)
def register_cropper_swarm(cropper_swarm: CropperSwarmOut):
    data = cropper_swarm.model_dump()
    path = os.path.join(CONFIGS_FD, f"cropper_swarms/{data['name']}/metadata.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


@router.post("/batch", status_code=status.HTTP_201_CREATED)
def batch_crop(
    cropper_swarm_name: str,
    move: bool = True,
    use_croppers: list[str] | None = None,
    use_fmts: list[FullModelType] | None = None,
):
    """[NEW] Run all Croppers iterating on images first.

    move: Whether to move the source images at the end of the cropping.
        Note: The MovableReport still avoid creating crops
        of duplicate source images.

    Filter options (cannot be used at the same time):
    - use_croppers: Filter cropping using Cropper names (level=Cropper).
    - use_fmt: Filter cropping using FullModelTypes names (level=crop type).
    """
    try:
        cps = CropperSwarm.from_register(cropper_swarm_name)
        parse_or_raise(requests.get(bendp(f"/cropper-swarm/{cps.id}")))

        matches = parse_or_raise(
            requests.get(bendp("/matches"), params={"limit": 300_000})
        )
        fs = cps.filter_fs_with_dbdv(matches)
        del matches

        cps.run(
            fs,
            move=move,
            use_croppers=use_croppers,
            use_fmts=use_fmts,
        )
    except AssertionError as e:
        print_exc()
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e)) from e
    except Exception as e:
        print_exc()
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(e)) from e
    return status.HTTP_201_CREATED
