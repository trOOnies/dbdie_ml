"""Endpoint for cropping related purposes."""

from dbdie_classes.base import FullModelType
from fastapi import APIRouter, status
from fastapi.exceptions import HTTPException
from traceback import print_exc

from backbone.cropping.cropper_swarm import CropperSwarm

router = APIRouter()


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
        cps.run(
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
