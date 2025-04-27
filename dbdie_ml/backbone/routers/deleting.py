"""Router for objects deletion related processes."""

import os

from dbdie_classes.paths import recursive_dirname
from fastapi import APIRouter, Response, status

BASE_SUPFD = recursive_dirname(__file__, 4)
MODELS_FD = f"{BASE_SUPFD}/models"
EXTRACTORS_FD = f"{BASE_SUPFD}/extractors"

router = APIRouter()


@router.delete("/model/{id}", status_code=status.HTTP_200_OK)
def delete_model(id: int):
    """Delete the files of a model."""
    model_fd = f"{MODELS_FD}/{id}"

    os.remove(f"{model_fd}/label_ref.json")
    os.remove(f"{model_fd}/metadata.yaml")
    os.remove(f"{model_fd}/model.pt")
    os.rmdir(model_fd)

    return Response(status_code=status.HTTP_200_OK)


@router.delete("/extractor/{id}", status_code=status.HTTP_200_OK)
def delete_extractor(id: int):
    """Delete the files of a extractor."""
    extractors_fd = f"{EXTRACTORS_FD}/{id}"

    os.remove(f"{extractors_fd}/metadata.yaml")
    os.rmdir(extractors_fd)

    return Response(status_code=status.HTTP_200_OK)
