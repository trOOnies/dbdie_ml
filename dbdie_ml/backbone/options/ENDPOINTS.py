"""DBDIE API high-level endpoints."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from dbdie_classes.base import Endpoint

CROP    : "Endpoint" = "/crop"
EXTRACT : "Endpoint" = "/extract"
TRAIN   : "Endpoint" = "/train"
