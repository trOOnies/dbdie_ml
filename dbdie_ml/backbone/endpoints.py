"""Endpoints-related helper functions."""

import re
from typing import TYPE_CHECKING

from fastapi import status
from fastapi.exceptions import HTTPException

from backbone.config import ST

if TYPE_CHECKING:
    from dbdie_classes.base import Endpoint, FullEndpoint


ENDPOINT_PATT = re.compile(r"\/[a-z\-]+$")
NOT_WS_PATT = re.compile(r"\S")


def endp(endpoint: "Endpoint") -> "FullEndpoint":
    """Get full URL of the endpoint."""
    return ST.fastapi_host + endpoint


def bendp(endpoint: "Endpoint") -> "FullEndpoint":
    """Get full URL of the base endpoint."""
    return ST.base_api_host + endpoint


def parse_or_raise(resp, exp_status_code: int = status.HTTP_200_OK):
    """Parse Response as JSON or raise error as exception, depending on status code."""
    if resp.status_code != exp_status_code:
        raise HTTPException(
            status_code=resp.status_code,
            detail=resp.reason,
        )
    return resp.json()
