from typing import Any
from copy import deepcopy


def pls(item: str, length: int) -> str:
    """Plural letter 's' friendly count"""
    return f"{length} {item}{'s' if length != 1 else ''}"


def filter_multitype(
    items: Any | list[Any] | None,
    default: list[Any],
    possible_values: list[Any] | set[Any] | None = None,
) -> list[Any]:
    """Centralized function to convert some type options to a list"""
    if items is None:
        return deepcopy(default)
    elif isinstance(items, str):
        if possible_values is not None:
            assert items in possible_values
        return [deepcopy(items)]
    elif isinstance(items, list):
        assert items
        if possible_values is not None:
            assert all(it in possible_values for it in items)
        return deepcopy(items)
    else:
        raise TypeError("Type of items not recognized")
