from copy import deepcopy
from os import mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import TYPE_CHECKING, Optional, Union

import pandas as pd

from dbdie_ml.utils import filter_multitype

if TYPE_CHECKING:
    from numpy import ndarray

    from dbdie_ml.classes import DBDVersionRange, FullModelType, PathToFolder
    from dbdie_ml.models import IEModel

# * Base


def get_version_range(
    models: dict["FullModelType", "IEModel"],
    expected: Optional["DBDVersionRange"] = None,
) -> "DBDVersionRange":
    assert all(model.selected_fd == mt for mt, model in models.items())

    vrs = [model.version_range for model in models.values()]
    version_range = vrs[0]

    assert all(vr == version_range for vr in vrs), "All model versions must match"

    if expected is not None:
        assert (
            version_range == expected
        ), f"Seen version ('{version_range}') is different from expected version ('{expected}')"

    return version_range


def get_printable_info(models: dict) -> pd.DataFrame:
    printable_info = {
        mn: str(model)[8:-1]  # "IEModel(" and ")"
        for mn, model in models.items()
    }
    printable_info = {mn: s.split(", ") for mn, s in printable_info.items()}
    printable_info = {
        mn: {
            v[: v.index("=")]: v[v.index("=") + 2 : -1] for v in vs if v.find("=") > -1
        }
        for mn, vs in printable_info.items()
    }
    printable_info = {
        mn: (
            {"name": (models[mn].name if models[mn].name is not None else "UNNAMED")}
            | d
        )
        for mn, d in printable_info.items()
    }
    printable_info = pd.DataFrame.from_dict(printable_info)
    printable_info = printable_info.T
    printable_info = printable_info.set_index("name", drop=True)
    return printable_info


# * Loading and saving


def folder_save_logic(
    models: dict["FullModelType", "IEModel"],
    extractor_fd: "PathToFolder",
    replace: bool,
) -> None:
    """Logic for the creation of the saving folder and subfolders."""
    if replace:
        if isdir(extractor_fd):
            rmtree(extractor_fd)
        mkdir(extractor_fd)
        mkdir(join(extractor_fd, "models"))
    else:
        models_fd = join(extractor_fd, "models")

        if not isdir(extractor_fd):
            mkdir(extractor_fd)
            mkdir(models_fd)
            for mn in models:
                path = join(models_fd, mn)
                mkdir(path)
        elif not isdir(models_fd):
            mkdir(models_fd)
            for mn in models:
                path = join(models_fd, mn)
                mkdir(path)
        else:
            for mn in models:
                path = join(models_fd, mn)
                if not isdir(path):
                    mkdir(path)


# * Prediction


def match_preds_types(
    preds: Union["ndarray", dict["FullModelType", "ndarray"]],
    on: Optional[Union["FullModelType", list["FullModelType"]]],
) -> tuple[dict["FullModelType", "ndarray"], list["FullModelType"]]:
    if isinstance(preds, dict):
        on_ = filter_multitype(
            on,
            default=list(preds.keys()),
        )
        preds_ = {deepcopy(k): p for k, p in preds.items() if k in on_}
    else:
        msg = "'on' must be a FullModelType if 'preds' is not a dict."
        assert isinstance(on, str), msg
        preds_ = {deepcopy(on): preds}
        on_ = [deepcopy(on)]

    return preds_, on_
