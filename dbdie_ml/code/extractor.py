from copy import deepcopy
from os import listdir, mkdir
from os.path import isdir, join
from shutil import rmtree
from typing import TYPE_CHECKING, Literal

import pandas as pd

from dbdie_ml.utils import filter_multitype
from dbdie_ml.options import MODEL_TYPES as MT
from dbdie_ml.options import PLAYER_TYPE as PT

if TYPE_CHECKING:
    from numpy import ndarray

    from dbdie_ml.classes.base import FullModelType, PathToFolder
    from dbdie_ml.classes.version import DBDVersionRange
    from dbdie_ml.ml.models import IEModel

# * Loading


def process_model_names(
    metadata: dict,
    models_fd: "PathToFolder",
) -> list:
    model_names = set(metadata["models"])
    assert len(model_names) == len(
        metadata["models"]
    ), "Duplicated model names in the metadata YAML file"

    assert model_names == set(
        fd for fd in listdir(models_fd) if isdir(join(models_fd, fd))
    ), "The model subfolders do not match the metadata YAML file"
    return list(model_names)


# * Base


def get_models(
    name: str,
    trained_models: dict["FullModelType", "IEModel"] | None,
    fmts: list["FullModelType"] | None,
) -> dict["FullModelType", "IEModel"]:
    if trained_models is not None:
        return trained_models
    else:
        from dbdie_ml.ml.models.custom import (
            CharacterModel, ItemModel, PerkModel, StatusModel
        )

        TYPES_TO_MODELS = {
            MT.CHARACTER: CharacterModel,
            MT.ITEM: ItemModel,
            MT.PERKS: PerkModel,
            MT.STATUS: StatusModel,
        }

        # TODO: This are the currently implemented models
        base_models = {
            f"{mt}__{PT.ifk_to_pt(ifk)}": ifk
            for mt in [MT.CHARACTER, MT.ITEM, MT.PERKS]
            for ifk in [True, False]
        } | {f"{MT.STATUS}__{PT.SURV}": False}

        if fmts is not None:
            try:
                models = {k: base_models[k] for k in fmts}
            except KeyError:
                raise KeyError(
                    "fmts must be one of the following implemented models: "
                    + str(list(base_models.keys()))
                )

        return {
            mt: TYPES_TO_MODELS[mt[: mt.index("")]](
                name=f"{name}__m{i}" if name is not None else None,
                is_for_killer=ifk,
            )
            for i, (mt, ifk) in enumerate(models.items())
        }


def get_version_range(
    models: dict["FullModelType", "IEModel"],
    mode: Literal["match_all", "intersection"] = "match_all",
    expected: "DBDVersionRange" | None = None,
) -> "DBDVersionRange":
    """Calculate DBDVersionRange from many IEModels."""
    assert all(model.selected_fd == mt for mt, model in models.items())

    vrs = [model.version_range for model in models.values()]
    if mode == "match_all":
        version_range = vrs[0]
        assert all(vr == version_range for vr in vrs), "All model versions must match"
    elif mode == "intersection":
        if len(vrs) == 1:
            version_range = vrs[0]
        else:
            version_range = vrs[0] & vrs[1]
            if len(vrs) > 2:
                for vr in vrs[2:]:
                    version_range = version_range & vr
    else:
        raise ValueError(f"Mode '{mode}' not recognized")

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


# * Saving


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
    preds: "ndarray" | dict["FullModelType", "ndarray"],
    on: "FullModelType" | list["FullModelType"] | None,
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
