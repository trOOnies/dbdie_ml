"""Extra code for the InfoExtractor class."""

from copy import deepcopy
from os import listdir, mkdir
from os.path import exists, isdir, join
from shutil import rmtree
from typing import TYPE_CHECKING, Literal, Optional, Union

import pandas as pd

from dbdie_classes.options import MODEL_TYPE as MT
from dbdie_classes.options import PLAYER_TYPE as PT
from dbdie_classes.options.FMT import from_fmt, to_fmt
from dbdie_classes.utils import filter_multitype

from backbone.classes.metadata import SavedExtractorMetadata

if TYPE_CHECKING:
    from numpy import ndarray

    from dbdie_classes.base import FullModelType, Path, PathToFolder
    from dbdie_classes.version import DBDVersionRange
    from backbone.ml.models import IEModel

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
    models_ids: dict["FullModelType", int],
    trained_models: dict["FullModelType", "IEModel"] | None,
    fmts_with_counts: dict["FullModelType", int],
) -> dict["FullModelType", "IEModel"]:
    if trained_models is not None:
        return trained_models
    else:
        from backbone.ml.models.custom import (
            CharacterModel, ItemModel, PerkModel, StatusModel
        )

        TYPES_TO_MODELS = {
            MT.CHARACTER: CharacterModel,
            MT.ITEM: ItemModel,
            MT.PERKS: PerkModel,
            MT.STATUS: StatusModel,
        }
        TYPES_TO_IMG_SIZES = {
            to_fmt(MT.CHARACTER, True):  (480, 33),
            to_fmt(MT.CHARACTER, False): (480, 33),
            to_fmt(MT.ITEM, True):        (43, 42),
            to_fmt(MT.ITEM, False):       (43, 43),
            to_fmt(MT.PERKS, True):       (55, 55),
            to_fmt(MT.PERKS, False):      (56, 55),
            to_fmt(MT.STATUS, False):     (29, 35),
        }

        # TODO: This are the currently implemented models
        base_models = {
            to_fmt(mt, ifk): ifk
            for mt in [MT.CHARACTER, MT.ITEM, MT.PERKS]
            for ifk in [True, False]
        } | {f"{MT.STATUS}__{PT.SURV}": False}

        try:
            models = {
                fmt: (base_models[fmt], total)
                for fmt, total in fmts_with_counts.items()
            }
        except KeyError:
            raise KeyError(
                "fmts must be one of the following implemented models: "
                + str(list(base_models.keys()))
            )
        del base_models

        return {
            fmt: TYPES_TO_MODELS[from_fmt(fmt)[0]](
                id=models_ids[fmt],
                ifk=ifk,
                total_classes=total,
                img_size=TYPES_TO_IMG_SIZES[fmt],  # TODO: MAKE IT DEPEND ON CPS
                version_range=["7.4.0-ptb", "8.1.0-ptb"],  # TODO: PARAMETRIZE
                cps_name="banner-badge",  # TODO: PARAMETRIZE
            )
            for fmt, (ifk, total) in models.items()
        }


def get_version_range(
    models: dict["FullModelType", "IEModel"],
    mode: Literal["match_all", "intersection"] = "match_all",
    expected: Optional["DBDVersionRange"] = None,
) -> "DBDVersionRange":
    """Calculate DBDVersionRange from many IEModels."""
    assert all(model.fmt == mt for mt, model in models.items())

    vrs = [model.version_range for model in models.values()]
    if mode == "match_all":
        version_range = vrs[0]
        assert all(vr == version_range for vr in vrs), "All model versions must match."
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
        ), f"Seen version ('{version_range}') is different from expected version ('{expected}')."

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


# * Training


def check_datasets(
    fmts: list["FullModelType"],
    datasets: dict["FullModelType", "Path"] | dict["FullModelType", pd.DataFrame],
) -> None:
    """Check labeled dataset integrity."""
    fmts_set = set(fmts)
    dataset_set = set(datasets.keys())
    assert fmts_set == dataset_set, (
        "Dataset keys and set keys don't match:\n"
        + f"- FMTS: {fmts_set}\n"
        + f"- DATA: {dataset_set}"
    )
    if isinstance(list(datasets.values())[0], str):
        assert all(exists(p) for p in datasets.values())


# * Saving


def save_metadata(obj, extractor_fd: "PathToFolder") -> None:
    dst = join(extractor_fd, "metadata.yaml")
    metadata = SavedExtractorMetadata(
        cropper_swarm_id=obj.cropper_swarm_id,
        id=obj.id,
        models={
            model.fmt: model.id
            for model in obj._models.values()
        },
        name=obj.name,
        version_range=[obj.version_range.id, obj.version_range.max_id],
    )
    metadata.save(dst)


def save_models(models, extractor_fd: "PathToFolder") -> None:
    models_fd = join(extractor_fd, "models")
    for mn, model in models.items():
        model.save(join(models_fd, mn))


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
    on: Union["FullModelType", list["FullModelType"], None],
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
