"""Extra code for the InfoExtractor class."""

from copy import deepcopy
from os import listdir, mkdir
from os.path import exists, isdir, join
from shutil import rmtree
from typing import TYPE_CHECKING, Literal, Optional, Union

import pandas as pd

from dbdie_classes.groupings import PredictableTuples
from dbdie_classes.options import MODEL_TYPE as MT
from dbdie_classes.options.IMPLEMENTED import FMTS as IMPLEMENTED_FMTS
from dbdie_classes.utils import filter_multitype

from backbone.ml.models import IEModel

if TYPE_CHECKING:
    from numpy import ndarray

    from dbdie_classes.base import FullModelType, ModelType, Path, PathToFolder
    from dbdie_classes.schemas.helpers import DBDVersionRange

    from backbone.classes.training import TrainModel

# * Loading


def process_models_metadata(
    metadata: dict,
    models_fd: "PathToFolder",
) -> dict["FullModelType", dict[str, str | int]]:
    model_names = set(list(metadata["models"].keys()))
    assert len(model_names) == len(
        metadata["models"]
    ), "Duplicated model names in the metadata YAML file"

    assert model_names == set(
        fd for fd in listdir(models_fd) if isdir(join(models_fd, fd))
    ), "The model subfolders do not match the metadata YAML file"
    return metadata["models"]


# * Base


def check_implemented_models(models_cfgs: list["TrainModel"]) -> None:
    implemented_pts = PredictableTuples.from_fmts(IMPLEMENTED_FMTS)
    assert all(mcfg.fmt in implemented_pts.fmts for mcfg in models_cfgs), (
        f"fmts must be one of the following implemented models: {implemented_pts.fmts}"
    )


def get_models(
    extr_id: int,
    extr_name: str,
    models_cfgs: dict["FullModelType", "TrainModel"],
    trained_fmts: list["FullModelType"],
) -> dict["FullModelType", IEModel]:
    """Get IEModels from their train configs."""
    if trained_fmts:
        all_trained = set(list(models_cfgs.keys())) == set(trained_fmts)
        if all_trained:
            return {
                fmt: IEModel.from_folder(extr_name, fmt)
                for fmt in trained_fmts
            }

    from backbone.ml.models.custom import (
        AddonsModel,
        CharacterModel,
        ItemModel,
        OfferingModel,
        PerkModel,
        StatusModel,
    )

    TYPES_TO_MODELS: dict["ModelType", IEModel] = {
        MT.ADDONS: AddonsModel,
        MT.CHARACTER: CharacterModel,
        MT.ITEM: ItemModel,
        MT.OFFERING: OfferingModel,
        MT.PERKS: PerkModel,
        MT.STATUS: StatusModel,
    }

    check_implemented_models(models_cfgs.values())

    pred_tuples = PredictableTuples.from_fmts(list(models_cfgs.keys()))

    return {
        mcfg.fmt: (
            IEModel.from_folder(extr_name, mcfg.fmt)
            if mcfg.fmt in trained_fmts
            else TYPES_TO_MODELS[ptup.mt](
                id=mcfg.id,
                ifk=ptup.ifk,
                total_classes=mcfg.total_classes,
                cps_name=mcfg.cps_name,
                extr_id=extr_id,
            )
        )
        for mcfg, ptup in zip(models_cfgs.values(), pred_tuples)
    }


def get_dbdvr(
    models: dict["FullModelType", IEModel],
    mode: Literal["match_all", "intersection"] = "match_all",
    expected: Optional["DBDVersionRange"] = None,
) -> tuple["DBDVersionRange", list[int]]:
    """Calculate DBDVersionRange from many IEModels."""
    assert all(model.fmt == mt for mt, model in models.items())

    vrs = [model.dbdvr for model in models.values()]
    vrs_ids = [model.dbdvr_ids for model in models.values()]
    if mode == "match_all":
        dbdvr = vrs[0]
        assert all(vr == dbdvr for vr in vrs), "All model versions must match."
        dbdvr_ids = vrs_ids[0]
    elif mode == "intersection":
        raise NotImplementedError("Not implemented yet because of missing DBDV ids implementation.")
        # if len(vrs) == 1:
        #     dbdvr = vrs[0]
        # else:
        #     dbdvr = vrs[0] & vrs[1]
        #     if len(vrs) > 2:
        #         for vr in vrs[2:]:
        #             dbdvr = dbdvr & vr
    else:
        raise ValueError(f"Mode '{mode}' not recognized")

    if expected is not None:
        assert (
            dbdvr == expected
        ), f"Seen version ('{dbdvr}') is different from expected version ('{expected}')."

    return dbdvr, dbdvr_ids


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


def save_metadata(obj, dst: "Path") -> None:
    metadata = obj.to_metadata()
    metadata.save(dst)


def save_models(models, extractor_fd: "PathToFolder") -> None:
    models_fd = join(extractor_fd, "models")
    for mn, model in models.items():
        model.save(join(models_fd, mn))


def folder_save_logic(
    models: dict["FullModelType", IEModel],
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
