"""InfoExtractor code (which manages many IEModels)"""

from __future__ import annotations

import pandas as pd
import os
from typing import TYPE_CHECKING, Optional, Union

from copy import deepcopy
import yaml

from dbdie_classes.base import Path
from dbdie_classes.code.version import filter_images_with_dbdv
from dbdie_classes.extract import PlayerInfo
from dbdie_classes.options.MODEL_TYPE import TO_ID_NAMES
from dbdie_classes.schemas.groupings import FullMatchOut
from dbdie_classes.schemas.objects import ExtractorModelsIds, ExtractorOut

from backbone.classes.metadata import (
    SavedDBDVersion,
    SavedExtractorMetadata,
)
from backbone.classes.register import get_extr_mpath
from backbone.classes.training import TrainModel
from backbone.code.extractor import (
    check_datasets,
    folder_save_logic,
    get_dbdvr,
    get_models,
    get_printable_info,
    match_preds_types,
    process_models_metadata,
    save_metadata,
    save_models,
)
# from backbone.db import to_player
from backbone.ml.models import IEModel
from backbone.options.COLORS import get_class_cprint
from dbdie_classes.schemas.helpers import DBDVersionRange, DBDVersionOut

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import DataFrame

    from dbdie_classes.base import FullModelType
    from dbdie_classes.extract import CropCoords, PlayersCropCoords, PlayersInfoDict
    from dbdie_classes.schemas.groupings import PlayerOut
    from dbdie_classes.schemas.objects import ModelOut

    from backbone.classes.training import TrainExtractor, TrainModel

ie_print = get_class_cprint("InfoExtractor")


class InfoExtractor:
    """Extracts information of an image using multiple IEModels.

    Inputs:
        name (str | None): Name of the InfoExtractor.

    Attrs:
        dbdvr (DBDVersionRange | None): Inferred from its models.
        fmts (list[FullModelType] | None)
        models_are_init (bool)
        models_are_trained (bool)

    Usage:
    >>> ie = InfoExtractor(name="my_info_extractor")
    >>> # ie.models = {"perks_surv": my_model, ...}  # ! only for super-users
    >>> ie.init_extractor()  # this uses all available models
    >>> ie.train(...)
    >>> ie.save("/path/to/extractor/folder")
    >>> preds_dict = ie.predict_batch({"perks": "/path/to/dataset.csv", ...})

    Load previously trained InfoExtractor:
    >>> ie = InfoExtractor.from_folder("/path/to/extractor/folder")
    >>> new_preds_dict = ie.predict_batch({"perks": "/path/to/other/dataset.csv", ...})
    """

    def __init__(
        self,
        id: int,
        name: str,
    ) -> None:
        self.id = id
        self.name = name
        self.flushed = False

    def __repr__(self) -> str:
        """InfoExtractor('my_info_extractor', version='7.5.0')"""
        vals = f"id={self.id}, name='{self.name}', version='{self.dbdvr}"
        return f"InfoExtractor({vals})"

    @property
    def fmts(self) -> list["FullModelType"]:
        assert self.models_are_init
        return list(self._models.keys())

    @property
    def models_are_init(self) -> bool:
        return hasattr(self, "_models")

    @property
    def models_are_trained(self) -> bool:
        if not self.models_are_init:
            return False
        else:
            return all(m.model_is_trained for m in self._models.values())

    @property
    def models_ids(self) -> dict["FullModelType", int]:
        return {fmt: m.id for fmt, m in self._models.items()}

    # @staticmethod
    # def to_players(players_info: "PlayersInfoDict") -> list["PlayerOut"]:
    #     return [to_player(i, sn_info) for i, sn_info in players_info.items()]

    # * Base

    def init_extractor(
        self,
        cps_name: str,
        models_cfgs: dict["FullModelType", TrainModel],
        trained_fmts: list["FullModelType"],
        expected_dbdvr: Optional[DBDVersionRange] = None,
    ) -> None:
        """Initialize the InfoExtractor and its IEModels.
        Can be provided with already trained models.
        """
        ie_print("Initializing...")
        self._check_flushed()
        assert (
            not self.models_are_init
        ), "InfoExtractor can't be reinitialized before being flushed first."
        assert models_cfgs, "'models_cfgs' can't be empty."

        self.cps_name = cps_name
        self._models = get_models(self.name, models_cfgs, trained_fmts)
        self.dbdvr, self.dbdvr_ids = get_dbdvr(
            self._models,
            expected=expected_dbdvr,
        )
        for model in self._models.values():
            if not model.model_is_trained:
                model.init_model()
        ie_print("Initialized.")

    def print_models(self) -> None:
        print("EXTRACTOR:", str(self))
        print("MODELS:", end="")

        if self.models_are_init:
            print()
            printable_info = get_printable_info(self._models)
            print(printable_info)
        else:
            print(" NONE")

    # * Loading and saving

    @classmethod
    def from_train_config(cls, cfg: "TrainExtractor") -> InfoExtractor:
        """Load an untrained `InfoExtractor` from a training config."""
        if cfg.custom_dbdvr is not None:
            raise NotImplementedError
        ie = cls(cfg.id, cfg.name)
        ie.init_extractor(cfg.cps_name, cfg.fmts, trained_fmts=[])
        return ie

    @classmethod
    def from_folder(cls, extr_name: str) -> InfoExtractor:
        """Loads a trained `InfoExtractor` using each model's metadata and the actual models."""
        mpath = get_extr_mpath(extr_name)
        extr_fd = os.path.dirname(mpath)

        with open(mpath, "r") as f:
            metadata = yaml.safe_load(f)

        models_fd = os.path.join(extr_fd, "models")
        models_md = process_models_metadata(metadata, models_fd)
        del metadata["models"]

        expected_dbdvr = DBDVersionRange.from_dicts(metadata["dbdv_min"], metadata["dbdv_max"])
        del metadata["dbdv_min"], metadata["dbdv_max"]

        ie = cls(id=metadata["id"], name=metadata["name"])
        trained_models = {
            fmt: IEModel.from_folder(extr_name, fmt=fmt)
            for fmt in models_md
        }
        ie.init_extractor(
            metadata["cps_name"],
            {
                fmt: TrainModel.from_model(iem)
                for fmt, iem in trained_models.items()
            },
            trained_fmts=list(trained_models.keys()),
            expected_dbdvr=expected_dbdvr,
        )
        return ie

    def to_metadata(self) -> SavedExtractorMetadata:
        """Return the extractor's `SavedExtractorMetadata`."""
        return SavedExtractorMetadata(
            cps_name=self.cps_name,
            dbdv_max=(
                SavedDBDVersion.from_dbdv(self.dbdvr.dbdv_max)
                if self.dbdvr.bounded else None
            ),
            dbdv_min=SavedDBDVersion.from_dbdv(self.dbdvr.dbdv_min),
            id=self.id,
            models={
                model.fmt: {
                    "id": model.id,
                    "name": model.name,
                }
                for model in self._models.values()
            },
            name=self.name,
        )

    def save(self, replace: bool = True) -> None:
        """Save all necessary objects of the InfoExtractor and all its IEModels."""
        self._check_flushed()
        assert self.models_are_trained, "Non-trained InfoExtractor cannot be saved."

        mpath = get_extr_mpath(self.name)
        extr_fd = os.path.dirname(mpath)

        folder_save_logic(self._models, extr_fd, replace)
        save_metadata(self, mpath)
        save_models(self._models, extr_fd)
        ie_print("All models have been saved.")

    # * Training

    def filter_matches_with_dbdv(self, matches: list[dict]) -> list[dict[str, int | str]]:
        """Filter matches list with DBDVersion ids."""
        return [
            {"id": m["id"], "filename": m["filename"]}
            for m in filter_images_with_dbdv(
                matches,
                self.dbdvr_ids[0],
                self.dbdvr_ids[1],
            )
        ]

    def train(
        self,
        label_ref_paths: dict["FullModelType", Path],
        train_datasets: dict["FullModelType", Path],
        val_datasets: dict["FullModelType", Path],
    ) -> None:
        """Train all models one after the other."""
        ie_print("Training extractor...")
        self._check_flushed()
        assert not self.models_are_trained

        check_datasets(self.fmts, label_ref_paths)
        check_datasets(self.fmts, train_datasets)
        check_datasets(self.fmts, val_datasets)

        print(50 * "-")
        for mt, model in self._models.items():
            model.train(
                label_ref_path=label_ref_paths[mt],
                train_dataset_path=train_datasets[mt],
                val_dataset_path=val_datasets[mt],
            )
            print(50 * "-")
        ie_print("All models have been trained.")

    def flush(self) -> None:
        """Flush InfoExtractor and its IEModels so as to free space.
        A flushed InfoExtractor shouldn't be reused, but deleted and reinstantiated.
        """
        self._check_flushed()
        self.flushed = True

        if self.models_are_init:
            model_names = [mn for mn in self._models]
            for mn in model_names:
                self._models[mn].flush()
                del self._models[mn]
            del self._models

    def _check_flushed(self) -> None:
        """Check whether the model has been flushed yet."""
        assert not self.flushed, "InfoExtractor was flushed"

    # * Prediction

    def predict_on_crop(self, crop: "CropCoords") -> PlayerInfo:
        self._check_flushed()
        preds = {
            TO_ID_NAMES[k]: model.predict(crop)
            for k, model in self._models.items()
        }
        return PlayerInfo(**preds)

    def predict(self, player_crops: "PlayersCropCoords") -> "PlayersInfoDict":
        self._check_flushed()
        return {i: self.predict_on_crop(s) for i, s in player_crops.items()}

    def predict_batch(
        self,
        datasets: dict["FullModelType", Path] | dict["FullModelType", "DataFrame"],
        fmts: list["FullModelType"] | None = None,
        probas: bool = False,
    ) -> dict["FullModelType", dict[str, "ndarray"]]:
        self._check_flushed()
        assert self.models_are_trained

        if fmts is None:
            fmts_ = self.fmts
        else:
            assert fmts, "'fmts' can't be an empty list."
            assert all(fmt in self._models for fmt in fmts)
            fmts_ = deepcopy(fmts)

        check_datasets(fmts_, datasets)

        return {
            fmt: {
                "match_ids": pd.read_csv(datasets[fmt], usecols=["match_id"])["match_id"].values,
                "player_ids": pd.read_csv(datasets[fmt], usecols=["player_id"])["player_id"].values,
                "preds": self._models[fmt].predict_batch(datasets[fmt], probas=probas),
            }
            for fmt in fmts_
        }

    def convert_names(
        self,
        preds: Union["ndarray", dict["FullModelType", "ndarray"]],
        on: Optional[Union["FullModelType", list["FullModelType"]]] = None,
    ) -> Union[list[str], dict["FullModelType", list[str]]]:
        """Convert model int predictions to named predictions.

        preds (Union[...]): Integer predictions to convert
        - ndarray: A certain model type preds
        - dict[FullModelType, ndarray]: Preds from different model types

        on (Union[...]): Work 'on' certain model types
        - FullModelType: Select a model type (only mode allowed when preds is a ndarray)
        - list[FullModelType]: Select many model types
        - None: Select all provided model types in preds
        """
        self._check_flushed()
        assert self.models_are_trained

        preds_, on_ = match_preds_types(preds, on)  # TODO: Is it working OK?
        names = {k: self.models[k].convert_names(preds[k]) for k in on}
        return names[list(names.keys())[0]] if isinstance(on, str) else names

    # * Schemas

    def to_schema(self, extra_info: dict) -> ExtractorOut:
        """Convert to corresponding Pydantic schema."""
        return ExtractorOut(
            **(
                self.to_metadata().typed_dict()
                | {
                    "dbdv_min_id": self.dbdvr_ids[0],
                    "dbdv_max_id": self.dbdvr_ids[1],
                    "models_ids": ExtractorModelsIds.from_fmt_dict(self.models_ids),
                }
                | extra_info
            )
        )

    def models_to_schemas(
        self,
        extra_info: dict["FullModelType", dict],
    ) -> dict["FullModelType", "ModelOut"]:
        """Convert models to dict of corresponding Pydantic schemas."""
        return {
            fmt: m.to_schema(extra_info[fmt])
            for fmt, m in self._models.items()
        }

    def form_match(
        self,
        version: DBDVersionOut,
        players: list["PlayerOut"]
    ) -> FullMatchOut:
        self._check_flushed()
        return FullMatchOut(version=version, players=players)
