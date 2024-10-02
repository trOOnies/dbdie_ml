"""InfoExtractor code (which manages many IEModels)"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union
from uuid import uuid4

from copy import deepcopy
import yaml

from dbdie_classes.base import Path, PathToFolder
from dbdie_classes.extract import PlayerInfo
from dbdie_classes.version import DBDVersion
from backbone.code.extractor import (
    check_datasets,
    folder_save_logic,
    get_models,
    get_printable_info,
    get_version_range,
    match_preds_types,
    process_model_names,
    save_metadata,
    save_models,
)
# from backbone.db import to_player
from backbone.ml.models import IEModel
from dbdie_classes.options.MODEL_TYPE import TO_ID_NAMES
from dbdie_classes.schemas.groupings import FullMatchOut

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import DataFrame

    from dbdie_classes.base import FullModelType
    from dbdie_classes.extract import CropCoords, PlayersCropCoords, PlayersInfoDict
    from dbdie_classes.version import DBDVersionRange
    from dbdie_classes.schemas.groupings import PlayerOut


class InfoExtractor:
    """Extracts information of an image using multiple IEModels.

    Inputs:
        name (str | None): Name of the InfoExtractor.

    Attrs:
        version_range (DBDVersionRange | None): Inferred from its models.
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

    def __init__(self, id: int, name: str = "") -> None:
        self.id = id
        self.name = (
            str(uuid4()) if name == "" else name
        )  # TODO: fill with a random friendlier name if empty

        self.flushed = False

    def __repr__(self) -> str:
        """InfoExtractor('my_info_extractor', version='7.5.0')"""
        vals = f"id={self.id}, name='{self.name}', version='{self.version_range}"
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

    # @staticmethod
    # def to_players(players_info: "PlayersInfoDict") -> list["PlayerOut"]:
    #     return [to_player(i, sn_info) for i, sn_info in players_info.items()]

    # * Base

    def init_extractor(
        self,
        models_ids: dict["FullModelType", int],
        fmts_with_counts: dict["FullModelType", int],
        trained_models: Optional[dict["FullModelType", IEModel]] = None,
        expected_version_range: Optional["DBDVersionRange"] = None,
    ) -> None:
        """Initialize the InfoExtractor and its IEModels.
        Can be provided with already trained models.
        """
        self._check_flushed()
        assert (
            not self.models_are_init
        ), "InfoExtractor can't be reinitialized before being flushed first"
        assert fmts_with_counts, "'fmts_with_counts' can't be empty"

        self._models = get_models(models_ids, trained_models, fmts_with_counts)
        self.version_range = get_version_range(
            self._models,
            expected=expected_version_range,
        )
        if trained_models is None:
            for model in self._models.values():
                model.init_model()

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
    def from_folder(cls, extractor_fd: "PathToFolder") -> InfoExtractor:
        """Loads a DBDIE extractor using each model's metadata and the actual models"""
        with open(os.path.join(extractor_fd, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)

        models_fd = os.path.join(extractor_fd, "models")
        model_names = process_model_names(metadata, models_fd)
        del metadata["models"]

        exp_version_range = metadata["version-range"]
        del metadata["version-range"]

        models_ids: dict["FullModelType", int] = metadata["models"]
        del metadata["models"]

        ie = InfoExtractor(**metadata)
        ie.init_extractor(
            models_ids,
            trained_models={
                mn: IEModel.from_folder(os.path.join(models_fd, mn))
                for mn in model_names
            },
            expected_version_range=exp_version_range,
        )
        return ie

    def save(self, extractor_fd: "PathToFolder", replace: bool = True) -> None:
        """Save all necessary objects of the InfoExtractor and all its IEModels."""
        self._check_flushed()
        assert self.models_are_trained, "Non-trained InfoExtractor cannot be saved."

        folder_save_logic(self._models, extractor_fd, replace)
        save_metadata(self, extractor_fd)
        save_models(self._models, extractor_fd)

    # * Training

    def train(
        self,
        label_ref_paths: dict["FullModelType", Path],
        train_datasets: dict["FullModelType", Path],
        val_datasets: dict["FullModelType", Path],
    ) -> None:
        """Train all models one after the other."""
        self._check_flushed()
        assert not self.models_are_trained

        check_datasets(self.fmts, label_ref_paths)
        check_datasets(self.fmts, train_datasets)
        check_datasets(self.fmts, val_datasets)

        print(50 * "-")
        for mt, model in self._models.items():
            print(f"Training {mt} model...")
            model.train(
                label_ref_path=label_ref_paths[mt],
                train_dataset_path=train_datasets[mt],
                val_dataset_path=val_datasets[mt],
            )
            print(50 * "-")
        print("All models have been trained.")

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
    ) -> dict["FullModelType", "ndarray"]:
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
            fmt: self._models[fmt].predict_batch(datasets[fmt], probas=probas)
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

    # * Match

    def form_match(
        self, version: DBDVersion, players: list["PlayerOut"]
    ) -> FullMatchOut:
        self._check_flushed()
        return FullMatchOut(version=version, players=players)
