"""InfoExtractor code (which manages many IEModels)"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

from copy import deepcopy
import yaml

from dbdie_classes.base import Path, PathToFolder
from dbdie_classes.code.version import filter_images_with_dbdv
from dbdie_classes.extract import PlayerInfo
from dbdie_classes.options.MODEL_TYPE import TO_ID_NAMES
from dbdie_classes.schemas.groupings import FullMatchOut
from dbdie_classes.schemas.objects import ExtractorModelsIds, ExtractorOut

from backbone.classes.metadata import SavedExtractorMetadata, SavedModelMetadata
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
from backbone.options.COLORS import get_class_cprint

if TYPE_CHECKING:
    from numpy import ndarray
    from pandas import DataFrame

    from dbdie_classes.base import FullModelType
    from dbdie_classes.extract import CropCoords, PlayersCropCoords, PlayersInfoDict
    from dbdie_classes.schemas.groupings import PlayerOut
    from dbdie_classes.schemas.helpers import DBDVersionRange, DBDVersionOut
    from dbdie_classes.schemas.objects import ModelOut

    from backbone.classes.training import TrainExtractor, TrainModel

ie_print = get_class_cprint("InfoExtractor")


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
        models_cfgs: list[TrainModel],
        expected_version_range: Optional["DBDVersionRange"] = None,
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
        self._models = get_models(models_cfgs)
        self.version_range, self.version_range_ids = get_version_range(
            self._models,
            expected=expected_version_range,
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
        ie = InfoExtractor(cfg.id, cfg.name)
        ie.init_extractor(cfg.cps_name, [model for model in cfg.fmts.values()])
        return ie

    @classmethod
    def from_folder(cls, ie_name: str) -> InfoExtractor:
        """Loads a trained `InfoExtractor` using each model's metadata and the actual models."""
        extractor_fd: "PathToFolder" = f"extractors/{ie_name}"
        with open(os.path.join(extractor_fd, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)

        models_fd = os.path.join(extractor_fd, "models")
        model_names = process_model_names(metadata, models_fd)
        del metadata["models"]

        exp_version_range = metadata["version_range"]
        del metadata["version_range"]

        # TODO: Untangle this implementation
        ie = InfoExtractor(**metadata)
        trained_models = {
            mn: IEModel.from_folder(os.path.join(models_fd, mn))
            for mn in model_names
        }
        ie.init_extractor(
            metadata["cps_name"],
            [SavedModelMetadata.from_model_class(iem) for iem in trained_models.values()],
            expected_version_range=exp_version_range,
        )
        return ie

    def to_metadata(self) -> SavedExtractorMetadata:
        """Return the extractor's `SavedExtractorMetadata`."""
        return SavedExtractorMetadata(
            cps_name=self.cps_name,
            id=self.id,
            models={
                model.fmt: model.id
                for model in self._models.values()
            },
            name=self.name,
            version_range=self.version_range.to_list(),
            version_range_ids=self.version_range_ids,
        )

    def save(self, replace: bool = True) -> None:
        """Save all necessary objects of the InfoExtractor and all its IEModels."""
        self._check_flushed()
        assert self.models_are_trained, "Non-trained InfoExtractor cannot be saved."

        extractor_fd: "PathToFolder" = f"extractors/{self.name}"
        folder_save_logic(self._models, extractor_fd, replace)
        save_metadata(self, extractor_fd)
        save_models(self._models, extractor_fd)
        ie_print("All models have been saved.")

    # * Training

    def filter_matches_with_dbdv(self, matches: list[dict]) -> list[dict[str, int | str]]:
        """Filter matches list with DBDVersion ids."""
        return [
            {"id": m["id"], "filename": m["filename"]}
            for m in filter_images_with_dbdv(
                matches,
                self.version_range_ids[0],
                self.version_range_ids[1],
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

    # * Schemas

    def to_schema(self, extra_info: dict) -> ExtractorOut:
        """Convert to corresponding Pydantic schema."""
        return ExtractorOut(
            **(
                self.to_metadata().typed_dict()
                | {
                    "dbdv_min_id": self.version_range_ids[0],
                    "dbdv_max_id": self.version_range_ids[1],
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
        version: "DBDVersionOut",
        players: list["PlayerOut"]
    ) -> FullMatchOut:
        self._check_flushed()
        return FullMatchOut(version=version, players=players)
