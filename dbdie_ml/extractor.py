from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

import yaml

from dbdie_ml.classes.base import Path, PathToFolder, PlayerInfo
from dbdie_ml.classes.version import DBDVersion
from dbdie_ml.code.extractor import (
    folder_save_logic,
    get_printable_info,
    get_version_range,
    match_preds_types,
)
from dbdie_ml.models import IEModel

# from dbdie_ml.db import to_player
from dbdie_ml.schemas.groupings import FullMatchOut

if TYPE_CHECKING:
    from numpy import ndarray

    from dbdie_ml.classes.base import (
        CropCoords,
        FullModelType,
        PlayersCropCoords,
        PlayersInfoDict,
    )
    from dbdie_ml.classes.version import DBDVersionRange
    from dbdie_ml.schemas.groupings import PlayerOut

TYPES_TO_ID_NAMES = {
    "character": "character_id",
    "perks": "perks_ids",
    "item": "item_id",
    "addons": "addons_ids",
    "offering": "offering_id",
    "status": "status_id",
    "points": "points",
}


class InfoExtractor:
    """Extracts information of an image using multiple IEModels.

    Inputs:
        name (str | None): Name of the InfoExtractor.

    Attrs:
        version_range (DBDVersionRange | None): Inferred from its models.
        model_types (list[FullModelType] | None)
        models_are_init (bool)
        models_are_trained (bool)

    Usage:
    >>> ie = InfoExtractor(name="my_info_extractor")
    >>> # ie.models = {"perks_surv": my_model, ...}  # ! only for super-users
    >>> ie.init_extractor()  # this uses all standard models
    >>> ie.train(...)
    >>> ie.save("/path/to/extractor/folder")
    >>> preds_dict = ie.predict_batch({"perks": "/path/to/dataset.csv", ...})

    Load previously trained InfoExtractor:
    >>> ie = InfoExtractor.from_folder("/path/to/extractor/folder")
    >>> new_preds_dict = ie.predict_batch({"perks": "/path/to/other/dataset.csv", ...})
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._set_empty_placeholders()

    def _set_empty_placeholders(self) -> None:
        self._models: Optional[dict["FullModelType", IEModel]] = None
        self.version_range: Optional["DBDVersionRange"] = None

    def __repr__(self) -> str:
        """InfoExtractor('my_info_extractor', version='7.5.0')"""
        vals = f"version='{self.version_range}'"
        if self.name is not None:
            vals = f"'{self.name}', " + vals
        return f"InfoExtractor({vals})"

    @property
    def model_types(self) -> Optional[list["FullModelType"]]:
        return list(self._models.keys()) if self.models_are_init else None

    @property
    def models_are_init(self) -> bool:
        return self._models is not None

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
        trained_models: Optional[dict["FullModelType", IEModel]] = None,
        expected_version_range: Optional["DBDVersionRange"] = None,
    ) -> None:
        """Initialize the InfoExtractor and its IEModels.
        Can be provided with already trained models.
        """
        assert (
            not self.models_are_init
        ), "InfoExtractor can't be reinitialized before being flushed first"

        if trained_models is None:
            from dbdie_ml.models.custom import CharacterModel, PerkModel

            TYPES_TO_MODELS = {
                "character": CharacterModel,
                "perks": PerkModel,
            }
            models = {
                "character__killer": True,
                "character__surv": False,
                "perks__killer": True,
                "perks__surv": False,
            }
            self._models = {
                mt: TYPES_TO_MODELS[mt[: mt.index("")]](
                    name=f"{self.name}__m{i}" if self.name is not None else None,
                    is_for_killer=ifk,
                )
                for i, (mt, ifk) in enumerate(models.items())
            }
        else:
            self._models = trained_models

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

        model_names = set(metadata["models"])
        assert len(model_names) == len(
            metadata["models"]
        ), "Duplicated model names in the metadata YAML file"
        del metadata["models"]

        models_fd = os.path.join(extractor_fd, "models")

        assert model_names == set(
            fd
            for fd in os.listdir(models_fd)
            if os.path.isdir(os.path.join(models_fd, fd))
        ), "The model subfolders do not match the metadata YAML file"
        model_names = list(model_names)

        exp_version_range = metadata["version-range"]
        del metadata["version-range"]

        ie = InfoExtractor(**metadata)
        ie.init_extractor(
            trained_models={
                mn: IEModel.from_folder(os.path.join(models_fd, mn))
                for mn in model_names
            },
            expected_version_range=exp_version_range,
        )
        return ie

    def _save_metadata(self, dst: "Path") -> None:
        assert dst.endswith(".yaml")
        metadata = {k: getattr(self, k) for k in ["name", "version_range"]}
        metadata["models"] = list(self._models.keys())
        with open(dst, "w") as f:
            yaml.dump(metadata, f)

    def save(self, extractor_fd: "PathToFolder", replace: bool = True) -> None:
        """Save all necessary objects of the InfoExtractor and all its IEModels"""
        assert self.models_are_trained, "Non-trained InfoExtractor cannot be saved"
        folder_save_logic(self._models, extractor_fd, replace)
        self._save_metadata(os.path.join(extractor_fd, "metadata.yaml"))
        for mn, model in self._models.items():
            model.save(os.path.join(extractor_fd, "models", mn))

    # * Training

    def _check_datasets(self, datasets: dict["FullModelType", Path]) -> None:
        assert set(self.model_types) == set(datasets.keys())
        assert all(os.path.exists(p) for p in datasets.values())

    def train(
        self,
        label_ref_paths: dict["FullModelType", Path],
        train_datasets: dict["FullModelType", Path],
        val_datasets: dict["FullModelType", Path],
    ) -> None:
        """Train all models one after the other."""
        assert not self.models_are_trained

        self._check_datasets(label_ref_paths)
        self._check_datasets(train_datasets)
        self._check_datasets(val_datasets)

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
        """Reset InfoExtractor to pre-init state."""
        model_names = (mn for mn in self._models)
        for mn in model_names:
            self._models[mn].flush()
        self._set_empty_placeholders()

    # * Prediction

    def predict_on_crop(self, crop: "CropCoords") -> PlayerInfo:
        preds = {
            TYPES_TO_ID_NAMES[k]: model.predict(crop)
            for k, model in self._models.items()
        }
        return PlayerInfo(**preds)

    def predict(self, player_crops: "PlayersCropCoords") -> "PlayersInfoDict":
        return {i: self.predict_on_crop(s) for i, s in player_crops.items()}

    def predict_batch(
        self, datasets: dict["FullModelType", Path], probas: bool = False
    ) -> dict["FullModelType", "ndarray"]:
        assert self.models_are_trained
        self._check_datasets(datasets)
        return {
            mt: model.predict_batch(datasets[mt], probas=probas)
            for mt, model in self._models.items()
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
        assert self.models_are_trained

        preds_, on_ = match_preds_types(preds, on)
        names = {k: self.models[k].convert_names(preds[k]) for k in on_}
        return names[list(names.keys())[0]] if isinstance(on, str) else names

    # * Match

    def form_match(self, version: DBDVersion, players: list["PlayerOut"]) -> FullMatchOut:
        return FullMatchOut(version=version, players=players)
