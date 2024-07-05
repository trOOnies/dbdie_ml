import os
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union
from dbdie_ml.classes import SnippetInfo, Path
from dbdie_ml.models import IEModel
from dbdie_ml.db import to_player
from dbdie_ml.schemas import MatchOut
if TYPE_CHECKING:
    from numpy import ndarray
    from dbdie_ml.classes import AllSnippetCoords, AllSnippetInfo, SnippetCoords, FullModelType
    from dbdie_ml.schemas import PlayerOut

TYPES_TO_ID_NAMES = {
    "character": "character_id",
    "perks": "perks_ids",
    "item": "item_id",
    "addons": "addons_ids",
    "offering": "offering_id",
    "status": "status_id",
    "points": "points"
}


class InfoExtractor:
    """Extracts information of an image using multiple `IEModels`.

    Inputs:
        name: An optional string.

    Attributes:
        version: A string that is inferred from its models.
        model_types: An optional list of `FullModelTypes`.

    Usage:
        >>> ie = InfoExtractor()
        >>> # ie._models = {"perks": my_model, ...}  # only for super-users
        >>> ie.init_extractor()  # this uses all standard models
        >>> ie.train(...)
        >>> preds_dict = ie.predict_batch({"perks": "/path/to/dataset.csv", ...})
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._models: Optional[dict["FullModelType", IEModel]] = None
        self.version: Optional[str] = None

    def __repr__(self) -> str:
        vals = {"version": self.version if self.models_are_init else "not_initialized"}
        vals = ', '.join([f"{k}='{v}'" for k, v in vals.items()])
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

    def _init_models(self) -> None:
        assert all(model.selected_fd == mt for mt, model in self._models.items())

        version = {model.version for model in self._models.values()}
        assert len(version) == 1, "All model versions must match"
        self.version = list(version)[0]

        for model in self._models.values():
            model.init_model()

    def init_extractor(self) -> None:
        if self._models is None:
            from dbdie_ml.models.custom import PerkModel, CharacterModel
            self._models = {
                "perks__killer": PerkModel(is_for_killer=True),
                "perks__surv": PerkModel(is_for_killer=False),
                "character__killer": CharacterModel(is_for_killer=True),
                "character__surv": CharacterModel(is_for_killer=False)
            }
        self._init_models()

    def _check_datasets(self, datasets: dict["FullModelType", Path]) -> None:
        assert set(self.model_types) == set(datasets.keys())
        assert all(os.path.exists(p) for p in datasets.values())

    def train(
        self,
        label_ref_paths: dict["FullModelType", Path],
        train_datasets: dict["FullModelType", Path],
        val_datasets: dict["FullModelType", Path]
    ) -> None:
        """Train all models one after the other."""
        # TODO: Fix
        assert self.models_are_init and not self.models_are_trained
        self._check_datasets(label_ref_paths)
        self._check_datasets(train_datasets)
        self._check_datasets(val_datasets)
        print(50 * "-")
        for mt, model in self._models.items():
            print(f"Training {mt} model...")
            model.train(
                label_ref_path=label_ref_paths[mt],
                train_dataset_path=train_datasets[mt],
                val_dataset_path=val_datasets[mt]
            )
            print(50 * "-")
        print("All models have been trained.")

    def save(self) -> None:
        assert self.models_are_trained
        raise NotImplementedError

    def predict_on_snippet(self, s: "SnippetCoords") -> SnippetInfo:
        preds = {
            TYPES_TO_ID_NAMES[k]: model.predict(s)
            for k, model in self._models.items()
        }
        return SnippetInfo(**preds)

    def predict(self, snippets: "AllSnippetCoords") -> "AllSnippetInfo":
        return {
            i: self.predict_on_snippet(s)
            for i, s in snippets.items()
        }

    def predict_batch(
        self,
        datasets: dict["FullModelType", Path],
        probas: bool = False
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
        on: Optional[Union["FullModelType", list["FullModelType"]]] = None
    ) -> Union[list[str], dict["FullModelType", list[str]]]:
        assert self.models_are_trained

        preds_is_dict = isinstance(preds, dict)
        if preds_is_dict:
            preds_ = {deepcopy(k): p for k, p in preds.items()}
            if on is None:
                on_ = list(preds_is_dict.keys())
            elif isinstance(on, list):
                on_ = deepcopy(on)
            else:  # Modeltype
                on_ = [deepcopy(on)]
        else:
            assert isinstance(on, str), "'on' must be a FullModelType if 'preds' is not a dict."
            preds_ = {deepcopy(on): preds}
            on_ = [deepcopy(on)]

        names = {
            k: self._models[k].convert_names(preds_[k])
            for k in on_
        }
        if isinstance(on, str):
            return names[list(names.keys())[0]]  # list
        else:
            return names  # dict

    @staticmethod
    def to_players(snippets_info: "AllSnippetInfo") -> list["PlayerOut"]:
        return [to_player(i, sn_info) for i, sn_info in snippets_info.items()]

    def form_match(self, players: list["PlayerOut"]) -> MatchOut:
        return MatchOut(
            version=self.version,
            players=players
        )

    def end_models(self) -> None:
        model_names = (mn for mn in self._models)
        for mn in model_names:
            del self._models[mn]
        self._models = None
        self.version = None
