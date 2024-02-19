import os
from typing import TYPE_CHECKING, Optional, Union, Dict, List
from dbdie_ml.classes import SnippetInfo
from dbdie_ml.models import IEModel
from dbdie_ml.db import to_player
from dbdie_ml.schemas import MatchOut
if TYPE_CHECKING:
    from numpy import ndarray
    from dbdie_ml.classes import AllSnippetCoords, AllSnippetInfo, SnippetCoords, ModelType
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
    """Extracts information of an image using multiple `IEModels`."""
    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name
        self._models: Optional[Dict["ModelType", IEModel]] = None
        self.version: Optional[str] = None

    def __repr__(self) -> str:
        vals = {"version": self.version if self.models_are_init else "not_initialized"}
        vals = ', '.join([f"{k}='{v}'" for k, v in vals.items()])
        if self.name is not None:
            vals = f"'{self.name}', " + vals
        return f"InfoExtractor({vals})"

    @property
    def model_types(self) -> Optional[List["ModelType"]]:
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
        version = None
        for mt, model in self._models.items():
            assert mt == model.model_type

            model.init_model()
            if version is None:
                version = model.version
            else:
                assert model.version == version, "All model versions must match"
        self.version = version

    def init_extractor(self) -> None:
        if self._models is None:
            from dbdie_ml.models.custom import PerkModel
            self._models = {
                "perks": PerkModel()
            }
        self._init_models()

    def _check_datasets(self, datasets: Dict["ModelType", str]) -> None:
        assert set(self.model_types) == set(datasets.keys())
        assert all(os.path.exists(p) for p in datasets.values())

    def train(
        self,
        label_ref_paths: Dict["ModelType", str],
        train_datasets: Dict["ModelType", str],
        val_datasets: Dict["ModelType", str]
    ) -> None:
        """Train all models one after the other."""
        assert self.models_are_init and not self.models_are_trained
        self._check_datasets(label_ref_paths)
        self._check_datasets(train_datasets)
        self._check_datasets(val_datasets)
        for mt, model in self._models.items():
            model.train(
                label_ref_path=label_ref_paths[mt],
                train_dataset_path=train_datasets[mt],
                val_dataset_path=val_datasets[mt]
            )

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

    def predict_batch(self, datasets: Dict["ModelType", str]) -> Dict["ModelType", "ndarray"]:
        assert self.models_are_trained
        self._check_datasets(datasets)
        return {
            mt: model.predict_batch(datasets[mt])
            for mt, model in self._models.items()
        }

    def convert_names(
        self,
        preds: Union["ndarray", Dict["ModelType", "ndarray"]],
        on: Optional[Union["ModelType", List["ModelType"]]] = None
    ) -> Union[List[str], Dict["ModelType", List[str]]]:
        raise NotImplementedError

    @staticmethod
    def to_players(snippets_info: "AllSnippetInfo") -> List["PlayerOut"]:
        return [to_player(i, sn_info) for i, sn_info in snippets_info.items()]

    def form_match(self, players: List["PlayerOut"]) -> MatchOut:
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
