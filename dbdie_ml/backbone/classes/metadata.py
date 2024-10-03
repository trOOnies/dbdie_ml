"""DBDIE ML package base classes."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from dbdie_classes.paths import recursive_dirname
from dbdie_classes.options.FMT import assert_mt_and_pt, from_fmt
import os
from typing import TYPE_CHECKING
import yaml

from backbone.classes.training import TrainingParams
from backbone.cropping import CropSettings

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path, ImgSize

CONFIGS_FD = os.path.join(recursive_dirname(__file__, 1), "configs")
EXTRACTORS_FD = os.path.join(recursive_dirname(__file__, 2), "extractors")


def process_metadata_ifk(metadata: dict) -> tuple:
    assert isinstance(metadata["cs"], str)
    assert isinstance(metadata["crop"], str)

    cs_dict = CropSettings.make_cs_dict(metadata["cs_name"])
    cs = cs_dict[metadata["cs"]]
    crop = metadata["crop"]

    metadata["img_size"] = cs.crop_shapes[crop]
    del metadata["cs"], metadata["crop"]
    return metadata, cs


def process_metadata_ifk_none(metadata: dict) -> tuple:
    assert isinstance(metadata["cs"], list)
    assert isinstance(metadata["crop"], list)
    assert len(metadata["cs"]) == 2
    assert len(metadata["crop"]) == 2

    cs_dict = CropSettings.make_cs_dict(metadata["cs_name"])
    both_cs = [
        cs_dict[cs_str]
        for cs_str in metadata["cs"]
    ]
    crop_shapes = [
        cs.crop_shapes[crop]
        for cs, crop in zip(both_cs, metadata["crop"])
    ]
    assert crop_shapes[0] == crop_shapes[1]

    metadata["img_size"] = crop_shapes[0]
    del metadata["cs"], metadata["crop"]

    return metadata, both_cs[0]


@dataclass(kw_only=True)
class SavedModelMetadata:
    id: int
    total_classes: int
    img_size: "ImgSize"  # Turn as the others
    version_range: list[str]  # Turn as the others

    cs_name: str
    fmt: "FullModelType"
    name: str
    norm_means: list[float]
    norm_std: list[float]
    training: TrainingParams

    def __post_init__(self):
        assert self.total_classes > 1

        assert len(self.norm_means) == 3
        assert len(self.norm_std) == 3

        assert len(self.version_range) == 2
        self.version_range = [
            (str(dbdv) if dbdv is not None else None)
            for dbdv in self.version_range
        ]

    def dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    # * Loading and saving

    @classmethod
    def load(
        cls,
        extr_name: str | None,
        model_id: int | None,
        total_classes: int | None,
        fmt: "FullModelType",
    ) -> SavedModelMetadata:
        """Load `SavedModelMetadata` from config YAML file."""
        is_trained_model = extr_name is not None
        assert (
            is_trained_model == (model_id is None)
        ), "'model_id' should be passed iif the model is being created."
        assert (
            is_trained_model == (total_classes is None)
        ), "'total_classes' should be passed iif the model is being created."

        path = (
            os.path.join(EXTRACTORS_FD, f"{extr_name}/models/{fmt}/metadata.yaml")
            if is_trained_model
            else os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
        )

        with open(path) as f:
            metadata = yaml.safe_load(f)
        if not is_trained_model:
            metadata["id"] = model_id
            metadata["total_classes"] = total_classes

        metadata["training"] = TrainingParams(**metadata["training"])

        return cls(**metadata)

    @classmethod
    def from_model_class(cls, iem) -> SavedModelMetadata:
        return cls(
            {
                k: getattr(iem, k) for k in ["id", "name", "mt", "ifk", "total_classes"]
            }
            | {k: getattr(iem, f"_{k}") for k in ["norm_means", "norm_std"]}
            | {
                "version_range": [iem.version_range.id, iem.version_range.max_id],
                "img_size": list(iem.img_size),
                "training": TrainingParams(**iem.training_params),
            }
        )

    def to_model_class_metadata(self) -> dict:
        """Process IEModel raw metadata dict (straight from the YAML file)."""
        mt, pt, ifk = from_fmt(self.fmt)
        assert_mt_and_pt(mt, pt)

        metadata = (
            self.dict()
            | {"mt": mt, "pt": pt, "ifk": ifk}
            | {"training": self.training.dict()}
        )
        metadata, cs = (
            process_metadata_ifk_none(metadata)
            if self.ifk is None
            else process_metadata_ifk(metadata)
        )
        metadata["version_range"] = cs.version_range

        return metadata

    def save(self, path: "Path") -> None:
        assert path.endswith(".yaml")
        m = self.dict()
        with open(path, "w") as f:
            yaml.dump(m, f)


@dataclass(kw_only=True)
class SavedExtractorMetadata:
    id: int
    models: dict["FullModelType", int]
    name: str
    version_range: list[str]

    def __post_init__(self):
        assert len(self.version_range) == 2
        self.version_range = [
            (str(dbdv) if dbdv is not None else None)
            for dbdv in self.version_range
        ]
