"""DBDIE ML package base classes."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from dbdie_classes.paths import recursive_dirname
from dbdie_classes.options.FMT import assert_mt_and_pt, from_fmt
import os
from typing import TYPE_CHECKING, Any
import yaml

from backbone.classes.training import TrainingParams
from backbone.cropping import CropSettings, CropperSwarm

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path, ImgSize

CONFIGS_FD = os.path.join(recursive_dirname(__file__, 2), "configs")
EXTRACTORS_FD = os.path.join(recursive_dirname(__file__, 4), "extractors")


def process_metadata_ifk(metadata: dict) -> tuple:
    assert isinstance(metadata["cs_name"], str)
    assert isinstance(metadata["fmt"], str)

    cs_dict = CropSettings.make_cs_dict(metadata["cps_name"])
    cs = cs_dict[metadata["cs_name"]]
    crop = metadata["fmt"]

    metadata["img_size"] = cs.crop_shapes[crop]
    del metadata["fmt"]
    return metadata, cs


def process_metadata_ifk_none(metadata: dict) -> tuple:
    assert isinstance(metadata["cs_name"], list)
    assert isinstance(metadata["fmt"], list)
    assert len(metadata["cs_name"]) == 2
    assert len(metadata["fmt"]) == 2

    cs_dict = CropSettings.make_cs_dict(metadata["cps_name"])
    both_cs = [
        cs_dict[cs_str]
        for cs_str in metadata["cs_name"]
    ]
    crop_shapes = [
        cs.crop_shapes[crop]
        for cs, crop in zip(both_cs, metadata["fmt"])
    ]
    assert crop_shapes[0] == crop_shapes[1]

    metadata["img_size"] = crop_shapes[0]
    del metadata["fmt"]

    return metadata, both_cs[0]


@dataclass(kw_only=True)
class SavedModelMetadata:
    id: int
    total_classes: int

    cps_name: str
    cs_name: str | list[str]
    fmt: "FullModelType"
    img_size: "ImgSize"
    name: str
    norm_means: list[float]
    norm_std: list[float]
    training: TrainingParams
    version_range: list[str]
    version_range_ids: list[int]

    def __post_init__(self):
        assert self.total_classes > 1

        assert len(self.img_size) == 2
        assert len(self.norm_means) == 3
        assert len(self.norm_std) == 3

        assert len(self.version_range) == 2
        self.version_range = [
            (str(dbdv) if dbdv is not None else None)
            for dbdv in self.version_range
        ]

        assert len(self.version_range_ids) == 2

    def typed_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    def dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    # * Loading and saving

    @classmethod
    def load(
        cls,
        fmt: "FullModelType",
        extr_name: str | None,
        model_id: int | None,
        total_classes: int | None,
        cps_name: str | None,
    ) -> SavedModelMetadata:
        """Load `SavedModelMetadata` from config YAML file."""
        is_trained_model = extr_name is not None
        assert all(
            is_trained_model == (v is None)
            for v in [model_id, total_classes, cps_name]
        ), "Model params should be passed iif the model is being created."

        path = (
            os.path.join(EXTRACTORS_FD, f"{extr_name}/models/{fmt}/metadata.yaml")
            if is_trained_model
            else os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
        )

        with open(path) as f:
            metadata = yaml.safe_load(f)

        if not is_trained_model:
            cps = CropperSwarm.from_register(cps_name)
            cs = cps.get_cs_that_contains_fmt(fmt)

            metadata["id"] = model_id
            metadata["total_classes"] = total_classes
            metadata["img_size"] = cs.crop_shapes[fmt]
            metadata["version_range"] = cps.version_range.to_list()
            metadata["cps_name"] = cps_name
            metadata["version_range_ids"] = cps.version_range_ids

        metadata["training"] = TrainingParams(**metadata["training"])

        return cls(**metadata)

    @classmethod
    def from_model_class(cls, iem) -> SavedModelMetadata:
        metadata = (
            {
                k: getattr(iem, k)
                for k in [
                    "id", "name", "fmt", "total_classes",
                    "cps_name", "cs_name", "version_range_ids",
                ]
            }
            | {k: getattr(iem, f"_{k}") for k in ["norm_means", "norm_std"]}
            | {
                "version_range": iem.version_range.to_list(),
                "img_size": list(iem.img_size),
                "training": TrainingParams(**iem.training_params),
            }
        )
        return cls(**metadata)

    def to_model_class_metadata(self) -> dict:
        """Process IEModel raw metadata dict (straight from the YAML file)."""
        mt, pt, ifk = from_fmt(self.fmt)
        assert_mt_and_pt(mt, pt)

        metadata = (
            self.typed_dict()
            | {"mt": mt, "pt": pt, "ifk": ifk}
            | {"training": self.training.typed_dict()}
        )
        metadata, cs = (
            process_metadata_ifk_none(metadata)
            if ifk is None
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
    cps_name: str
    id: int
    models: dict["FullModelType", int]
    name: str
    version_range: list[str]
    version_range_ids: list[int]

    def __post_init__(self):
        assert self.id >= 0
        assert self.models

        assert len(self.version_range) == 2
        self.version_range = [
            (str(dbdv) if dbdv is not None else None)
            for dbdv in self.version_range
        ]

        assert len(self.version_range_ids) == 2

    def typed_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    def dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def save(self, path: "Path") -> None:
        assert path.endswith(".yaml")
        m = self.dict()
        with open(path, "w") as f:
            yaml.dump(m, f)
