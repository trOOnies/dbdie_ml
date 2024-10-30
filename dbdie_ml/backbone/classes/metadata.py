"""DBDIE ML package base classes."""

from __future__ import annotations
from dataclasses import asdict, dataclass
from dbdie_classes.options.FMT import assert_mt_and_pt, from_fmt
from typing import TYPE_CHECKING, Any
import yaml

from backbone.classes.register import get_model_mpath
from backbone.classes.training import TrainingParams
from backbone.code.metadata import (
    form_metadata_dict,
    load_assertions,
    patch_untrained_metadata,
    process_metadata,
    process_metadata_ifk_none,
)

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, Path, ImgSize


@dataclass(kw_only=True)
class SavedDBDVersion:
    """YAML-formatted `DBDVersion`."""
    id: int
    name: str
    common_name: str | None
    release_date: str | None

    @classmethod
    def from_dbdv(cls, dbdv) -> SavedDBDVersion:
        """Create `SavedDBDVersion` from a `DBDVersion`."""
        return cls(
            id=dbdv.id,
            name=dbdv.name,
            common_name=dbdv.common_name,
            release_date=(
                None if dbdv.release_date is None
                else dbdv.release_date.strftime("%Y-%m-%d")
            ),
        )

    @classmethod
    def dbdvr_to_saved_dbdvs(
        cls,
        dbdvr,
    ) -> tuple[SavedDBDVersion, SavedDBDVersion | None]:
        """Create 2 `SavedDBDVersions` (min and max) from a `DBDVersionRange`."""
        return (
            cls.from_dbdv(dbdvr.dbdv_min),
            (
                cls.from_dbdv(dbdvr.dbdv_max)
                if dbdvr.bounded else None
            ),
        )


@dataclass(kw_only=True)
class SavedModelMetadata:
    """YAML-formatted `IEModel`."""
    id: int
    total_classes: int

    cps_name: str
    cs_name: str | list[str]
    dbdv_max: SavedDBDVersion | None
    dbdv_min: SavedDBDVersion
    fmt: "FullModelType"
    img_size: "ImgSize"
    name: str
    norm_means: list[float]
    norm_std: list[float]
    training: TrainingParams

    def __post_init__(self):
        assert self.total_classes > 1

        assert len(self.img_size) == 2
        assert len(self.norm_means) == 3
        assert len(self.norm_std) == 3

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
        is_trained_model = load_assertions(extr_name, [model_id, total_classes, cps_name])

        path = get_model_mpath(extr_name, fmt, is_trained_model)
        with open(path) as f:
            metadata = yaml.safe_load(f)

        if not is_trained_model:
            metadata = patch_untrained_metadata(
                metadata, SavedDBDVersion, fmt, model_id, cps_name, total_classes,
            )

        metadata["training"] = TrainingParams(**metadata["training"])

        return cls(**metadata)

    @classmethod
    def from_model_class(cls, iem) -> SavedModelMetadata:
        dbdv_min, dbdv_max = SavedDBDVersion.dbdvr_to_saved_dbdvs(iem.dbdvr)
        metadata = form_metadata_dict(iem, dbdv_min, dbdv_max)
        return cls(**metadata)

    def to_model_class_metadata(self) -> dict:
        """Process IEModel raw metadata dict (straight from the YAML file)."""
        mt, pt, ifk = from_fmt(self.fmt)
        assert_mt_and_pt(mt, pt)

        metadata = (
            self.typed_dict()
            | {"mt": mt, "pt": pt, "ifk": ifk, "training": self.training.typed_dict()}
        )
        metadata, cs = (
            process_metadata_ifk_none(metadata)
            if ifk is None
            else process_metadata(metadata)
        )
        metadata["dbdvr"] = cs.dbdvr
        metadata["dbdvr_ids"] = cs.dbdvr.to_ids()

        return metadata

    def save(self, path: "Path") -> None:
        assert path.endswith(".yaml")
        m = self.typed_dict()
        with open(path, "w") as f:
            yaml.dump(m, f)


@dataclass(kw_only=True)
class SavedExtractorMetadata:
    """YAML-formatted `InfoExtractor`."""
    cps_name: str
    dbdv_max: SavedDBDVersion | None
    dbdv_min: SavedDBDVersion
    id: int
    models: dict["FullModelType", dict[str, str | int]]
    name: str

    def __post_init__(self):
        assert self.id >= 0
        assert self.models

    def typed_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items()}

    def dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}

    def save(self, path: "Path") -> None:
        assert path.endswith(".yaml")
        m = self.typed_dict()
        with open(path, "w") as f:
            yaml.dump(m, f)
