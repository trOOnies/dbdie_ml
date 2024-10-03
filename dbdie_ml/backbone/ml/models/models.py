"""Extraction models (IEModel) code."""

from __future__ import annotations

import os
from functools import partial
import pandas as pd
from typing import TYPE_CHECKING

from dbdie_classes.options.FMT import to_fmt
from dbdie_classes.options.PLAYER_TYPE import ifk_to_pt
import numpy as np
from torch.cuda import device as get_device
from torch.cuda import empty_cache
from torch.cuda import is_available as cuda_is_available
from torch.utils.data import DataLoader
from torchsummary import summary

from backbone.classes.metadata import SavedModelMetadata
from backbone.code.models import (
    is_str_like,
    load_label_ref,
    load_metadata_and_model,
    print_memory,
    save_label_ref,
    save_metadata,
    save_model,
)
from backbone.code.training import (
    label_ref_transformations,
    load_process,
    load_training_config,
    predict_probas_process,
    predict_process,
    train_process,
)
from backbone.data import DatasetClass
from backbone.options.COLORS import OKGREEN, make_cprint_with_header

if TYPE_CHECKING:
    from pandas import DataFrame
    from torch.nn import Sequential

    from dbdie_classes.base import (
        FullModelType,
        ImgSize,
        LabelName,
        ModelType,
        Path,
        PathToFolder,
        PlayerType,
    )
    from dbdie_classes.extract import CropCoords
    from dbdie_classes.version import DBDVersionRange

iem_print = make_cprint_with_header(OKGREEN, "[IEModel]")


class IEModel:
    """ML model for information extraction

    Inputs:
        metadata (dict): Raw metadata YAML for the IEModel configuration.
            Its processing is done within the init function.
        model (torch.nn.Sequential)

    Parameters:
        name (str): IEModel name.
        is_for_killer (bool | None)
        model_type (ModelType)
        img_size (tuple[int, int])
        version_range (DBDVersionRange): DBD game version range for which
            the model works.
        norm_means (list[float]): 3 floats for the torch 'Compose'.
        norm_std (list[float]): Idem norm_means.

    Usage:
    >>> model = IEModel(Sequential(...), "perks", "7.6.0", ...)
    >>> model.init_model()  # this uses all standard models
    >>> model.get_summary()
    >>> model.train(...)
    >>> model.save("/path/to/model/folder")
    >>> preds = model.predict_batch("/path/to/dataset.csv")
    >>> names = model.convert_names(preds)
    >>> model.save_preds(preds, "/path/to/preds.txt")
    >>> probas = model.predict_batch("/path/to/dataset.csv", probas=True)

    Deleting a model
    >>> model.flush()
    >>> del model

    Load previously trained 'IEModel':
    >>> model = IEModel.from_folder("/path/to/model/folder")
    >>> new_preds = model.predict_batch("/path/to/other/dataset.csv")
    """

    def __init__(
        self,
        metadata: SavedModelMetadata,
        model: "Sequential",
        total_classes: int,
    ) -> None:
        assert total_classes > 1
        md = metadata.to_model_class_metadata()

        self.id              :               int = md["id"]
        self.name            :               str = md["name"]
        self.ifk             :        str | None = md["is_for_killer"]
        self.mt              :       "ModelType" = md["model_type"]
        self.img_size        :         "ImgSize" = md["img_size"]  # replaces cs & crop
        self.version_range   : "DBDVersionRange" = md["version_range"]
        self._norm_means     :       list[float] = md["norm_means"]
        self._norm_std       :       list[float] = md["norm_std"]
        self.training_params                     = md["training"]

        self.pt:     "PlayerType" = ifk_to_pt(self.ifk)
        self.fmt: "FullModelType" = to_fmt(self.mt, self.ifk)

        self.flushed: bool = False
        self._model = model
        self.model_is_trained: bool = False
        self.total_classes: int = total_classes

    def __repr__(self) -> str:
        """IEModel(...)"""
        vals = {
            "version": self.version_range,
            "trained": self.model_is_trained,
        }
        vals = ", ".join(
            [f"{k}='{v}'" if is_str_like(v) else f"{k}={v}" for k, v in vals.items()]
        )
        vals = f"id={id.self}, name='{self.name}', cl={self.total_classes} {self.pt}" + vals
        return f"IEModel({vals})"

    @property
    def model_is_init(self) -> bool:
        try:
            getattr(self, "cfg")
            return True
        except Exception:
            return False

    # * Base

    def init_model(self) -> None:
        """Initialize model to allow it to be trained."""
        assert not self.flushed, "IEModel was flushed."
        assert not self.model_is_init, "IEModel can't be initialized more than once."

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        assert cuda_is_available()
        self._device = get_device("cuda")

        self._model = self._model.cuda()

        self.cfg = load_training_config(self)

    def get_summary(self) -> None:
        """Get underlying model summary."""
        assert not self.flushed, "IEModel was flushed."
        assert self.model_is_init

        summary(
            self._model,
            input_size=(
                3,
                self.img_size[0],
                self.img_size[1],
            ),  # The number of channels is 3 (RGB)
            batch_size=self.cfg.batch_size,
            device="cuda",
        )

        print_memory(self._device)
        print(64 * "-")

    # * Loading and saving

    @classmethod
    def from_folder(cls, model_fd: "PathToFolder") -> IEModel:
        """Loads a DBDIE model using its metadata and the actual model."""
        # TODO: Check if any other PT object needs to be saved
        metadata, model, total_classes = load_metadata_and_model(model_fd)

        iem = cls(metadata, model=model, total_classes=total_classes)

        # Init the already trained model
        iem.init_model()
        iem.model_is_trained = True
        iem.label_ref = load_label_ref(model_fd)
        iem.to_net_ids, iem.to_label_ids = label_ref_transformations(iem.label_ref)

        return iem

    def save(self, model_fd: "PathToFolder") -> None:
        """Save all necessary objects of the IEModel."""
        assert not self.flushed, "IEModel was flushed."
        iem_print("Saving model...")
        if not os.path.isdir(model_fd):
            os.mkdir(model_fd)

        mfd = partial(os.path.join, model_fd)

        save_metadata(self, mfd("metadata.yaml"))
        save_label_ref(self.label_ref, mfd("label_ref.json"))
        save_model(self.model_is_trained, self._model, mfd("model.pt"))

        iem_print("Model saved.")

    def flush(self) -> None:
        """Flush IEModel params so as to free space.
        A flushed IEModel shouldn't be reused, but deleted and reinstantiated.
        """
        assert not self.flushed, "IEModel was flushed."
        self.flushed = True
        if not self.model_is_init:
            return
        del self._model
        del self.cfg
        del self._device
        empty_cache()

    # * Training

    def train(
        self,
        label_ref_path: "Path",
        train_dataset_path: "Path",
        val_dataset_path: "Path",
    ) -> None:
        """Trains the 'IEModel'."""
        assert not self.flushed, "IEModel was flushed."

        # TODO: Add training scores as attributes once trained
        assert self.model_is_init, "IEModel is not initialized"
        assert (
            not self.model_is_trained
        ), "IEModel cannot be retrained without being flushed first."

        label_ref = pd.read_csv(
            label_ref_path,
            usecols=["id", "name", "net_id"],
            dtype={"id": int, "name": str, "net_id": int},
        )
        self.label_ref = {row["id"]: row["name"] for _, row in label_ref.iterrows()}
        del label_ref

        self.to_net_ids, self.to_label_ids = label_ref_transformations(self.label_ref)

        train_loader, val_loader = load_process(
            self.fmt,
            train_dataset_path,
            val_dataset_path,
            self.to_net_ids,
            cfg=self.cfg,
        )
        train_process(
            self._model,
            train_loader,
            val_loader,
            cfg=self.cfg,
        )
        self.model_is_trained = True

    # * Prediction

    def predict(self, crop: "CropCoords"):
        assert not self.flushed, "IEModel was flushed."
        raise NotImplementedError

    def predict_batch(
        self,
        raw_dataset: "Path" | "DataFrame",
        probas: bool = False,
    ) -> np.ndarray:
        """Returns: preds or probas."""
        assert not self.flushed, "IEModel was flushed."
        assert self.model_is_trained, "IEModel is not trained"

        dataset = DatasetClass(
            self.fmt,
            raw_dataset,
            self.to_net_ids,
            self.cfg.transform,
        )
        loader = DataLoader(dataset, batch_size=self.cfg.batch_size)

        if probas:
            return predict_probas_process(
                self._model,
                dataset,
                loader,
                self.total_classes,
            )
        else:
            return predict_process(self._model, dataset, loader)

    def convert_names(self, preds: np.ndarray) -> list["LabelName"]:
        """Convert integer predictions to named predictions."""
        assert not self.flushed, "IEModel was flushed."
        assert isinstance(preds[0], (np.ushort, int))
        assert self.model_is_trained, "IEModel is not trained"
        return [self.label_ref[lbl] for lbl in preds]

    @staticmethod
    def save_preds(preds: np.ndarray, dst: "Path") -> None:
        """Save predictions to the 'dst' path."""
        iem_print("Saving preds...")
        assert dst.endswith(".txt")
        np.savetxt(dst, preds, fmt="%d")
        iem_print("Preds saved.")
