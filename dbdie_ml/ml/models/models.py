from __future__ import annotations

import json
import os
import yaml
from functools import partial
from typing import TYPE_CHECKING, Optional

import numpy as np
from torch.cuda import device as get_device
from torch.cuda import is_available as cuda_is_available
from torch.cuda import mem_get_info
from torch.utils.data import DataLoader
from torchsummary import summary

from dbdie_ml.code.models import (
    is_str_like,
    load_label_ref,
    process_metadata,
    load_with_trained_model,
    save_metadata,
    save_model,
)
from dbdie_ml.code.training import (
    TrainingConfig,
    load_label_ref_for_train,
    load_process,
    load_training_config,
    predict_probas_process,
    predict_process,
    train_process,
)
from dbdie_ml.data import DatasetClass, get_total_classes

if TYPE_CHECKING:
    from torch.nn import Sequential

    from dbdie_ml.classes.base import (
        CropCoords,
        Height,
        Path,
        PathToFolder,
        Width,
    )


class IEModel:
    """ML model for information extraction

    Inputs:
        model (torch.nn.Sequential)
        model_type (ModelType)
        is_for_killer (bool)
        img_size (tuple[int, int])
        version_range (DBDVersionRange): DBD game version range for which
            the model works.
        norm_means (list[float]): 3 floats for the torch 'Compose'.
        norm_std (list[float]): Idem norm_means.
        name (str | None): Model name.

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

    Load previously trained 'IEModel':
    >>> model = IEModel.from_folder("/path/to/model/folder")
    >>> new_preds = model.predict_batch("/path/to/other/dataset.csv")
    """

    def __init__(
        self,
        metadata: dict,
        model: "Sequential",
    ) -> None:
        metadata = process_metadata(metadata)
        for k, v in metadata.items():
            setattr(self, k, v)

        self.selected_fd = (
            f"{self.model_type}{'killer' if self.is_for_killer else 'surv'}"
        )
        self._model = model

        self._set_empty_placeholders()

    def _set_empty_placeholders(self) -> None:
        self.total_classes: Optional[int] = None

        self._device = None
        self.cfg: Optional[TrainingConfig] = None

        self.label_ref: Optional[dict[int, str]] = None
        self.model_is_trained = False

    def __repr__(self) -> str:
        """IEModel(...)"""
        vals = {
            "type": self.model_type,
            "for_killer": self.is_for_killer,
            "version": self.version_range,
            "classes": self.total_classes,
            "trained": self.model_is_trained,
        }
        vals = ", ".join(
            [f"{k}='{v}'" if is_str_like(v) else f"{k}={v}" for k, v in vals.items()]
        )
        if self.name is not None:
            vals = f"'{self.name}', " + vals
        return f"IEModel({vals})"

    @property
    def model_is_init(self) -> bool:
        return self.cfg is not None

    # * Base

    def init_model(self) -> None:
        """Initialize model to allow it to be trained"""
        assert (
            not self.model_is_init
        ), "IEModel can't be reinitialized before being flushed first"

        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

        assert cuda_is_available()
        self._device = get_device("cuda")

        self.total_classes = get_total_classes(self.selected_fd)
        self._model = self._model.cuda()

        load_training_config(self)

    def get_summary(self) -> None:
        """Get underlying model summary"""
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

        print("MEMORY")
        print(
            "- Free: {:,.2} GiB\n- Total: {:,.2} GiB".format(
                *[v / (2**30) for v in mem_get_info(self._device)]
            )
        )
        print(64 * "-")

    # * Loading and saving

    @classmethod
    def from_folder(cls, model_fd: "PathToFolder") -> IEModel:
        """Loads a DBDIE model using its metadata and the actual model"""
        # TODO: Check if any other PT object needs to be saved
        with open(os.path.join(model_fd, "metadata.yaml"), "r") as f:
            metadata = yaml.safe_load(f)

        metadata = process_metadata(model_fd)
        iem = load_with_trained_model(cls, model_fd, metadata)
        iem.label_ref = load_label_ref(model_fd)
        return iem

    def save(self, model_fd: "PathToFolder") -> None:
        """Save all necessary objects of the IEModel"""
        print("Saving model...", end=" ")
        if not os.path.isdir(model_fd):
            os.mkdir(model_fd)

        mfd = partial(os.path.join, model_fd)

        save_metadata(self, mfd("metadata.yaml"))
        with open(mfd("label_ref.json"), "w") as f:
            json.dump(self.label_ref, f, indent=4)
        save_model(self, mfd("model.pt"))

        print("Model saved.")

    def flush(self) -> None:
        """Reset IEModel to pre-init state."""
        # TODO: Reinit a flushed model
        if not self.model_is_init:
            return
        del self._model
        self._model = None
        self._set_empty_placeholders()

    # * Training

    def train(
        self,
        label_ref_path: "Path",
        train_dataset_path: "Path",
        val_dataset_path: "Path",
    ) -> None:
        """Trains the 'IEModel'"""
        # TODO: Add training scores as attributes once trained
        assert self.model_is_init, "IEModel is not initialized"
        assert (
            not self.model_is_trained
        ), "IEModel cannot be retrained without being flushed first"
        self.label_ref = load_label_ref_for_train(label_ref_path)
        train_loader, val_loader = load_process(
            self.selected_fd,
            train_dataset_path,
            val_dataset_path,
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
        raise NotImplementedError

    def predict_batch(self, dataset_path: "Path", probas: bool = False) -> np.ndarray:
        """Returns: preds or probas"""
        assert self.model_is_trained, "IEModel is not trained"

        print("Predictions for:", dataset_path)
        dataset = DatasetClass(
            self.selected_fd,
            dataset_path,
            transform=self.cfg.transform,
        )
        loader = DataLoader(dataset, batch_size=self.batch_size)

        if probas:
            return predict_probas_process(
                self._model,
                dataset,
                loader,
                self.total_classes,
            )
        else:
            return predict_process(self._model, dataset, loader)

    def convert_names(self, preds: np.ndarray) -> list[str]:
        """Convert integer predictions to named predictions"""
        assert isinstance(preds[0], (np.ushort, int))
        assert self.model_is_trained, "IEModel is not trained"
        return [self.label_ref[lbl] for lbl in preds]

    @staticmethod
    def save_preds(preds: np.ndarray, dst: "Path") -> None:
        """Save predictions to the 'dst' path"""
        print("Saving preds...", end=" ")
        assert dst.endswith(".txt")
        np.savetxt(dst, preds, fmt="%d")
        print("Preds saved.")
