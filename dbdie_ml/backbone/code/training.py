"""Extra for the training script."""

from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import torch.nn.functional as F
from torch import max as torch_max
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from backbone.data import DatasetClass
from backbone.options.COLORS import OKBLUE, make_cprint_with_header

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from dbdie_classes.base import FullModelType, LabelId, LabelRef, NetId, Path

iem_print = make_cprint_with_header(OKBLUE, "[IEModel]")


class EarlyStopper:
    """In charge of the early-stopping in the training process"""

    def __init__(self, patience=1, min_delta=0.0):
        assert min_delta >= 0.0
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        """Check if training should early stop"""
        val_diff = self.min_validation_loss - validation_loss
        if val_diff > self.min_delta:
            self.min_validation_loss = validation_loss
            self.counter = 0
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience


@dataclass
class TrainingConfig:
    """Helper dataclass for training configuration."""
    epochs: int
    batch_size: int
    optimizer: "Optimizer"
    criterion: Any
    transform: transforms.Compose
    estop: EarlyStopper


# * Base


def get_transform(
    norm_means: list[float],
    norm_std: list[float],
) -> transforms.Compose:
    """Define any image transformations here"""
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_means, std=norm_std),
        ]
    )


def load_training_config(iem) -> TrainingConfig:
    cfg = deepcopy(iem.training_params)
    cfg = cfg | {
        "optimizer": Adam(
            iem._model.parameters(),
            lr=iem.training_params["adam_lr"],
        ),
        "criterion": CrossEntropyLoss(),
        "transform": get_transform(iem._norm_means, iem._norm_std),
        "estop": EarlyStopper(patience=3, min_delta=0.01),
    }
    del cfg["adam_lr"]

    return TrainingConfig(**cfg)


# * Training


def label_ref_transformations(
    label_ref: "LabelRef",
) -> tuple[dict["LabelId", "NetId"], np.ndarray]:
    return (
        {lid: i for i, lid in enumerate(label_ref.keys())},
        np.fromiter((v for v in label_ref.keys()), dtype=int),
    )


def load_process(
    full_model_type: "FullModelType",
    train_ds_path: "Path",
    val_ds_path: "Path",
    to_net_ids,
    cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    iem_print("Loading data...")

    datasets = {
        "train": DatasetClass(
            full_model_type,
            train_ds_path,
            to_net_ids,
            transform=cfg.transform,
        ),
        "val": DatasetClass(
            full_model_type,
            val_ds_path,
            to_net_ids,
            transform=cfg.transform,
        ),
    }

    loaders = {
        "train": DataLoader(datasets["train"], batch_size=cfg.batch_size, shuffle=True),
        "val": DataLoader(datasets["val"], batch_size=cfg.batch_size),
    }

    iem_print("Data loaded.")
    iem_print(f"- Datapoints train: {len(datasets['train'])}")
    iem_print(f"- Datapoints val:   {len(datasets['val'])}")

    return loaders["train"], loaders["val"]


def train_backprop(model, train_loader: DataLoader, cfg: TrainingConfig):
    """Train backpropagation"""
    for images, labels in train_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        loss = cfg.criterion(outputs, labels)

        cfg.optimizer.zero_grad()
        loss.backward()
        cfg.optimizer.step()

    return loss


def train_eval(model, val_loader: DataLoader) -> float:
    """Calculate validation accuracy in percentage points"""
    correct = 0
    total = 0
    for images, labels in val_loader:
        images = images.cuda()
        labels = labels.cuda()

        outputs = model(images)
        _, predicted = torch_max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    return 100.0 * correct / total


def train_process(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: TrainingConfig,
) -> None:
    iem_print("Training model...")

    epochs_clen = len(str(cfg.epochs))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss = train_backprop(model, train_loader, cfg)

        model.eval()
        with no_grad():
            val_acc_pp = train_eval(model, val_loader)
            iem_print(
                (
                    f"- Epoch [{epoch:>{epochs_clen}}/{cfg.epochs}] "
                    + f"Loss: {loss.item():.4f} "
                    + f"Val Acc: {val_acc_pp:.2f}%"
                )
            )
            if cfg.estop.early_stop(100.0 - val_acc_pp):
                break

    model.eval()
    iem_print("Training complete.")


# * Prediction


def predict_process(
    model,
    dataset: DatasetClass,
    loader: DataLoader,
) -> np.ndarray:
    all_preds = np.zeros(len(dataset), dtype=np.ushort)
    i = 0
    with no_grad():
        for images, labels in loader:
            labels_len = labels.size()[0]
            images = images.cuda()

            outputs = model(images)
            _, predicted = torch_max(outputs.data, 1)
            all_preds[i : i + labels_len] = predicted.cpu().numpy()
            i += labels_len
    return all_preds


def predict_probas_process(
    model,
    dataset: DatasetClass,
    loader: DataLoader,
    total_classes: int,
) -> np.ndarray:
    all_preds = np.zeros((len(dataset), total_classes), dtype=float)
    i = 0
    with no_grad():
        for images, labels in loader:
            labels_len = labels.size()[0]
            images = images.cuda()

            outputs = model(images)
            outputs = F.softmax(outputs, dim=1)
            all_preds[i : i + labels_len, :] = outputs.cpu().numpy()
            i += labels_len
    return all_preds
