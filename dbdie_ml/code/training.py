from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import max as torch_max
from torch import no_grad
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms

from dbdie_ml.data import DatasetClass

if TYPE_CHECKING:
    from torch.optim import Optimizer

    from dbdie_classes.base import FullModelType, Path


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
    """Helper dataclass for training configuration"""

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


def load_label_ref_for_train(path: "Path") -> dict[int, str]:
    label_ref = pd.read_csv(
        path,
        usecols=["label_id", "name"],
        dtype={"label_id": int, "name": str},
    )

    unique_vals = label_ref.label_id.unique()
    assert unique_vals.min() == 0
    assert unique_vals.max() + 1 == label_ref.shape[0]
    assert unique_vals.size == label_ref.shape[0]

    return {row["label_id"]: row["name"] for _, row in label_ref.iterrows()}


def load_process(
    full_model_type: "FullModelType",
    train_ds_path: "Path",
    val_ds_path: "Path",
    cfg: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    print("Loading data...", end=" ")

    train_dataset = DatasetClass(
        full_model_type,
        train_ds_path,
        transform=cfg.transform,
    )
    val_dataset = DatasetClass(
        full_model_type,
        val_ds_path,
        transform=cfg.transform,
    )

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size)

    print("Data loaded.")
    print("- Train datapoints:", len(train_dataset))
    print("- Val datapoints:", len(val_dataset))

    return train_loader, val_loader


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
    print("Training initialized...")
    epochs_clen = len(str(cfg.epochs))
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        loss = train_backprop(model, train_loader, cfg)

        model.eval()
        with no_grad():
            val_acc_pp = train_eval(model, val_loader)
            print(
                f"- Epoch [{epoch:>{epochs_clen}}/{cfg.epochs}]",
                f"Loss: {loss.item():.4f}",
                f"Val Acc: {val_acc_pp:.2f}%",
            )
            if cfg.estop.early_stop(100.0 - val_acc_pp):
                break

    print("Training complete.")
    model.eval()


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
