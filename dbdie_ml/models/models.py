from dotenv import load_dotenv
load_dotenv("../.env", override=True)

import os
import yaml
import numpy as np
import pandas as pd
from typing import TYPE_CHECKING, Optional, List, Tuple, Dict, Any
from torch import no_grad, save
from torch import max as torch_max
from torch.cuda import mem_get_info
from torch.cuda import device as get_device
from torch.cuda import is_available as cuda_is_available
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from torch.nn import CrossEntropyLoss
from dbdie_ml.data import DatasetClass, get_total_classes
if TYPE_CHECKING:
    from torch.nn import Sequential
    from torch.optim import Optimizer
    from torch.nn.modules.loss import _Loss
    from dbdie_ml.classes import ModelType


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss: float) -> bool:
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif (self.min_validation_loss + self.min_delta) < validation_loss:
            self.counter += 1
            return (self.counter >= self.patience)
        return False


class IEModel:
    """ML model for information extraction

    Inputs:
        model: A `torch.nn.Sequential`.
        model_type: A `ModelType` (string).
        version: A string that matches the DBD version,
        preferably the minimum version used for the training process.
        norm_means: A list of 3 floats for the torch `Compose`.
        norm_std: Idem norm_means.
        name: An optional string.

    Usage:
        >>> model = IEModel(Sequential(...), "perks", "7.6.0", ...)
        >>> model.init_model()  # this uses all standard models
        >>> model.get_summary()
        >>> model.train(...)
        >>> model.save_model("/path/to/model.pt")
        >>> preds = model.predict_batch("/path/to/dataset.csv")
        >>> names = model.convert_names(preds)
        >>> model.save_preds(preds, "/path/to/preds.txt")
        >>> probas = model.predict_batch("/path/to/dataset.csv", probas=True)
    """

    def __init__(
        self,
        model: "Sequential",
        model_type: "ModelType",
        version: str,
        norm_means: List[float],
        norm_std: List[float],
        name: Optional[str] = None
    ) -> None:
        self.name = name
        self._model = model
        self.model_type = model_type
        self.version = version
        self._norm_means = norm_means
        self._norm_std = norm_std
        self.total_classes: Optional[int] = None
        self._device = None
        self._transform: Optional[transforms.Compose] = None
        self._optimizer: Optional[Optimizer] = None
        self._criterion: Optional[_Loss] = None
        self._estop: Optional[EarlyStopper] = None
        self._cfg: Dict[str, Any] = None
        self.label_ref: Optional[Dict[int, str]] = None
        self.model_is_trained = False

    def __repr__(self) -> str:
        vals = {
            "type": self.model_type,
            "version": self.version,
            "classes": self.total_classes,
            "trained": self.model_is_trained,
        }
        vals = ', '.join([f"{k}='{v}'" for k, v in vals.items()])
        if self.name is not None:
            vals = f"'{self.name}', " + vals
        return f"IEModel({vals})"

    @property
    def model_is_init(self) -> bool:
        return self._optimizer is not None

    def _get_transform(self) -> transforms.Compose:
        """Define any image transformations here"""
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self._norm_means, std=self._norm_std)
        ])

    def init_model(self) -> None:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

        assert cuda_is_available()
        self._device = get_device("cuda")

        config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
        with open(config_path) as f:
            self._cfg = yaml.safe_load(f)

        self.total_classes = get_total_classes()

        # Input image size is 55x56 and the number of channels is 3 (RGB)
        self._optimizer = Adam(self._model.parameters(), lr=self._cfg["adam_lr"])
        self._criterion = CrossEntropyLoss()
        self._estop = EarlyStopper(patience=3, min_delta=-0.01)

        self._model = self._model.cuda()

        self._transform = self._get_transform()

    def get_summary(self) -> None:
        assert self.model_is_init
        summary(self._model, (3, 55, 57), batch_size=self._cfg["batch_size"], device="cuda")

        print("MEMORY")
        print("- Free: {:,.2} GiB\n- Total: {:,.2} GiB".format(
            *[v / (2**30) for v in mem_get_info(self._device)])
        )
        print(64 * "-")

    def _load_process(self, train_ds_path: str, val_ds_path: str) -> Tuple[DataLoader, DataLoader]:
        print("Loading data...", end=" ")
        train_dataset = DatasetClass(train_ds_path, transform=self._transform)
        val_dataset = DatasetClass(val_ds_path, transform=self._transform)

        train_loader = DataLoader(train_dataset, batch_size=self._cfg["batch_size"], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self._cfg["batch_size"])

        print("Data loaded.")
        print("- Train datapoints:", len(train_dataset))
        print("- Val datapoints:", len(val_dataset))

        return train_loader, val_loader

    def _load_label_ref(self, path: str) -> None:
        self.label_ref = pd.read_csv(path, usecols=["label_id", "name"], dtype={"label_id": int, "name": str})
        assert self.label_ref.label_id.min() == 0
        assert self.label_ref.label_id.max() + 1 == self.label_ref.shape[0]
        assert self.label_ref.label_id.nunique() == self.label_ref.shape[0]
        self.label_ref = {row["label_id"]: row["name"] for _, row in self.label_ref.iterrows()}

    def _train_process(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        print("Training initialized...")
        epochs_clen = len(str(self._cfg["epochs"]))
        for epoch in range(1, self._cfg["epochs"] + 1):
            self._model.train()
            for images, labels in train_loader:
                images = images.cuda()
                labels = labels.cuda()

                outputs = self._model(images)
                loss = self._criterion(outputs, labels)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()

            self._model.eval()
            with no_grad():
                correct = 0
                total = 0
                for images, labels in val_loader:
                    images = images.cuda()
                    labels = labels.cuda()

                    outputs = self._model(images)
                    _, predicted = torch_max(outputs.data, 1)

                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                val_acc_pp = 100.0 * correct / total
                print(
                    f"- Epoch [{epoch:>{epochs_clen}}/{self._cfg['epochs']}]",
                    f"Loss: {loss.item():.4f}",
                    f"Val Acc: {val_acc_pp:.2f}%"
                )
                if self._estop.early_stop(100.0 - val_acc_pp):
                    break
        print("Training complete.")
        self.model_is_trained = True
        self._model.eval()

    def train(self, label_ref_path: str, train_dataset_path: str, val_dataset_path: str) -> None:
        """Trains the `IEModel`"""
        assert self.model_is_init and not self.model_is_trained
        self._load_label_ref(label_ref_path)
        train_loader, val_loader = self._load_process(train_dataset_path, val_dataset_path)
        self._train_process(train_loader, val_loader)

    def save_model(self, dst: str) -> None:
        print("Saving model...", end=" ")
        assert self.model_is_trained
        assert dst.endswith(".pt")
        save(self._model, dst)
        print("Model saved.")

    def _predict_process(
        self,
        dataset: DatasetClass,
        loader: DataLoader
    ) -> np.ndarray:
        all_preds = np.zeros(len(dataset), dtype=np.ushort)
        i = 0
        with no_grad():
            for images, labels in loader:
                labels_len = labels.size()[0]
                images = images.cuda()

                outputs = self._model(images)
                _, predicted = torch_max(outputs.data, 1)
                all_preds[i:i+labels_len] = predicted.cpu().numpy()
                i += labels_len
        return all_preds

    def _predict_probas_process(
        self,
        dataset: DatasetClass,
        loader: DataLoader
    ) -> np.ndarray:
        all_preds = np.zeros((len(dataset), self.total_classes), dtype=int)
        i = 0
        with no_grad():
            for images, labels in loader:
                labels_len = labels.size()[0]
                images = images.cuda()

                outputs = self._model(images)
                all_preds[i:i+labels_len, :] = outputs.cpu().numpy()
                i += labels_len
        return all_preds

    def predict_batch(self, dataset_path: str, probas: bool = False) -> np.ndarray:
        """Returns: preds or probas, indices_with_errors"""
        assert self.model_is_trained

        print("Predictions for:", dataset_path)
        dataset = DatasetClass(dataset_path, transform=self._transform)
        loader = DataLoader(dataset, batch_size=self._cfg["batch_size"])

        if probas:
            return self._predict_probas_process(dataset, loader)
        else:
            return self._predict_process(dataset, loader)

    def convert_names(self, labels: np.ndarray) -> List[str]:
        assert isinstance(labels[0], (np.ushort, int))
        assert self.model_is_trained
        return [self.label_ref[lbl] for lbl in labels]

    @staticmethod
    def save_preds(preds: np.ndarray, dst: str) -> None:
        print("Saving preds...", end=" ")
        assert dst.endswith(".txt")
        np.savetxt(dst, preds, fmt="%d")
        print("Preds saved.")
