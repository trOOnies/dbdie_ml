import os
import pandas as pd
from typing import TYPE_CHECKING, Optional
from PIL import Image
from torch.utils.data import Dataset
from dbdie_ml.paths import absp

if TYPE_CHECKING:
    from numpy import int64 as np_int64
    from torch import Tensor
    from torchvision.transforms import Compose
    from dbdie_ml.classes import FullModelType


def get_total_classes(selected_fd: str) -> int:
    """Calculate total classes from the corresponding `label_ref.csv`"""
    class_df = pd.read_csv(
        absp(
            os.path.join(
                "data/labels/labels",
                selected_fd,
                "label_ref.csv",
            )
        )
    )
    assert (class_df.label_id == class_df.index).all()
    return class_df.shape[0]


class DatasetClass(Dataset):
    """DBDIE implementation of torch's `Dataset`"""

    def __init__(
        self,
        full_model_type: "FullModelType",
        csv_path: str,
        transform: Optional["Compose"] = None,
    ) -> None:
        self.full_model_type = full_model_type
        self.labels = pd.read_csv(csv_path, usecols=["name", "label_id"])
        self.transform = transform

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple["Tensor", "np_int64"]:
        image = Image.open(
            absp(
                os.path.join(
                    "data/crops",
                    self.full_model_type,
                    self.labels.name.iat[idx],
                )
            )
        )
        label = self.labels.label_id.iat[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
