"""Data related code."""

from dbdie_classes.paths import absp, CROPS_MAIN_FD_RP
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType
    from numpy import int64 as np_int64
    from torch import Tensor
    from torchvision.transforms import Compose


class DatasetClass(Dataset):
    """DBDIE implementation of torch's `Dataset`."""

    def __init__(
        self,
        full_model_type: "FullModelType",
        labels: str | pd.DataFrame,
        transform: Optional["Compose"] = None,
    ) -> None:
        if isinstance(labels, str):
            self.labels = pd.read_csv(labels, usecols=["name", "label_id"])
        elif isinstance(labels, pd.DataFrame):
            self.labels = labels[["name", "label_id"]].copy()
        else:
            raise TypeError("'labels' must be either a path (str) or a DataFrame.")

        self.full_model_type = full_model_type
        self.transform = transform

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple["Tensor", "np_int64"]:
        image = Image.open(
            absp(
                os.path.join(
                    CROPS_MAIN_FD_RP,
                    self.full_model_type,
                    self.labels.name.iat[idx],
                )
            )
        )
        label = self.labels.label_id.iat[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
