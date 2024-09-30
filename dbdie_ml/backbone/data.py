"""Data related code."""

from dbdie_classes.paths import absp, CROPS_MAIN_FD_RP
import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from dbdie_classes.base import FullModelType, LabelId, NetId
    from numpy import int64 as np_int64
    from torch import Tensor
    from torchvision.transforms import Compose


class DatasetClass(Dataset):
    """DBDIE implementation of torch's `Dataset`."""

    def __init__(
        self,
        full_model_type: "FullModelType",
        labels: str | pd.DataFrame,
        to_net_ids: dict["LabelId", "NetId"],
        transform: Optional["Compose"] = None,
    ) -> None:
        usecols = ["filename", "label_id"]

        if isinstance(labels, str):
            self.labels = pd.read_csv(labels, usecols=usecols)
        elif isinstance(labels, pd.DataFrame):
            self.labels = labels[usecols].copy()
        else:
            raise TypeError("'labels' must be either a path (str) or a DataFrame.")

        self.labels["net_id"] = self.labels["label_id"].map(to_net_ids)
        self.labels = self.labels.drop("label_id", axis=1)

        self.full_model_type = full_model_type
        self.transform = transform
        self.fmt_fd = os.path.join(absp(CROPS_MAIN_FD_RP), self.full_model_type)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple["Tensor", "np_int64"]:
        image = Image.open(
            os.path.join(self.fmt_fd, self.labels["filename"].iat[idx])
        )
        label = self.labels["net_id"].iat[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
