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
        fmt: "FullModelType",
        labels: str | pd.DataFrame,
        training: bool,
        to_net_ids: dict["LabelId", "NetId"],
        transform: Optional["Compose"] = None,
    ) -> None:
        usecols = ["filename", "label_id"]

        self.training = training
        if isinstance(labels, str):
            self.labels = pd.read_csv(labels, usecols=usecols)
        elif isinstance(labels, pd.DataFrame):
            self.labels = labels[usecols].copy()
        else:
            raise TypeError("'labels' must be either a path (str) or a DataFrame.")

        self.labels["net_id"] = self.labels["label_id"].map(to_net_ids)
        if training:
            mask = self.labels["net_id"].isnull()
            assert not mask.any(), f"Some null net_ids were found:\n{self.labels[mask]}"
        self.labels = self.labels.drop("label_id", axis=1)

        self.fmt = fmt
        self.transform = transform
        self.fmt_fd = os.path.join(absp(CROPS_MAIN_FD_RP), self.fmt)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple["Tensor", "np_int64"]:
        image = Image.open(
            os.path.join(self.fmt_fd, self.labels["filename"].iat[idx])
        )
        label = self.labels["net_id"].iat[idx]

        if self.transform is not None:
            image = self.transform(image)

        return image, label
