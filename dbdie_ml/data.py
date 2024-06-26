import os
import pandas as pd
from typing import TYPE_CHECKING, Optional
from PIL import Image
from torch.utils.data import Dataset
if TYPE_CHECKING:
    from numpy import int64 as np_int64
    from torch import Tensor
    from torchvision.transforms import Compose


def get_total_classes(selected_fd: str) -> int:
    class_df = pd.read_csv(
        os.path.join(os.environ["LABELS_FD"], selected_fd, "label_ref.csv")
    )
    assert (class_df.label_id == class_df.index).all()
    return class_df.shape[0]


class DatasetClass(Dataset):
    def __init__(
        self,
        csv_path: str,
        transform: Optional["Compose"] = None
    ) -> None:
        self.labels = pd.read_csv(csv_path, usecols=["name", "label_id"])
        self.transform = transform

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> tuple["Tensor", "np_int64"]:
        image = Image.open(
            os.path.join(
                os.environ["CROPS_FD"],
                os.environ["SELECTED_FD"],
                self.labels.name.iat[idx]
            )
        )
        label = self.labels.label_id.iat[idx]

        if self.transform:
            image = self.transform(image)

        return image, label
