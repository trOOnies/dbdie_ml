from typing import Optional
from torch.nn import (
    Sequential,
    Conv2d,
    ReLU,
    MaxPool2d,
    Flatten,
    Linear
)
from dbdie_ml.models import IEModel
from dbdie_ml.data import get_total_classes


class PerkModel(IEModel):
    """Recommended custom `IEModel` with a perk-based non-trained `model`"""
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(
            name=name,
            model=Sequential(
                Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 32x27x28  (rounded down)
                Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
                ReLU(),
                MaxPool2d((2,2)),  # Output size: 64x13x14
                Flatten(),  # Output size: 64*13*14
                Linear(64*13*14, 128),
                ReLU(),
                Linear(128, get_total_classes()),
            ),
            model_type="perks",
            version="7.5.0",
            norm_means=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225]
        )
