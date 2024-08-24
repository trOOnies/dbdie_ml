from typing import Optional

from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

from dbdie_ml.classes.version import DBDVersionRange
from dbdie_ml.data import get_total_classes
from dbdie_ml.models import IEModel


class PerkModel(IEModel):
    """Recommended custom `IEModel` with a perk-based non-trained `model`"""

    def __init__(self, is_for_killer: bool, name: Optional[str] = None) -> None:
        super().__init__(
            name=name,
            model=Sequential(
                Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 32x27x28  (rounded down)
                Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 64x13x14
                Flatten(),  # Output size: 64*13*14
                Linear(64 * 13 * 14, 128),
                ReLU(),
                Linear(
                    128,
                    get_total_classes(
                        f"perks__{'killer' if is_for_killer else 'surv'}"
                    ),
                ),
            ),
            model_type="perks",
            is_for_killer=is_for_killer,
            image_size=(55, 56),
            version_range=DBDVersionRange("7.5.0"),
            norm_means=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )


class CharacterModel(IEModel):
    """Recommended custom `IEModel` with a character-based non-trained `model`"""

    def __init__(self, is_for_killer: bool, name: Optional[str] = None) -> None:
        super().__init__(
            name=name,
            model=Sequential(
                Conv2d(3, 32, (5, 5), padding=2),  # 32 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 32x238x16  (rounded down)
                Conv2d(32, 64, (5, 5), padding=2),  # 64 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 64x119x8
                Conv2d(64, 128, (5, 5), padding=2),  # 128 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 128x59x4
                Flatten(),  # Output size: 128*59*4
                Linear(128 * 59 * 4, 256),
                ReLU(),
                Linear(
                    256,
                    get_total_classes(
                        f"character__{'killer' if is_for_killer else 'surv'}"
                    ),
                ),
            ),
            model_type="character",
            is_for_killer=is_for_killer,
            image_size=(476, 33),
            version_range=DBDVersionRange("7.5.0"),
            # Update normalization means and stds if necessary
            norm_means=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )


class StatusModel(IEModel):
    """Recommended custom `IEModel` with a status-based non-trained `model`"""

    def __init__(self, is_for_killer: bool, name: Optional[str] = None) -> None:
        super().__init__(
            name=name,
            model=Sequential(
                Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 32x15x20  (rounded down)
                Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
                ReLU(),
                MaxPool2d((2, 2)),  # Output size: 64x7x10
                Flatten(),  # Output size: 64*7*10
                Linear(64 * 7 * 10, 128),
                ReLU(),
                Linear(128, get_total_classes("status")),
            ),
            model_type="status",
            is_for_killer=is_for_killer,
            image_size=(31, 41),
            version_range=DBDVersionRange("7.5.0"),
            norm_means=[0.485, 0.456, 0.406],
            norm_std=[0.229, 0.224, 0.225],
        )
