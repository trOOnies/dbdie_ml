"""Custom models code.
Mainly default IEModels for each (implemented) predictable.
"""

import os
import yaml

from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

from dbdie_ml.data import get_total_classes
from dbdie_ml.paths import recursive_dirname
from dbdie_ml.ml.models import IEModel

CONFIGS_FD = os.path.join(recursive_dirname(__file__, 3), "configs")


class PerkModel(IEModel):
    """Recommended custom IEModel with a perk-based non-trained model"""

    def __init__(self, is_for_killer: bool) -> None:
        fmt = f"perks__{'killer' if is_for_killer else 'surv'}"

        path = os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
        with open(path) as f:
            metadata = yaml.safe_load(f)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x27x28  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x13x14
            Flatten(),  # Output size: 64*13*14
            Linear(64 * 13 * 14, 128),
            ReLU(),
            Linear(128, get_total_classes(fmt)),
        )

        super().__init__(metadata=metadata, model=model)


class CharacterModel(IEModel):
    """Recommended custom IEModel with a character-based non-trained model"""

    def __init__(self, is_for_killer: bool) -> None:
        fmt = f"character__{'killer' if is_for_killer else 'surv'}"

        path = os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
        with open(path) as f:
            metadata = yaml.safe_load(f)

        model = Sequential(
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
            Linear(256, get_total_classes(fmt)),
        )

        super().__init__(metadata=metadata, model=model)


class StatusModel(IEModel):
    """Recommended custom IEModel with a status-based non-trained model"""

    def __init__(self, is_for_killer: bool) -> None:
        fmt = f"status__{'killer' if is_for_killer else 'surv'}"

        path = os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
        with open(path) as f:
            metadata = yaml.safe_load(f)

        model = Sequential(
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
        )

        super().__init__(metadata=metadata, model=model)
