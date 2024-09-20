"""Custom models code.
Mainly default IEModels for each (implemented) predictable.
"""

import os
import yaml
from typing import TYPE_CHECKING

from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

from dbdie_ml.data import get_total_classes
from dbdie_ml.options import MODEL_TYPES as MT
from dbdie_ml.options.PLAYER_TYPE import ifk_to_pt
from dbdie_ml.paths import recursive_dirname
from dbdie_ml.ml.models import IEModel

if TYPE_CHECKING:
    from dbdie_ml.classes.base import FullModelType, ModelType

CONFIGS_FD = os.path.join(recursive_dirname(__file__, 3), "configs")


def get_params(mt: "ModelType", is_for_killer: bool) -> tuple["FullModelType", dict]:
    fmt = f"{mt}__{ifk_to_pt(is_for_killer)}"

    path = os.path.join(CONFIGS_FD, f"custom_models/{fmt}/metadata.yaml")
    with open(path) as f:
        metadata = yaml.safe_load(f)

    return fmt, metadata


class AddonsModel(IEModel):
    """Recommended custom IEModel with an addon-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        fmt, metadata = get_params(MT.ADDONS, is_for_killer)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x20x20  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x10x10
            Flatten(),  # Output size: 64*10*10
            Linear(64 * 10 * 10, 128),
            ReLU(),
            Linear(128, get_total_classes(fmt)),
        )

        super().__init__(metadata=metadata, model=model)


class CharacterModel(IEModel):
    """Recommended custom IEModel with a character-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        fmt, metadata = get_params(MT.CHARACTER, is_for_killer)

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


class ItemModel(IEModel):
    """Recommended custom IEModel with an item-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        fmt, metadata = get_params(MT.ITEM, is_for_killer)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x21x21  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x10x10
            Flatten(),  # Output size: 64*10*10
            Linear(64 * 10 * 10, 128),
            ReLU(),
            Linear(128, get_total_classes(fmt)),
        )

        super().__init__(metadata=metadata, model=model)


class OfferingModel(IEModel):
    """Recommended custom IEModel with an offering-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        fmt, metadata = get_params(MT.OFFERING, is_for_killer)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x24x26  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x12x13
            Flatten(),  # Output size: 64*12*13
            Linear(64 * 12 * 13, 128),
            ReLU(),
            Linear(128, get_total_classes(fmt)),
        )

        super().__init__(metadata=metadata, model=model)


class PerkModel(IEModel):
    """Recommended custom IEModel with a perk-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        fmt, metadata = get_params(MT.PERKS, is_for_killer)

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


# class PointsModel(IEModel): pass


class PrestigeModel(IEModel):
    """Recommended custom IEModel with a prestige-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        _, metadata = get_params(MT.PRESTIGE, is_for_killer)

        model = Sequential(
            Conv2d(3, 32, (5, 5), padding=2),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x58x58  (rounded down)
            Conv2d(32, 64, (5, 5), padding=2),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x29x29
            Conv2d(32, 128, (5, 5), padding=2),  # 128 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 128x14x14
            Flatten(),  # Output size: 128*14*14
            Linear(128 * 14 * 14, 128),
            ReLU(),
            Linear(128, 101),
        )

        super().__init__(metadata=metadata, model=model)


class StatusModel(IEModel):
    """Recommended custom IEModel with a status-based non-trained model."""

    def __init__(self, is_for_killer: bool) -> None:
        fmt, metadata = get_params(MT.STATUS, is_for_killer)

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
            Linear(128, get_total_classes(fmt)),
        )

        super().__init__(metadata=metadata, model=model)
