"""Custom models code.
Mainly default IEModels for each (implemented) predictable.
"""

from dbdie_classes.options import MODEL_TYPE as MT
from dbdie_classes.options.FMT import to_fmt
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

from backbone.classes.metadata import SavedModelMetadata
from backbone.ml.models import IEModel


class AddonsModel(IEModel):
    """Recommended custom IEModel with an addon-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-addons",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.ADDONS, ifk),
        )

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
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )


class CharacterModel(IEModel):
    """Recommended custom IEModel with a character-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-character",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.CHARACTER, ifk),
        )

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
            Linear(256, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )


class ItemModel(IEModel):
    """Recommended custom IEModel with an item-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-item",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.ITEM, ifk),
        )

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
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )


class OfferingModel(IEModel):
    """Recommended custom IEModel with an offering-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-offering",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.OFFERING, ifk),
        )

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
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )


class PerkModel(IEModel):
    """Recommended custom IEModel with a perk-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-perks",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.PERKS, ifk),
        )

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
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )


# class PointsModel(IEModel): pass


class PrestigeModel(IEModel):
    """Recommended custom IEModel with a prestige-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-prestige",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.PRESTIGE, ifk),
        )

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
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )


class StatusModel(IEModel):
    """Recommended custom IEModel with a status-based non-trained model."""

    def __init__(self, id: int, ifk: bool, total_classes: int) -> None:
        metadata = SavedModelMetadata.load(
            extr_name="custom-status",
            model_id=id,
            total_classes=total_classes,
            fmt=to_fmt(MT.STATUS, ifk),
        )

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
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )
