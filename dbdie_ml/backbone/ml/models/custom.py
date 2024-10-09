"""Custom models code.
Mainly default IEModels for each (implemented) predictable.
"""

from dbdie_classes.options import MODEL_TYPE as MT
from dbdie_classes.options.FMT import to_fmt
from torch.nn import Conv2d, Flatten, Linear, MaxPool2d, ReLU, Sequential

from backbone.classes.metadata import SavedModelMetadata
from backbone.ml.models import IEModel


def max_pool_round(int_tuple: tuple[int, int]) -> tuple[int, int]:
    return (
        round(0.5 * int_tuple[0] - 0.01),
        round(0.5 * int_tuple[1] - 0.01),
    )


class AddonsModel(IEModel):
    """Recommended custom IEModel with an addon-based non-trained model."""

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.ADDONS, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x20x20  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x10x10
            Flatten(),  # Output size: 64*10*10
            Linear(64 * img_size_4[0] * img_size_4[1], 128),
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

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.CHARACTER, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)
        img_size_8 = max_pool_round(img_size_4)

        model = Sequential(
            Conv2d(3, 32, (5, 5), padding=2),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x(W/2)x(H/2)  (rounded down)
            Conv2d(32, 64, (5, 5), padding=2),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x(W/4)x(H/4)
            Conv2d(64, 128, (5, 5), padding=2),  # 128 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 128x(W/8)x(H/8)
            Flatten(),  # Output size: 128*(W/8)*(H/8)
            Linear(128 * img_size_8[0] * img_size_8[1], 256),
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

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.ITEM, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x21x21  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x10x10
            Flatten(),  # Output size: 64*10*10
            Linear(64 * img_size_4[0] * img_size_4[1], 128),
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

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.OFFERING, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x24x26  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x12x13
            Flatten(),  # Output size: 64*12*13
            Linear(64 * img_size_4[0] * img_size_4[1], 128),
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

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.PERKS, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x27x28  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x13x14
            Flatten(),  # Output size: 64*13*14
            Linear(64 * img_size_4[0] * img_size_4[1], 128),
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

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.PRESTIGE, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)
        img_size_8 = max_pool_round(img_size_4)

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
            Linear(128 * img_size_8[0] * img_size_8[1], 128),
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

    def __init__(
        self,
        id: int,
        ifk: bool,
        total_classes: int,
        cps_name: str,
    ) -> None:
        metadata = SavedModelMetadata.load(
            fmt=to_fmt(MT.STATUS, ifk),
            extr_name=None,
            model_id=id,
            total_classes=total_classes,
            cps_name=cps_name,
        )

        img_size_2 = max_pool_round(metadata.img_size)
        img_size_4 = max_pool_round(img_size_2)

        model = Sequential(
            Conv2d(3, 32, (3, 3), padding=1),  # 32 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 32x15x20  (rounded down)
            Conv2d(32, 64, (3, 3), padding=1),  # 64 filters
            ReLU(),
            MaxPool2d((2, 2)),  # Output size: 64x7x10
            Flatten(),  # Output size: 64*7*10
            Linear(64 * img_size_4[0] * img_size_4[1], 128),
            ReLU(),
            Linear(128, total_classes),
        )

        super().__init__(
            metadata=metadata,
            model=model,
            total_classes=total_classes,
        )
