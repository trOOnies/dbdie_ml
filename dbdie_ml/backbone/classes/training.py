"""Training light classes."""

from dataclasses import asdict, dataclass


@dataclass
class TrainingParams:
    """Helper dataclass for training params."""
    epochs: int
    batch_size: int
    adam_lr: float

    def dict(self) -> dict[str, str]:
        return {k: str(v) for k, v in asdict(self).items()}
