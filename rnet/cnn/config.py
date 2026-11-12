from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


@dataclass(frozen=True)
class DataConfig:
    """Configuration for reading an RGridSimulator dataset manifest."""

    manifest: Path
    validation_fraction: float = 0.1
    test_fraction: float = 0.1
    split_seed: int = 2027
    normalization_samples: int = 512
    shuffle_buffer: int = 256
    cache_in_memory: bool = False
    target_scale: float = 1.0e-2
    frequencies: Tuple[int, ...] | None = None
    variant: str = "nofg"

    def __post_init__(self) -> None:
        object.__setattr__(self, "manifest", Path(self.manifest).expanduser().resolve())
        if not self.manifest.is_file():
            raise FileNotFoundError(f"Dataset manifest not found: {self.manifest}")
        if self.validation_fraction < 0 or self.test_fraction < 0:
            raise ValueError("validation_fraction and test_fraction must be >= 0")
        if self.validation_fraction + self.test_fraction >= 1:
            raise ValueError("validation_fraction + test_fraction must be < 1")
        if self.normalization_samples <= 0:
            raise ValueError("normalization_samples must be > 0")
        if self.shuffle_buffer <= 0:
            raise ValueError("shuffle_buffer must be > 0")
        if self.target_scale <= 0:
            raise ValueError("target_scale must be > 0")
        if self.frequencies is not None:
            values = tuple(int(value) for value in self.frequencies)
            if not values or len(values) != len(set(values)):
                raise ValueError("frequencies must be a nonempty set of unique bands")
            object.__setattr__(self, "frequencies", values)
        if self.variant not in {"nofg", "fg_d11s6"}:
            raise ValueError("variant must be 'nofg' or 'fg_d11s6'")


@dataclass(frozen=True)
class ModelConfig:
    """A CPU-conscious DeepSphere regression architecture."""

    nside: int = 128
    n_channels: int = 2
    conv_blocks: Sequence[Tuple[int, int]] = field(
        default_factory=lambda: ((5, 8), (5, 16), (5, 32), (3, 64))
    )
    pool_features: Sequence[int] = field(default_factory=lambda: (16, 32, 64))
    dense_features: int | Sequence[int] = 64
    dropout: float = 0.1
    l2_regularization: float = 1.0e-6
    n_neighbors: int = 8
    pooling: Sequence[str] = field(default_factory=lambda: ("mean", "var", "max"))

    def __post_init__(self) -> None:
        if self.nside <= 0 or self.n_channels <= 0:
            raise ValueError("nside and n_channels must be > 0")
        if len(self.conv_blocks) != len(self.pool_features) + 1:
            raise ValueError("conv_blocks must contain one more block than pool_features")
        if self.nside % (2 ** len(self.pool_features)) != 0:
            raise ValueError("nside must be divisible by the total pooling factor")
        if self.n_neighbors not in {8, 20, 40, 60}:
            raise ValueError("n_neighbors must be one of 8, 20, 40, or 60")
        widths = (self.dense_features,) if isinstance(self.dense_features, int) else tuple(self.dense_features)
        if not widths or any(not isinstance(width, int) or width <= 0 for width in widths):
            raise ValueError("dense_features must contain positive integer widths")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        allowed = {"mean", "var", "max"}
        if not self.pooling or any(value not in allowed for value in self.pooling):
            raise ValueError("pooling must contain only mean, var, and max")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 8
    epochs: int = 100
    learning_rate: float = 1.0e-3
    save_every_epochs: int = 5
    early_stopping_patience: int = 15
    reduce_lr_patience: int = 5
    cpu_threads: int = 24
    seed: int = 2027
    beta_nll: float = 0.5
    signed_null_penalty: float = 0.0
    zero_samples_per_batch: int = 0
    device: str = "cpu"

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "save_every_epochs", "cpu_threads"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")
        if self.beta_nll < 0:
            raise ValueError("beta_nll must be >= 0")
        if self.signed_null_penalty < 0:
            raise ValueError("signed_null_penalty must be >= 0")
        if self.signed_null_penalty > 0 and not 2 <= self.zero_samples_per_batch < self.batch_size:
            raise ValueError("Signed training needs 2 <= zero_samples_per_batch < batch_size")
        if self.signed_null_penalty == 0 and self.zero_samples_per_batch != 0:
            raise ValueError("zero_samples_per_batch requires signed_null_penalty > 0")


def config_to_dict(config: object) -> Dict[str, Any]:
    data = asdict(config)
    for key, value in list(data.items()):
        if isinstance(value, Path):
            data[key] = str(value)
    return data
