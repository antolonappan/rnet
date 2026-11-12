from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple


@dataclass(frozen=True)
class BNNModelConfig:
    """DeepSphere feature extractor with a variational Bayesian head."""

    nside: int = 128
    n_channels: int = 2
    conv_blocks: Sequence[Tuple[int, int]] = field(
        default_factory=lambda: ((5, 8), (5, 16), (5, 32), (3, 64))
    )
    pool_features: Sequence[int] = field(default_factory=lambda: (16, 32, 64))
    bayesian_dense_features: Sequence[int] = field(default_factory=lambda: (128, 64, 32))
    dropout: float = 0.1
    l2_regularization: float = 1.0e-6
    n_neighbors: int = 8
    pooling: Sequence[str] = field(default_factory=lambda: ("mean", "var", "max"))
    prior_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.nside <= 0 or self.n_channels <= 0:
            raise ValueError("nside and n_channels must be > 0")
        if len(self.conv_blocks) != len(self.pool_features) + 1:
            raise ValueError("conv_blocks must contain one more block than pool_features")
        if self.nside % (2 ** len(self.pool_features)) != 0:
            raise ValueError("nside must be divisible by the total pooling factor")
        if self.n_neighbors not in {8, 20, 40, 60}:
            raise ValueError("n_neighbors must be one of 8, 20, 40, or 60")
        if not self.bayesian_dense_features:
            raise ValueError("bayesian_dense_features must not be empty")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.prior_scale <= 0:
            raise ValueError("prior_scale must be > 0")
        if set(self.pooling) - {"mean", "var", "max"} or not self.pooling:
            raise ValueError("pooling must contain mean, var, and/or max")


@dataclass(frozen=True)
class BNNTrainingConfig:
    batch_size: int = 32
    epochs: int = 150
    learning_rate: float = 1.0e-4
    save_every_epochs: int = 5
    early_stopping_patience: int = 20
    reduce_lr_patience: int = 7
    cpu_threads: int = 24
    seed: int = 2027
    mc_samples: int = 32
    kl_weight: float | None = None
    beta_nll: float = 0.0
    signed_null_penalty: float = 0.0
    zero_samples_per_batch: int = 0
    device: str = "cpu"
    initialize_from_cnn: str | None = None
    freeze_backbone: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "epochs", "save_every_epochs", "cpu_threads", "mc_samples"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.freeze_backbone and not self.initialize_from_cnn:
            raise ValueError("freeze_backbone requires initialize_from_cnn")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.kl_weight is not None and self.kl_weight <= 0:
            raise ValueError("kl_weight must be > 0 when provided")
        if self.beta_nll < 0:
            raise ValueError("beta_nll must be >= 0")
        if self.signed_null_penalty < 0:
            raise ValueError("signed_null_penalty must be >= 0")
        if self.signed_null_penalty > 0 and not 2 <= self.zero_samples_per_batch < self.batch_size:
            raise ValueError("Signed training needs 2 <= zero_samples_per_batch < batch_size")
        if self.signed_null_penalty == 0 and self.zero_samples_per_batch != 0:
            raise ValueError("zero_samples_per_batch requires signed_null_penalty > 0")
        if self.device not in {"cpu", "gpu"}:
            raise ValueError("device must be 'cpu' or 'gpu'")


def config_to_dict(config: object) -> Dict[str, Any]:
    values = asdict(config)
    for key, value in list(values.items()):
        if isinstance(value, Path):
            values[key] = str(value)
    return values
