from __future__ import annotations

import argparse
from pathlib import Path

from .config import DataConfig, ModelConfig, TrainingConfig
from .data import SimulationDataset
from .training import train


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the rnet DeepSphere r regressor on CPU")
    parser.add_argument("--manifest", type=Path, required=True, help="RGridSimulator active_dataset.json")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1.0e-3)
    parser.add_argument("--cpu-threads", type=int, default=24)
    parser.add_argument("--save-every-epochs", type=int, default=5)
    parser.add_argument(
        "--resume",
        default=None,
        help="Use 'latest', an optimizer-iteration checkpoint number, or a checkpoint path",
    )
    parser.add_argument("--inspect-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    data_config = DataConfig(manifest=args.manifest)
    dataset = SimulationDataset(data_config)
    print(dataset.summary())
    if args.inspect_only:
        return

    model_config = ModelConfig(nside=dataset.nside, n_channels=len(dataset.frequencies))
    training_config = TrainingConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        cpu_threads=args.cpu_threads,
        save_every_epochs=args.save_every_epochs,
    )
    train(
        data_config=data_config,
        model_config=model_config,
        training_config=training_config,
        run_dir=args.run_dir,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
