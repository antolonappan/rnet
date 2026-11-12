from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from ..cnn.config import DataConfig
from ..cnn.data import SimulationDataset, Standardizer
from ..cnn.training import configure_cpu
from .config import BNNModelConfig
from .model import mc_prediction_to_r


def load_run(run_dir: Path | str, checkpoint: str = "final", cpu_threads: int = 24):
    configure_cpu(cpu_threads)
    import tensorflow as tf

    from .model import build_model

    run_path = Path(run_dir).expanduser().resolve()
    saved = json.loads((run_path / "config.json").read_text(encoding="utf-8"))
    dataset = SimulationDataset(DataConfig(**saved["data"]))
    model = build_model(BNNModelConfig(**saved["model"]), dataset.pixel_indices, float(saved["kl_weight_used"]))
    if str(checkpoint).lower() == "final":
        model.load_weights(run_path / "final.weights.h5")
    else:
        restore = tf.train.Checkpoint(model=model)
        path = tf.train.latest_checkpoint(str(run_path / "checkpoints"))
        if path is None:
            raise FileNotFoundError(f"No checkpoint found in {run_path / 'checkpoints'}")
        restore.restore(path).expect_partial()
    return model, dataset, Standardizer.load(run_path / "normalization.npz")


def _calibration(run_dir: Path | str) -> float:
    path = Path(run_dir).expanduser().resolve() / "calibration.json"
    if not path.is_file():
        return 1.0
    return float(json.loads(path.read_text(encoding="utf-8")).get("scale", 1.0))


def predict_split(run_dir: Path | str, split: str = "test", mc_samples: int = 64, batch_size: int = 32, cpu_threads: int = 24) -> Dict[str, np.ndarray]:
    model, dataset, standardizer = load_run(run_dir, cpu_threads=cpu_threads)
    from .training import _mc_predict

    indices = dataset.splits[split]
    raw = _mc_predict(model, dataset, indices, standardizer, batch_size, mc_samples)
    result = mc_prediction_to_r(raw, dataset.config.target_scale, _calibration(run_dir))
    result["r_true"] = np.asarray([float(dataset.entries[int(i)]["r"]) for i in indices])
    result["sim_idx"] = np.asarray([int(dataset.entries[int(i)]["sim_idx"]) for i in indices])
    return result


def predict_manifest(run_dir: Path | str, manifest: Path | str, mc_samples: int = 64, batch_size: int = 32, cpu_threads: int = 24) -> Dict[str, np.ndarray]:
    model, training_dataset, standardizer = load_run(run_dir, cpu_threads=cpu_threads)
    external = SimulationDataset(DataConfig(manifest=manifest, target_scale=training_dataset.config.target_scale, validation_fraction=0.0, test_fraction=0.0, normalization_samples=1))
    if external.nside != training_dataset.nside or external.frequencies != training_dataset.frequencies or not np.array_equal(external.pixel_indices, training_dataset.pixel_indices):
        raise ValueError("Benchmark manifest is incompatible with the BNN input geometry")
    from .training import _mc_predict

    indices = np.arange(len(external.entries), dtype=np.int64)
    raw = _mc_predict(model, external, indices, standardizer, batch_size, mc_samples)
    result = mc_prediction_to_r(raw, external.config.target_scale, _calibration(run_dir))
    result["r_true"] = np.asarray([float(e["r"]) for e in external.entries])
    result["sim_idx"] = np.asarray([int(e["sim_idx"]) for e in external.entries])
    return result
