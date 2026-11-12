from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .config import DataConfig, ModelConfig
from .data import SimulationDataset, Standardizer
from .training import configure_cpu


def _calibration_scale(run_path: Path, apply_calibration: bool = True) -> float:
    if not apply_calibration:
        return 1.0
    path = run_path / "calibration.json"
    if not path.is_file():
        return 1.0
    with open(path, "r", encoding="utf-8") as handle:
        return float(json.load(handle).get("scale", 1.0))


def load_run(
    run_dir: Path | str,
    checkpoint: str = "final",
    cpu_threads: int = 24,
):
    """Rebuild a saved run and restore final weights or a numbered checkpoint."""
    configure_cpu(cpu_threads)
    import tensorflow as tf

    from .model import build_model

    run_path = Path(run_dir).expanduser().resolve()
    with open(run_path / "config.json", "r", encoding="utf-8") as handle:
        saved_config = json.load(handle)
    dataset = SimulationDataset(DataConfig(**saved_config["data"]))
    model = build_model(ModelConfig(**saved_config["model"]), dataset.pixel_indices)

    if str(checkpoint).lower() == "final":
        model.load_weights(run_path / "final.weights.h5")
    else:
        checkpoint_dir = run_path / "checkpoints"
        restore = tf.train.Checkpoint(model=model)
        if str(checkpoint).lower() == "latest":
            restore_path = tf.train.latest_checkpoint(str(checkpoint_dir))
        else:
            candidate = Path(str(checkpoint)).expanduser()
            restore_path = (
                str(candidate)
                if candidate.exists() or Path(str(candidate) + ".index").exists()
                else str(checkpoint_dir / f"ckpt-{int(checkpoint)}")
            )
        if restore_path is None or not Path(str(restore_path) + ".index").exists():
            raise FileNotFoundError(f"Could not find checkpoint {checkpoint!r} in {checkpoint_dir}")
        restore.restore(restore_path).expect_partial()

    standardizer = Standardizer.load(run_path / "normalization.npz")
    return model, dataset, standardizer


def predict_split(
    run_dir: Path | str,
    split: str = "test",
    checkpoint: str = "final",
    batch_size: int = 8,
    cpu_threads: int = 24,
    apply_calibration: bool = True,
) -> Dict[str, np.ndarray]:
    """Predict one saved split and return truth, point estimates, and errors."""
    from .model import prediction_to_r

    model, dataset, standardizer = load_run(run_dir, checkpoint, cpu_threads)
    tf_dataset = dataset.make_tf_dataset(split, standardizer, batch_size, training=False)
    prediction = np.asarray(model.predict(tf_dataset, verbose=0))
    scale = _calibration_scale(Path(run_dir).expanduser().resolve(), apply_calibration)
    predicted_r, sigma_r = prediction_to_r(prediction, dataset.config.target_scale, scale)
    indices = dataset.splits[split]
    true_r = np.asarray([float(dataset.entries[int(i)]["r"]) for i in indices])
    sim_idx = np.asarray([int(dataset.entries[int(i)]["sim_idx"]) for i in indices])
    return {
        "r_true": true_r,
        "r_predicted": predicted_r,
        "sigma_r": sigma_r,
        "r_mean_scaled": prediction[:, 0],
        "r_log_variance": prediction[:, 1],
        "sim_idx": sim_idx,
    }


def predict_entry(
    run_dir: Path | str,
    entry_index: Optional[int] = None,
    r: Optional[float | str] = None,
    sim_idx: Optional[int] = None,
    checkpoint: str = "final",
    cpu_threads: int = 24,
) -> Dict[str, float | int]:
    """Predict one manifest entry, selected by index or by the pair ``(r, sim_idx)``."""
    from .model import prediction_to_r

    model, dataset, standardizer = load_run(run_dir, checkpoint, cpu_threads)
    if entry_index is None:
        if r is None or sim_idx is None:
            raise ValueError("Pass entry_index or both r and sim_idx")
        r_key = f"{float(r):.6f}"
        matches = [
            index
            for index, entry in enumerate(dataset.entries)
            if str(entry["r"]) == r_key and int(entry["sim_idx"]) == int(sim_idx)
        ]
        if len(matches) != 1:
            raise KeyError(f"Expected one entry for r={r_key}, sim_idx={sim_idx}; found {len(matches)}")
        entry_index = matches[0]

    index = int(entry_index)
    if index < 0 or index >= len(dataset.entries):
        raise IndexError(f"entry_index={index} is outside [0, {len(dataset.entries) - 1}]")
    maps, _ = dataset.load_sample(index, standardizer)
    raw = np.asarray(model.predict(maps[None, ...], verbose=0))
    scale = _calibration_scale(Path(run_dir).expanduser().resolve(), True)
    predicted_r, sigma_r = prediction_to_r(raw, dataset.config.target_scale, scale)
    entry = dataset.entries[index]
    return {
        "entry_index": index,
        "r_true": float(entry["r"]),
        "sim_idx": int(entry["sim_idx"]),
        "r_predicted": float(predicted_r[0]),
        "sigma_r": float(sigma_r[0]),
        "r_mean_scaled": float(raw[0, 0]),
        "r_log_variance": float(raw[0, 1]),
    }


def predict_entries(
    run_dir: Path | str,
    entry_indices: Sequence[int],
    checkpoint: str = "final",
    batch_size: int = 32,
    cpu_threads: int = 24,
) -> Dict[str, np.ndarray]:
    """Predict explicit manifest entries while loading the trained model only once."""
    from .model import prediction_to_r

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    model, dataset, standardizer = load_run(run_dir, checkpoint, cpu_threads)
    indices = np.asarray(entry_indices, dtype=np.int64)
    if indices.ndim != 1 or indices.size == 0:
        raise ValueError("entry_indices must be a non-empty one-dimensional sequence")
    if np.any(indices < 0) or np.any(indices >= len(dataset.entries)):
        raise IndexError(f"entry_indices must lie inside [0, {len(dataset.entries) - 1}]")

    raw_predictions = []
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        batch_maps = np.stack(
            [dataset.load_sample(int(index), standardizer)[0] for index in batch_indices]
        )
        raw_predictions.append(np.asarray(model.predict(batch_maps, verbose=0)))
    raw = np.concatenate(raw_predictions, axis=0)
    scale = _calibration_scale(Path(run_dir).expanduser().resolve(), True)
    predicted_r, sigma_r = prediction_to_r(raw, dataset.config.target_scale, scale)
    entries = [dataset.entries[int(index)] for index in indices]
    return {
        "entry_index": indices,
        "r_true": np.asarray([float(entry["r"]) for entry in entries]),
        "sim_idx": np.asarray([int(entry["sim_idx"]) for entry in entries]),
        "r_predicted": predicted_r,
        "sigma_r": sigma_r,
        "r_mean_scaled": raw[:, 0],
        "r_log_variance": raw[:, 1],
    }


def predict_manifest(
    run_dir: Path | str,
    manifest: Path | str,
    checkpoint: str = "final",
    batch_size: int = 32,
    cpu_threads: int = 24,
) -> Dict[str, np.ndarray]:
    """Predict every entry in a manifest using a saved CNN run.

    This is used for an independent benchmark manifest whose entries are not
    part of the training run's manifest.
    """
    from .model import prediction_to_r

    if int(batch_size) <= 0:
        raise ValueError("batch_size must be > 0")
    model, training_dataset, standardizer = load_run(run_dir, checkpoint, cpu_threads)
    external = SimulationDataset(
        DataConfig(
            manifest=manifest,
            target_scale=training_dataset.config.target_scale,
            validation_fraction=0.0,
            test_fraction=0.0,
            normalization_samples=1,
        )
    )
    if external.nside != training_dataset.nside or external.frequencies != training_dataset.frequencies:
        raise ValueError("Benchmark manifest is incompatible with the saved CNN input geometry")
    if not np.array_equal(external.pixel_indices, training_dataset.pixel_indices):
        raise ValueError("Benchmark manifest uses different selected pixels than the training run")

    indices = np.arange(len(external.entries), dtype=np.int64)
    raw_predictions = []
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        batch_maps = np.stack(
            [external.load_sample(int(index), standardizer)[0] for index in batch_indices]
        )
        raw_predictions.append(np.asarray(model.predict(batch_maps, verbose=0)))
    raw = np.concatenate(raw_predictions, axis=0)
    scale = _calibration_scale(Path(run_dir).expanduser().resolve(), True)
    predicted_r, sigma_r = prediction_to_r(raw, external.config.target_scale, scale)
    return {
        "entry_index": indices,
        "r_true": np.asarray([float(entry["r"]) for entry in external.entries]),
        "sim_idx": np.asarray([int(entry["sim_idx"]) for entry in external.entries]),
        "r_predicted": predicted_r,
        "sigma_r": sigma_r,
        "r_mean_scaled": raw[:, 0],
        "r_log_variance": raw[:, 1],
    }


def coverage(result: Dict[str, np.ndarray], levels=(0.68, 0.95)) -> Dict[float, float]:
    """Return empirical Gaussian interval coverage for a prediction result."""
    truth = np.asarray(result["r_true"])
    mean = np.asarray(result["r_predicted"])
    sigma = np.asarray(result["sigma_r"])
    normal_quantiles = {0.68: 0.994457883209753, 0.95: 1.959963984540054}
    return {
        float(level): float(
            np.mean(np.abs(truth - mean) <= sigma * normal_quantiles[float(level)])
        )
        for level in levels
    }
