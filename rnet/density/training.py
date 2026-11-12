from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rnet.cnn.config import DataConfig, ModelConfig, config_to_dict
from rnet.cnn.data import SimulationDataset, Standardizer

from .model import (
    N_COMPONENTS, R_MAX_SCALED, R_SCALE, anchored_posterior_nll,
    build_anchored_model, build_model, posterior_nll, posterior_summary,
)


def train(
    data_config: DataConfig,
    run_dir: Path | str,
    *,
    batch_size: int = 64,
    epochs: int = 300,
    learning_rate: float = 1.0e-4,
    seed: int = 2027,
    initial_cnn_weights: Path | str | None = None,
):
    if batch_size <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError(f"Density-head training requires one visible GPU, found {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.utils.set_random_seed(seed)

    run_path = Path(run_dir).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = run_path / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    dataset = SimulationDataset(data_config)
    if dataset.nside != 128 or len(dataset.frequencies) != 6:
        raise ValueError("This posterior model expects nside=128 and six frequency channels")
    if dataset.manifest.get("dataset_kind") != "continuous":
        raise ValueError("The grid suite is reserved for evaluation")

    source_weights = None
    source_model = None
    source_norm = None
    if initial_cnn_weights is not None:
        source_weights = Path(initial_cnn_weights).expanduser().resolve()
        if not source_weights.is_file():
            raise FileNotFoundError(source_weights)
        source_config = json.loads((source_weights.parent / "config.json").read_text(encoding="utf-8"))
        if source_config.get("dataset", {}).get("dataset_key") != dataset.dataset_key:
            raise ValueError("Pretrained CNN and density run must use the same dataset")
        if float(source_config.get("data", {}).get("target_scale", -1)) != R_SCALE:
            raise ValueError("Pretrained CNN uses a different r target scale")
        source_model = ModelConfig(**source_config["model"])
        if source_model.n_channels != len(dataset.frequencies) or source_model.nside != dataset.nside:
            raise ValueError("Pretrained CNN geometry or channels differ")
        source_norm = Standardizer.load(source_weights.parent / "normalization.npz")

    config = {
        "data": config_to_dict(data_config),
        "dataset_key": dataset.dataset_key,
        "model": {
            "head": "anchored_truncated_normal" if source_weights else "zero_atom_truncated_normal_mixture",
            "n_components": 1 if source_weights else N_COMPONENTS,
            "r_min": 0.0,
            "r_max": R_MAX_SCALED * R_SCALE,
            "r_scale": R_SCALE,
            "dense_features": 64,
            "source_cnn_weights": str(source_weights) if source_weights else None,
        },
        "training": {
            "batch_size": batch_size, "epochs": epochs,
            "learning_rate": learning_rate, "seed": seed,
            "early_stopping_patience": 20, "reduce_lr_patience": 7,
        },
    }
    config_file = run_path / "config.json"
    if config_file.is_file():
        previous = json.loads(config_file.read_text(encoding="utf-8"))
        if previous.get("dataset_key") != config["dataset_key"] or previous.get("model") != config["model"]:
            raise ValueError("Cannot resume density training with different data or model")
        if previous.get("training", {}).get("batch_size") != batch_size:
            raise ValueError("Cannot resume density training with a different batch size")
    else:
        config_file.write_text(json.dumps(config, indent=2, sort_keys=True), encoding="utf-8")

    norm_file = run_path / "normalization.npz"
    if norm_file.is_file():
        standardizer = Standardizer.load(norm_file)
    else:
        standardizer = source_norm if source_norm is not None else dataset.fit_standardizer()
        standardizer.save(norm_file)
    if source_norm is not None and not (
        np.allclose(standardizer.mean, source_norm.mean, rtol=1e-6, atol=1e-8)
        and np.allclose(standardizer.std, source_norm.std, rtol=1e-6, atol=1e-8)
    ):
        raise ValueError("Pretrained CNN and density run use different input normalization")

    final_weights = run_path / "final.weights.h5"
    diagnostic_file = run_path / "validation_diagnostics.json"
    if final_weights.is_file() and diagnostic_file.is_file():
        print(f"Completed density run found: {run_path}", flush=True)
        return json.loads(diagnostic_file.read_text(encoding="utf-8"))

    model = (
        build_anchored_model(dataset.pixel_indices, source_weights, source_model)
        if source_weights else build_model(dataset.pixel_indices)
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    loss = anchored_posterior_nll if source_weights else posterior_nll
    model.compile(optimizer=optimizer, loss=loss, jit_compile=False)
    epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=5)
    if manager.latest_checkpoint is not None:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Resuming density run at epoch {int(epoch.numpy())}", flush=True)

    class SaveEveryEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, completed_epoch, logs=None):
            epoch.assign(completed_epoch + 1)
            manager.save(checkpoint_number=int(epoch.numpy()))

    callbacks = [
        SaveEveryEpoch(),
        tf.keras.callbacks.CSVLogger(
            str(run_path / "history.csv"), append=int(epoch.numpy()) > 0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=20, restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=7, factor=0.5, min_lr=1.0e-6,
        ),
    ]
    training_data = dataset.make_tf_dataset("train", standardizer, batch_size, training=True)
    validation_data = dataset.make_tf_dataset("validation", standardizer, batch_size, training=False)
    model.fit(
        training_data, validation_data=validation_data,
        initial_epoch=int(epoch.numpy()), epochs=epochs, callbacks=callbacks,
        shuffle=False, verbose=2,
    )

    raw = np.asarray(model.predict(validation_data, verbose=0), dtype=np.float64)
    np.savez_compressed(run_path / "validation_raw_parameters.npz", raw=raw)
    summary = posterior_summary(raw)
    indices = dataset.splits["validation"]
    truth = np.asarray([float(dataset.entries[int(i)]["r"]) for i in indices])
    residual = summary["r_mean"] - truth
    train_zero_fraction = float(np.mean([
        float(dataset.entries[int(i)]["r"]) == 0.0 for i in dataset.splits["train"]
    ]))
    validation_zero_fraction = float(np.mean(truth == 0.0))
    prior_nll = -validation_zero_fraction * np.log(train_zero_fraction)
    prior_nll -= (1.0 - validation_zero_fraction) * np.log1p(-train_zero_fraction)
    prior_nll += (1.0 - validation_zero_fraction) * np.log(R_MAX_SCALED)
    validation_nll = float(loss(
        tf.convert_to_tensor(truth / R_SCALE, dtype=tf.float32),
        tf.convert_to_tensor(raw, dtype=tf.float32),
    ).numpy())
    diagnostics = {
        "n_validation": int(len(truth)),
        "n_zero": int(np.count_nonzero(truth == 0.0)),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "predicted_r_std": float(np.std(summary["r_mean"])),
        "true_r_std": float(np.std(truth)),
        "mean_p_zero_at_zero": float(np.mean(summary["p_zero"][truth == 0.0])),
        "mean_p_zero_at_positive": float(np.mean(summary["p_zero"][truth > 0.0])),
        "coverage_68": float(np.mean((truth >= summary["r_q16"]) & (truth <= summary["r_q84"]))),
        "coverage_95": float(np.mean((truth >= summary["r_q025"]) & (truth <= summary["r_q975"]))),
        "mean_interval_width_68": float(np.mean(summary["r_q84"] - summary["r_q16"])),
        "validation_nll": validation_nll,
        "uniform_prior_nll": float(prior_nll),
        "training_zero_fraction": train_zero_fraction,
        "validation_zero_fraction": validation_zero_fraction,
        "nll_gain_over_prior": float(prior_nll - validation_nll),
    }
    predictions = pd.DataFrame({
        "sim_idx": [int(dataset.entries[int(i)]["sim_idx"]) for i in indices],
        "r_true": truth,
        **summary,
    })
    predictions.to_csv(run_path / "validation_predictions.csv", index=False)
    model.save_weights(final_weights)
    diagnostic_file.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics
