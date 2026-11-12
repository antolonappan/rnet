from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf

from rnet.cnn.config import DataConfig, ModelConfig, config_to_dict
from rnet.cnn.data import SimulationDataset, Standardizer

from .model import LEVELS, R_SCALE, build_model, pinball_loss, summarize


def train(data_config: DataConfig, cnn_weights: Path | str, run_dir: Path | str, *,
          batch_size: int = 64, epochs: int = 150, learning_rate: float = 1.0e-4,
          seed: int = 2027):
    if batch_size <= 0 or epochs <= 0 or learning_rate <= 0:
        raise ValueError("batch_size, epochs, and learning_rate must be positive")
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) != 1:
        raise RuntimeError(f"Quantile training expects one GPU, found {gpus}")
    tf.config.experimental.set_memory_growth(gpus[0], True)
    tf.keras.utils.set_random_seed(seed)

    source_weights = Path(cnn_weights).expanduser().resolve()
    source_config = json.loads((source_weights.parent / "config.json").read_text(encoding="utf-8"))
    source_model = ModelConfig(**source_config["model"])
    source_norm = Standardizer.load(source_weights.parent / "normalization.npz")
    dataset = SimulationDataset(data_config)
    if dataset.manifest.get("dataset_kind") != "continuous":
        raise ValueError("The grid suite is reserved for evaluation")
    if dataset.dataset_key != source_config.get("dataset", {}).get("dataset_key"):
        raise ValueError("Quantile head and pretrained CNN must use the same dataset")
    if float(source_config.get("data", {}).get("target_scale", -1)) != R_SCALE:
        raise ValueError("Pretrained CNN uses a different r target scale")
    if dataset.nside != source_model.nside or len(dataset.frequencies) != source_model.n_channels:
        raise ValueError("Pretrained CNN geometry or channels differ")

    run_path = Path(run_dir).expanduser().resolve()
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoints = run_path / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    config = {
        "data": config_to_dict(data_config),
        "dataset_key": dataset.dataset_key,
        "model": {
            "head": "anchored_ordered_quantiles", "levels": LEVELS,
            "source_cnn_weights": str(source_weights), "r_scale": R_SCALE,
        },
        "training": {"batch_size": batch_size, "epochs": epochs,
                     "learning_rate": learning_rate, "seed": seed,
                     "early_stopping_patience": 20, "reduce_lr_patience": 7},
    }
    config_file = run_path / "config.json"
    if config_file.is_file():
        old = json.loads(config_file.read_text(encoding="utf-8"))
        if old.get("dataset_key") != config["dataset_key"] or old.get("model") != json.loads(json.dumps(config["model"])):
            raise ValueError("Cannot resume quantile run with changed data or model")
        if old.get("training", {}).get("batch_size") != batch_size:
            raise ValueError("Cannot resume quantile run with a different batch size")
    else:
        config_file.write_text(json.dumps(config, indent=2), encoding="utf-8")
    norm_file = run_path / "normalization.npz"
    if norm_file.is_file():
        standardizer = Standardizer.load(norm_file)
        if not (np.allclose(standardizer.mean, source_norm.mean, rtol=1e-6, atol=1e-8)
                and np.allclose(standardizer.std, source_norm.std, rtol=1e-6, atol=1e-8)):
            raise ValueError("Pretrained CNN and quantile run use different normalization")
    else:
        standardizer = source_norm
        standardizer.save(norm_file)

    final_weights = run_path / "final.weights.h5"
    diagnostic_file = run_path / "validation_diagnostics.json"
    if final_weights.is_file() and diagnostic_file.is_file():
        print(f"Completed quantile run found: {run_path}", flush=True)
        return json.loads(diagnostic_file.read_text(encoding="utf-8"))

    model = build_model(dataset.pixel_indices, source_weights, source_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate)
    model.compile(optimizer=optimizer, loss=pinball_loss, jit_compile=False)
    epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoints), max_to_keep=5)
    if manager.latest_checkpoint is not None:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Resuming quantile run at epoch {int(epoch.numpy())}", flush=True)

    class SaveEveryEpoch(tf.keras.callbacks.Callback):
        def on_epoch_end(self, completed_epoch, logs=None):
            epoch.assign(completed_epoch + 1)
            manager.save(checkpoint_number=int(epoch.numpy()))

    callbacks = [
        SaveEveryEpoch(),
        tf.keras.callbacks.CSVLogger(str(run_path / "history.csv"), append=int(epoch.numpy()) > 0),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", patience=7, factor=0.5, min_lr=1.0e-6,
        ),
    ]
    training_data = dataset.make_tf_dataset("train", standardizer, batch_size, training=True)
    validation_data = dataset.make_tf_dataset("validation", standardizer, batch_size, training=False)
    model.fit(training_data, validation_data=validation_data,
              initial_epoch=int(epoch.numpy()), epochs=epochs,
              callbacks=callbacks, shuffle=False, verbose=2)

    raw = np.asarray(model.predict(validation_data, verbose=0), dtype=np.float64)
    summary = summarize(raw)
    indices = dataset.splits["validation"]
    truth = np.asarray([float(dataset.entries[int(i)]["r"]) for i in indices])
    train_truth = np.asarray([
        float(dataset.entries[int(i)]["r"]) / R_SCALE for i in dataset.splits["train"]
    ])
    prior_quantiles = np.quantile(train_truth, LEVELS)
    prior_raw = np.broadcast_to(prior_quantiles, raw.shape).astype(np.float32)
    baseline = float(pinball_loss(tf.convert_to_tensor(truth / R_SCALE, tf.float32),
                                  tf.convert_to_tensor(prior_raw)).numpy())
    score = float(pinball_loss(tf.convert_to_tensor(truth / R_SCALE, tf.float32),
                               tf.convert_to_tensor(raw, tf.float32)).numpy())
    residual = summary["r_median"] - truth
    diagnostics = {
        "n_validation": len(truth),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "predicted_median_std": float(np.std(summary["r_median"])),
        "true_r_std": float(np.std(truth)),
        "mean_sigma_r": float(np.mean(summary["sigma_r"])),
        "coverage_68": float(np.mean((truth >= summary["r_q16"]) & (truth <= summary["r_q84"]))),
        "coverage_95": float(np.mean((truth >= summary["r_q025"]) & (truth <= summary["r_q975"]))),
        "validation_pinball": score,
        "training_prior_pinball_on_validation": baseline,
        "pinball_gain_over_prior": baseline - score,
    }
    predictions = pd.DataFrame({
        "sim_idx": [int(dataset.entries[int(i)]["sim_idx"]) for i in indices],
        "r_true": truth, **summary,
    })
    predictions.to_csv(run_path / "validation_predictions.csv", index=False)
    model.save_weights(final_weights)
    diagnostic_file.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    return diagnostics
