from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .config import DataConfig, ModelConfig, TrainingConfig, config_to_dict
from .data import SimulationDataset, Standardizer


def configure_cpu(threads: int) -> None:
    """Set TensorFlow's CPU runtime before TensorFlow is imported."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(int(threads)))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(int(threads)))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")


def _resolve_checkpoint(manager, resume: Optional[str]) -> Optional[str]:
    if resume is None:
        return None
    if str(resume).lower() == "latest":
        if manager.latest_checkpoint is None:
            # A failed startup can leave config/normalization files without a
            # checkpoint. Treat that state as a fresh run.
            return None
        return manager.latest_checkpoint
    candidate = Path(str(resume)).expanduser()
    if candidate.exists() or Path(str(candidate) + ".index").exists():
        return str(candidate)
    numbered = Path(manager.directory) / f"ckpt-{int(resume)}"
    if Path(str(numbered) + ".index").exists():
        return str(numbered)
    raise FileNotFoundError(f"Could not resolve checkpoint {resume!r}")


def train(
    data_config: DataConfig,
    model_config: ModelConfig,
    training_config: TrainingConfig,
    run_dir: Path | str,
    resume: Optional[str] = None,
    overtrain: bool = False,
    extra_callbacks: Optional[Sequence[object]] = None,
    initial_weights: Optional[Path | str] = None,
):
    """Train on the requested device and return ``(model, history, dataset)``."""
    if training_config.device == "cpu":
        configure_cpu(training_config.cpu_threads)
    else:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("OMP_NUM_THREADS", str(training_config.cpu_threads))
    import tensorflow as tf

    from .model import build_model, gaussian_nll, r_bias, r_mae

    if training_config.device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if not gpus:
            raise RuntimeError("GPU training requested, but TensorFlow sees no GPU")
        for gpu in gpus:
            if not tf.config.experimental.get_memory_growth(gpu):
                tf.config.experimental.set_memory_growth(gpu, True)
    else:
        tf.config.threading.set_intra_op_parallelism_threads(training_config.cpu_threads)
        tf.config.threading.set_inter_op_parallelism_threads(2)
        try:
            tf.config.set_visible_devices([], "GPU")
        except RuntimeError:
            pass
    tf.keras.utils.set_random_seed(training_config.seed)

    run_path = Path(run_dir).expanduser().resolve()
    checkpoint_dir = run_path / "checkpoints"
    run_config_file = run_path / "config.json"
    has_checkpoint = any(checkpoint_dir.glob("*.index"))
    if resume is None and run_config_file.is_file() and has_checkpoint:
        raise FileExistsError(
            f"Run directory already contains a training run: {run_path}. "
            "Use resume='latest' or choose a new run directory."
        )
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = SimulationDataset(data_config)
    if dataset.is_paired_hdf5 and dataset.manifest.get("dataset_kind") != "continuous":
        raise ValueError("The paired grid suite is reserved for testing and covariance")
    if model_config.nside != dataset.nside:
        raise ValueError(f"Model nside={model_config.nside} but dataset nside={dataset.nside}")
    if model_config.n_channels != len(dataset.frequencies):
        raise ValueError(
            f"Model expects {model_config.n_channels} channels but dataset has {len(dataset.frequencies)}"
        )

    normalization_file = run_path / "normalization.npz"
    if normalization_file.is_file():
        standardizer = Standardizer.load(normalization_file)
    else:
        standardizer = dataset.fit_standardizer()
        standardizer.save(normalization_file)
    if initial_weights is not None:
        source_normalization = Path(initial_weights).expanduser().resolve().parent / "normalization.npz"
        if source_normalization.is_file():
            source = Standardizer.load(source_normalization)
            if not (np.allclose(source.mean, standardizer.mean, rtol=1e-6, atol=1e-8)
                    and np.allclose(source.std, standardizer.std, rtol=1e-6, atol=1e-8)):
                raise ValueError("Warm-start weights use different input normalization")

    final_weights = run_path / "final.weights.h5"
    calibration_file = run_path / "calibration.json"
    if final_weights.is_file() and calibration_file.is_file() and not overtrain:
        print(f"Completed CNN run found in {run_path}; skipping training. Pass overtrain=True to continue.")
        model = build_model(model_config, dataset.pixel_indices)
        model.load_weights(final_weights)
        history = tf.keras.callbacks.History()
        history.history = {}
        history_file = run_path / "history.csv"
        if history_file.is_file():
            with open(history_file, newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            if rows:
                history.history = {
                    key: [float(row[key]) for row in rows if row.get(key, "") != ""]
                    for key in rows[0] if key != "epoch"
                }
        return model, history, dataset

    run_config = json.loads(
        json.dumps(
            {
                "data": config_to_dict(data_config),
                "model": config_to_dict(model_config),
                "training": config_to_dict(training_config),
                "dataset": dataset.summary(),
            }
        )
    )
    if resume is not None and run_config_file.is_file():
        with open(run_config_file, "r", encoding="utf-8") as handle:
            previous_config = json.load(handle)
        if previous_config.get("model") != run_config["model"]:
            raise ValueError("Cannot resume with a different model configuration")
        old_key = previous_config.get("dataset", {}).get("dataset_key")
        new_key = run_config["dataset"].get("dataset_key")
        if old_key is not None and new_key is not None and old_key != new_key:
            raise ValueError("Cannot resume with a different simulation dataset")
        for key in ("beta_nll", "signed_null_penalty", "zero_samples_per_batch"):
            if previous_config.get("training", {}).get(key, 0) != run_config["training"][key]:
                raise ValueError(f"Cannot resume with a different {key}")
    with open(run_config_file, "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    model = build_model(model_config, dataset.pixel_indices)
    optimizer = tf.keras.optimizers.Adam(training_config.learning_rate)
    loss = lambda y_true, y_pred: gaussian_nll(
        y_true, y_pred, beta=training_config.beta_nll,
        signed_null_penalty=training_config.signed_null_penalty,
    )
    loss.__name__ = "gaussian_nll"
    # DeepSphere's sparse graph operations have no XLA GPU kernels.
    model.compile(optimizer=optimizer, loss=loss, metrics=[r_mae, r_bias], jit_compile=False)

    epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=10)
    restore_path = _resolve_checkpoint(manager, resume)
    if restore_path is not None:
        checkpoint.restore(restore_path).expect_partial()
    elif initial_weights is not None:
        weights = Path(initial_weights).expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Initial CNN weights not found: {weights}")
        model.load_weights(weights)
        print(f"Initialized CNN weights from {weights}")

    class PeriodicCheckpoint(tf.keras.callbacks.Callback):
        def on_epoch_end(self, completed_epoch, logs=None):
            epoch.assign(int(completed_epoch) + 1)
            if int(epoch.numpy()) % training_config.save_every_epochs == 0:
                manager.save(checkpoint_number=int(optimizer.iterations.numpy()))

        def on_train_end(self, logs=None):
            manager.save(checkpoint_number=int(optimizer.iterations.numpy()))

    callbacks = [
        PeriodicCheckpoint(),
        tf.keras.callbacks.CSVLogger(str(run_path / "history.csv"), append=restore_path is not None),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=training_config.early_stopping_patience,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=training_config.reduce_lr_patience,
            factor=0.5,
            min_lr=1.0e-6,
        ),
    ]
    if extra_callbacks:
        callbacks.extend(extra_callbacks)
    if training_config.signed_null_penalty:
        training_data = dataset.make_signed_tf_dataset(
            "train", standardizer, training_config.batch_size,
            training_config.zero_samples_per_batch, training=True,
        )
        validation_data = dataset.make_signed_tf_dataset(
            "validation", standardizer, training_config.batch_size,
            training_config.zero_samples_per_batch,
        )
    else:
        training_data = dataset.make_tf_dataset(
            "train", standardizer, training_config.batch_size, training=True
        )
        validation_data = dataset.make_tf_dataset(
            "validation", standardizer, training_config.batch_size, training=False
        )
    if len(dataset.splits["validation"]) == 0:
        raise ValueError("Validation split is empty; increase samples per r or set a larger fraction")

    history = model.fit(
        training_data,
        validation_data=validation_data,
        initial_epoch=int(epoch.numpy()),
        epochs=training_config.epochs,
        callbacks=callbacks,
        shuffle=False,
    )
    calibration_data = dataset.make_tf_dataset(
        "validation", standardizer, training_config.batch_size, training=False
    )
    validation_predictions = np.asarray(model.predict(calibration_data, verbose=0))
    validation_truth = np.concatenate(
        [target for _, target in dataset.iter_samples("validation", standardizer)], axis=0
    )
    from .model import prediction_to_r

    predicted_r, sigma_r = prediction_to_r(
        validation_predictions, data_config.target_scale, calibration_scale=1.0
    )
    # ``validation_truth`` is stored in the scaled training units, while
    # ``predicted_r`` has already been converted back to physical r.
    true_r = np.asarray(validation_truth, dtype=np.float64).reshape(-1) * float(data_config.target_scale)
    residual = true_r - np.asarray(predicted_r, dtype=np.float64)
    raw_sigma = np.maximum(np.asarray(sigma_r, dtype=np.float64), 1.0e-12)
    calibration_scale = float(np.sqrt(np.mean((residual / raw_sigma) ** 2)))
    validation_diagnostics = {
        "n": int(len(true_r)),
        "mean_predicted_r": float(np.mean(predicted_r)),
        "predicted_r_std": float(np.std(predicted_r, ddof=1)),
        "true_r_std": float(np.std(true_r, ddof=1)),
        "bias": float(np.mean(-residual)),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "mean_raw_sigma_r": float(np.mean(raw_sigma)),
    }
    with open(run_path / "validation_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(validation_diagnostics, handle, indent=2)
    with open(run_path / "calibration.json", "w", encoding="utf-8") as handle:
        json.dump({"scale": calibration_scale, "split": "validation"}, handle, indent=2)
    model.save_weights(run_path / "final.weights.h5")
    if len(dataset.splits["test"]) > 0:
        test_data = dataset.make_tf_dataset(
            "test", standardizer, training_config.batch_size, training=False
        )
        test_metrics = model.evaluate(test_data, return_dict=True, verbose=0)
        with open(run_path / "test_metrics.json", "w", encoding="utf-8") as handle:
            json.dump({key: float(value) for key, value in test_metrics.items()}, handle, indent=2)
    return model, history, dataset
