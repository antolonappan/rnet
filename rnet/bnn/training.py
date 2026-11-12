from __future__ import annotations

import json
import os
import csv
from pathlib import Path
from typing import Optional

import numpy as np

from ..cnn.config import DataConfig
from ..cnn.data import SimulationDataset, Standardizer
from .config import BNNModelConfig, BNNTrainingConfig, config_to_dict


def configure_cpu(threads: int) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("OMP_NUM_THREADS", str(int(threads)))
    os.environ.setdefault("TF_NUM_INTRAOP_THREADS", str(int(threads)))
    os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")


def _resolve_checkpoint(manager, resume: Optional[str]):
    if resume is None:
        return None
    if str(resume).lower() == "latest":
        return manager.latest_checkpoint
    candidate = Path(str(resume)).expanduser()
    if candidate.exists() or Path(str(candidate) + ".index").exists():
        return str(candidate)
    numbered = Path(manager.directory) / f"ckpt-{int(resume)}"
    return str(numbered) if Path(str(numbered) + ".index").exists() else None


def _mc_predict(model, dataset, indices, standardizer, batch_size, mc_samples):
    raw_batches = []
    for start in range(0, len(indices), int(batch_size)):
        batch_indices = indices[start : start + int(batch_size)]
        batch_maps = np.stack([dataset.load_sample(int(i), standardizer)[0] for i in batch_indices])
        raw_batches.append(
            np.stack([np.asarray(model(batch_maps, training=False)) for _ in range(int(mc_samples))])
        )
    return np.concatenate(raw_batches, axis=1) if raw_batches else np.empty((mc_samples, 0, 2))


def train(
    data_config: DataConfig,
    model_config: BNNModelConfig,
    training_config: BNNTrainingConfig,
    run_dir: Path | str,
    resume: Optional[str] = None,
    overtrain: bool = False,
):
    """Train the DeepSphere-BNN and save checkpoints plus calibration."""
    if training_config.device == "cpu":
        configure_cpu(training_config.cpu_threads)
    else:
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("OMP_NUM_THREADS", str(training_config.cpu_threads))
    import tensorflow as tf

    from .model import build_model, gaussian_nll, r_bias, r_mae, mc_prediction_to_r

    if training_config.device == "gpu":
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) != 1:
            raise RuntimeError(f"BNN GPU training requires exactly one visible GPU, found {gpus}")
        tf.config.experimental.set_memory_growth(gpus[0], True)
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
        raise FileExistsError(f"Training run already exists: {run_path}; use resume='latest'.")
    run_path.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    dataset = SimulationDataset(data_config)
    if dataset.is_paired_hdf5 and dataset.manifest.get("dataset_kind") != "continuous":
        raise ValueError("The paired grid suite is reserved for testing and covariance")
    if model_config.nside != dataset.nside or model_config.n_channels != len(dataset.frequencies):
        raise ValueError("BNN model geometry does not match the simulation dataset")
    normalization_file = run_path / "normalization.npz"
    source_run = Path(training_config.initialize_from_cnn).expanduser().resolve() if training_config.initialize_from_cnn else None
    source_config = None
    if source_run is not None:
        source_config = json.loads((source_run / "config.json").read_text())
        if source_config["dataset"]["dataset_key"] != dataset.dataset_key:
            raise ValueError("CNN initialization requires the same training dataset")
        for key in ("nside", "n_channels", "conv_blocks", "pool_features", "pooling", "n_neighbors", "dropout"):
            a, b = source_config["model"][key], config_to_dict(model_config)[key]
            if json.dumps(a) != json.dumps(b):
                raise ValueError(f"CNN initialization geometry differs: {key}")
        if source_config["data"]["target_scale"] != data_config.target_scale:
            raise ValueError("CNN initialization target scale differs")
        standardizer = Standardizer.load(source_run / "normalization.npz")
        if normalization_file.is_file():
            saved_norm = Standardizer.load(normalization_file)
            if not (np.array_equal(saved_norm.mean, standardizer.mean) and np.array_equal(saved_norm.std, standardizer.std)):
                raise ValueError("Initialization normalization differs from saved run")
    else:
        standardizer = Standardizer.load(normalization_file) if normalization_file.is_file() else dataset.fit_standardizer()
    if not normalization_file.is_file():
        standardizer.save(normalization_file)

    final_weights = run_path / "final.weights.h5"
    calibration_file = run_path / "calibration.json"
    if final_weights.is_file() and calibration_file.is_file() and not overtrain:
        print(f"Completed BNN run found in {run_path}; skipping training. Pass overtrain=True to continue.")
        kl_weight = training_config.kl_weight or (1.0 / max(len(dataset.splits["train"]), 1))
        model = build_model(model_config, dataset.pixel_indices, kl_weight)
        model.load_weights(final_weights)
        history = type("CompletedHistory", (), {"history": {}})()
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

    kl_weight = training_config.kl_weight or (1.0 / max(len(dataset.splits["train"]), 1))
    run_config = {
        "data": config_to_dict(data_config),
        "model": config_to_dict(model_config),
        "training": config_to_dict(training_config),
        "kl_weight_used": kl_weight,
        "dataset": dataset.summary(),
    }
    if resume is not None and run_config_file.is_file():
        previous = json.loads(run_config_file.read_text(encoding="utf-8"))
        if previous.get("model") != run_config["model"] or previous.get("dataset", {}).get("dataset_key") != dataset.dataset_key:
            raise ValueError("Cannot resume BNN with a different model or dataset")
        for name, default in (("beta_nll", 0), ("signed_null_penalty", 0), ("zero_samples_per_batch", 0), ("initialize_from_cnn", None), ("freeze_backbone", False)):
            if previous.get("training", {}).get(name, default) != run_config["training"][name]:
                raise ValueError(f"Cannot resume BNN with a different {name}")
    with open(run_config_file, "w", encoding="utf-8") as handle:
        json.dump(run_config, handle, indent=2, sort_keys=True)

    model = build_model(model_config, dataset.pixel_indices, kl_weight)
    if source_run is not None and not has_checkpoint:
        from ..cnn.config import ModelConfig
        from ..cnn.model import build_model as build_cnn
        from .initialization import initialize_from_deterministic

        source = build_cnn(ModelConfig(**source_config["model"]), dataset.pixel_indices)
        source.load_weights(source_run / "final.weights.h5")
        initialize_from_deterministic(source, model)
        del source
        import hashlib
        (run_path / "initialization.json").write_text(json.dumps({
            "cnn_run": str(source_run), "posterior_rho": -7.0,
            "weights_sha256": hashlib.sha256((source_run / "final.weights.h5").read_bytes()).hexdigest(),
            "freeze_backbone": training_config.freeze_backbone,
        }, indent=2))
    if training_config.freeze_backbone:
        from .initialization import graph_backbone
        graph_backbone(model).trainable = False
    optimizer = tf.keras.optimizers.Adam(training_config.learning_rate)
    loss = lambda y_true, y_pred: gaussian_nll(
        y_true, y_pred, beta=training_config.beta_nll,
        signed_null_penalty=training_config.signed_null_penalty,
    )
    loss.__name__ = "gaussian_nll"
    model.compile(optimizer=optimizer, loss=loss, metrics=[r_mae, r_bias], jit_compile=False)
    epoch = tf.Variable(0, trainable=False, dtype=tf.int64, name="completed_epoch")
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, epoch=epoch)
    manager = tf.train.CheckpointManager(checkpoint, str(checkpoint_dir), max_to_keep=10)
    restore_path = _resolve_checkpoint(manager, resume)
    if restore_path is not None:
        checkpoint.restore(restore_path).expect_partial()

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
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=training_config.early_stopping_patience, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=training_config.reduce_lr_patience, factor=0.5, min_lr=1.0e-6),
    ]
    if training_config.signed_null_penalty:
        train_data = dataset.make_signed_tf_dataset(
            "train", standardizer, training_config.batch_size,
            training_config.zero_samples_per_batch, training=True,
        )
        validation_data = dataset.make_signed_tf_dataset(
            "validation", standardizer, training_config.batch_size,
            training_config.zero_samples_per_batch,
        )
    else:
        train_data = dataset.make_tf_dataset("train", standardizer, training_config.batch_size, training=True)
        validation_data = dataset.make_tf_dataset("validation", standardizer, training_config.batch_size, training=False)
    history = model.fit(
        train_data, validation_data=validation_data,
        initial_epoch=int(epoch.numpy()), epochs=training_config.epochs,
        callbacks=callbacks, shuffle=False, verbose=2,
    )

    validation_raw = _mc_predict(model, dataset, dataset.splits["validation"], standardizer, training_config.batch_size, training_config.mc_samples)
    prediction = mc_prediction_to_r(validation_raw, data_config.target_scale)
    truth = np.asarray([float(dataset.entries[int(i)]["r"]) for i in dataset.splits["validation"]])
    residual = truth - prediction["r_predicted"]
    calibration_scale = float(np.sqrt(np.mean((residual / np.maximum(prediction["sigma_r"], 1.0e-12)) ** 2)))
    diagnostics = {
        "n_validation": int(len(truth)),
        "true_r_std": float(np.std(truth)),
        "predicted_r_std": float(np.std(prediction["r_predicted"])),
        "rmse": float(np.sqrt(np.mean(np.square(residual)))),
        "bias": float(np.mean(prediction["r_predicted"] - truth)),
        "mean_sigma_r": float(np.mean(prediction["sigma_r"])),
    }
    with open(run_path / "validation_diagnostics.json", "w", encoding="utf-8") as handle:
        json.dump(diagnostics, handle, indent=2)
    with open(run_path / "calibration.json", "w", encoding="utf-8") as handle:
        json.dump({"scale": calibration_scale, "split": "validation", "mc_samples": training_config.mc_samples}, handle, indent=2)
    model.save_weights(run_path / "final.weights.h5")
    return model, history, dataset
