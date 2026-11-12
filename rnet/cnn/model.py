from __future__ import annotations

from typing import Tuple

import numpy as np

from .config import ModelConfig
import tensorflow as tf
from deepsphere.healpy_layers import HealpyChebyshev, HealpyPseudoConv
from deepsphere.healpy_networks import HealpyGCNN


def gaussian_nll(y_true, y_pred, beta: float = 0.0, signed_null_penalty: float = 0.0):
    """Gaussian NLL for scaled r with optional beta-NLL weighting."""

    mu = y_pred[..., :1]
    log_variance = tf.clip_by_value(y_pred[..., 1:2], -30.0, 5.0)
    nll = 0.5 * (log_variance + tf.square(y_true - mu) * tf.exp(-log_variance))
    if beta:
        factor = tf.stop_gradient(tf.exp(beta * log_variance))
        nll = factor * nll
    loss = tf.reduce_mean(nll)
    if signed_null_penalty:
        zero = tf.cast(tf.equal(y_true, 0.0), mu.dtype)
        null_mean = tf.math.divide_no_nan(tf.reduce_sum(mu * zero), tf.reduce_sum(zero))
        loss += float(signed_null_penalty) * tf.square(null_mean)
    return loss


def r_mae(y_true, y_pred):
    return tf.reduce_mean(tf.abs(y_true - y_pred[..., :1]))


def r_bias(y_true, y_pred):
    return tf.reduce_mean(y_pred[..., :1] - y_true)


def prediction_to_r(
    prediction: np.ndarray,
    target_scale: float = 1.0e-2,
    calibration_scale: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert direct, scaled-r network outputs to r and its uncertainty."""
    prediction = np.asarray(prediction)
    mu = prediction[..., 0] * float(target_scale)
    sigma_r = np.exp(0.5 * np.clip(prediction[..., 1], -30.0, 5.0)) * float(target_scale)
    return mu, sigma_r * float(calibration_scale)


def build_model(config: ModelConfig, pixel_indices: np.ndarray):
    """Build a DeepSphere regressor for selected NEST B-mode pixels."""
    indices = np.asarray(pixel_indices, dtype=np.int64)
    regularizer = (
        tf.keras.regularizers.l2(config.l2_regularization)
        if config.l2_regularization > 0
        else None
    )

    layers = []
    for level, (order, features) in enumerate(config.conv_blocks):
        layers.append(
            HealpyChebyshev(
                K=int(order),
                Fout=int(features),
                activation="relu",
                use_bias=True,
                use_bn=True,
                regularizer=regularizer,
            )
        )
        if level < len(config.pool_features):
            layers.append(
                HealpyPseudoConv(
                    p=1,
                    Fout=int(config.pool_features[level]),
                    activation="relu",
                    use_bias=True,
                )
            )

    graph_network = HealpyGCNN(
        nside=config.nside,
        indices=indices,
        layers=layers,
        n_neighbors=config.n_neighbors,
        max_batch_size=None,
        initial_Fin=config.n_channels,
    )
    inputs = tf.keras.Input((len(indices), config.n_channels), name="bmode_maps")
    features = graph_network(inputs)
    pooled = []
    if "mean" in config.pooling:
        pooled.append(tf.keras.layers.GlobalAveragePooling1D(name="spatial_average")(features))
    if "var" in config.pooling:
        mean = tf.keras.layers.GlobalAveragePooling1D(name="spatial_mean_for_variance")(features)
        second = tf.keras.layers.Lambda(lambda value: tf.reduce_mean(tf.square(value), axis=1))(features)
        mean_square = tf.keras.layers.Lambda(
            lambda value: tf.square(value), name="spatial_mean_square"
        )(mean)
        pooled.append(tf.keras.layers.Subtract(name="spatial_variance")([second, mean_square]))
    if "max" in config.pooling:
        pooled.append(tf.keras.layers.GlobalMaxPooling1D(name="spatial_maximum")(features))
    features = tf.keras.layers.Concatenate()(pooled)
    if config.dropout > 0:
        features = tf.keras.layers.Dropout(config.dropout)(features)
    widths = (config.dense_features,) if isinstance(config.dense_features, int) else config.dense_features
    for units in widths:
        features = tf.keras.layers.Dense(
            int(units), activation="relu", kernel_regularizer=regularizer,
        )(features)
    outputs = tf.keras.layers.Dense(2, name="r_mean_log_variance")(features)
    return tf.keras.Model(inputs, outputs, name="rnet_deepsphere")
