from __future__ import annotations

import numpy as np
import tensorflow as tf


R_SCALE = 0.01
LEVELS = (0.025, 0.16, 0.5, 0.84, 0.975)
_GAP_68 = float(np.log(np.expm1(1.0)))
_GAP_TAIL = float(np.log(np.expm1(0.96)))


def pinball_loss(y_true, quantiles):
    """Mean proper quantile score in scaled-r units."""
    levels = tf.constant(LEVELS, dtype=quantiles.dtype)
    residual = tf.reshape(tf.cast(y_true, quantiles.dtype), (-1, 1)) - quantiles
    return tf.reduce_mean(tf.maximum(levels * residual, (levels - 1.0) * residual))


def build_model(pixel_indices, cnn_weights, cnn_config):
    from rnet.cnn.model import build_model as build_cnn_model

    cnn = build_cnn_model(cnn_config, np.asarray(pixel_indices, dtype=np.int64))
    cnn.load_weights(cnn_weights)
    encoder = tf.keras.Model(
        cnn.inputs,
        [cnn.get_layer("r_mean_log_variance").input, cnn.output],
        name="frozen_cnn_encoder",
    )
    encoder.trainable = False

    inputs = tf.keras.Input((len(pixel_indices), cnn_config.n_channels), name="bmode_maps")
    features, point = encoder(inputs, training=False)
    corrections = tf.keras.layers.Dense(
        5, kernel_initializer="zeros", bias_initializer="zeros",
        name="quantile_calibration",
    )(features)

    def ordered(values):
        point, correction = values
        center = point[:, :1] + correction[:, :1]
        scale = tf.exp(0.5 * tf.clip_by_value(point[:, 1:2], -20.0, 5.0))
        lower_68 = scale * tf.nn.softplus(correction[:, 1:2] + _GAP_68)
        lower_tail = scale * tf.nn.softplus(correction[:, 2:3] + _GAP_TAIL)
        upper_68 = scale * tf.nn.softplus(correction[:, 3:4] + _GAP_68)
        upper_tail = scale * tf.nn.softplus(correction[:, 4:5] + _GAP_TAIL)
        return tf.concat([
            center - lower_68 - lower_tail,
            center - lower_68,
            center,
            center + upper_68,
            center + upper_68 + upper_tail,
        ], axis=-1)

    quantiles = tf.keras.layers.Lambda(ordered, name="r_quantiles")( [point, corrections] )
    return tf.keras.Model(inputs, quantiles, name="rnet_anchored_quantiles")


def summarize(prediction):
    values = np.asarray(prediction, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LEVELS):
        raise ValueError("Expected five ordered quantiles per map")
    if np.any(np.diff(values, axis=1) < -1.0e-8):
        raise ValueError("Quantiles are not ordered")
    physical = values * R_SCALE
    return {
        "r_q025": physical[:, 0],
        "r_q16": physical[:, 1],
        "r_median": physical[:, 2],
        "r_q84": physical[:, 3],
        "r_q975": physical[:, 4],
        "sigma_r": 0.5 * (physical[:, 3] - physical[:, 1]),
    }
