from __future__ import annotations

import math

import numpy as np
import tensorflow as tf
from scipy.special import ndtr


R_SCALE = 0.01
R_MAX_SCALED = 3.0
N_COMPONENTS = 3
N_OUTPUTS = 1 + 3 * N_COMPONENTS
_SQRT2 = math.sqrt(2.0)
_SQRT2PI = math.sqrt(2.0 * math.pi)


def _tf_parts(raw):
    atom = raw[..., 0]
    logits, centers, spreads = tf.split(raw[..., 1:], 3, axis=-1)
    weights = tf.nn.softmax(logits, axis=-1)
    means = R_MAX_SCALED * tf.nn.sigmoid(centers)
    scales = tf.nn.softplus(spreads) + 0.05
    return atom, weights, means, scales


def _tf_cdf(z):
    return 0.5 * (1.0 + tf.math.erf(z / _SQRT2))


def posterior_nll(y_true, raw):
    """NLL with respect to a discrete atom at zero plus continuous r/0.01."""
    atom, weights, means, scales = _tf_parts(raw)
    y = tf.reshape(tf.cast(y_true, raw.dtype), (-1, 1))
    lo = -means / scales
    hi = (R_MAX_SCALED - means) / scales
    normalization = tf.maximum(_tf_cdf(hi) - _tf_cdf(lo), 1.0e-12)
    z = (y - means) / scales
    log_components = (
        tf.math.log(tf.maximum(weights, 1.0e-12))
        - 0.5 * tf.square(z) - tf.math.log(scales)
        - 0.5 * math.log(2.0 * math.pi) - tf.math.log(normalization)
    )
    log_slab = tf.reduce_logsumexp(log_components, axis=-1)
    log_probability = tf.where(
        tf.equal(tf.reshape(y, (-1,)), 0.0),
        -tf.nn.softplus(-atom),
        -tf.nn.softplus(atom) + log_slab,
    )
    return -tf.reduce_mean(log_probability)


def build_model(pixel_indices: np.ndarray, nside: int = 128, n_channels: int = 6):
    """Use the same graph backbone, pooling, and 64-unit head as the CNN."""
    from deepsphere.healpy_layers import HealpyChebyshev, HealpyPseudoConv
    from deepsphere.healpy_networks import HealpyGCNN

    layers = []
    regularizer = tf.keras.regularizers.l2(1.0e-6)
    for level, (order, features) in enumerate(((5, 8), (5, 16), (5, 32), (3, 64))):
        layers.append(HealpyChebyshev(
            K=order, Fout=features, activation="relu", use_bias=True,
            use_bn=True, regularizer=regularizer,
        ))
        if level < 3:
            layers.append(HealpyPseudoConv(
                p=1, Fout=(16, 32, 64)[level], activation="relu", use_bias=True,
            ))
    graph = HealpyGCNN(
        nside=nside, indices=np.asarray(pixel_indices, dtype=np.int64),
        layers=layers, n_neighbors=8, max_batch_size=None,
        initial_Fin=n_channels,
    )
    inputs = tf.keras.Input((len(pixel_indices), n_channels), name="bmode_maps")
    features = graph(inputs)
    average = tf.keras.layers.GlobalAveragePooling1D()(features)
    second = tf.keras.layers.Lambda(lambda value: tf.reduce_mean(tf.square(value), axis=1))(features)
    variance = tf.keras.layers.Subtract()([
        second, tf.keras.layers.Lambda(tf.square)(average),
    ])
    maximum = tf.keras.layers.GlobalMaxPooling1D()(features)
    features = tf.keras.layers.Concatenate()([average, variance, maximum])
    features = tf.keras.layers.Dropout(0.1)(features)
    features = tf.keras.layers.Dense(64, activation="relu", kernel_regularizer=regularizer)(features)
    outputs = tf.keras.layers.Dense(N_OUTPUTS, name="r_posterior_parameters")(features)
    return tf.keras.Model(inputs, outputs, name="rnet_deepsphere_posterior")


def _np_parts(raw):
    values = np.asarray(raw, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] not in (3, N_OUTPUTS):
        raise ValueError(f"Expected posterior parameters with 3 or {N_OUTPUTS} columns")
    atom = 1.0 / (1.0 + np.exp(-np.clip(values[:, 0], -60.0, 60.0)))
    if values.shape[1] == 3:
        weights = np.ones((len(values), 1), dtype=np.float64)
        means = values[:, 1:2]
        scales = np.exp(0.5 * np.clip(values[:, 2:3], -20.0, 5.0))
        return atom, weights, means, scales
    logits, centers, spreads = np.split(values[:, 1:], 3, axis=1)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    weights = np.exp(shifted)
    weights /= np.sum(weights, axis=1, keepdims=True)
    means = R_MAX_SCALED / (1.0 + np.exp(-np.clip(centers, -60.0, 60.0)))
    scales = np.logaddexp(0.0, spreads) + 0.05
    return atom, weights, means, scales


def anchored_posterior_nll(y_true, raw):
    """NLL of a zero atom and one Gaussian truncated to 0 <= r/0.01 <= 3."""
    atom_logit, mean, log_variance = tf.split(raw, 3, axis=-1)
    scale = tf.exp(0.5 * tf.clip_by_value(log_variance, -20.0, 5.0))
    y = tf.reshape(tf.cast(y_true, raw.dtype), (-1, 1))
    normalization = tf.maximum(
        _tf_cdf((R_MAX_SCALED - mean) / scale) - _tf_cdf(-mean / scale),
        1.0e-12,
    )
    z = (y - mean) / scale
    log_slab = -0.5 * tf.square(z) - tf.math.log(scale) - 0.5 * math.log(2.0 * math.pi) - tf.math.log(normalization)
    log_probability = tf.where(
        tf.equal(y, 0.0),
        -tf.nn.softplus(-atom_logit),
        -tf.nn.softplus(atom_logit) + log_slab,
    )
    return -tf.reduce_mean(log_probability)


def build_anchored_model(pixel_indices: np.ndarray, cnn_weights, cnn_config):
    """Calibrate a frozen, pretrained CNN into a conditional posterior."""
    from rnet.cnn.model import build_model as build_cnn_model

    cnn = build_cnn_model(cnn_config, pixel_indices)
    cnn.load_weights(cnn_weights)
    encoder = tf.keras.Model(
        cnn.inputs,
        [cnn.get_layer("r_mean_log_variance").input, cnn.output],
        name="frozen_cnn_encoder",
    )
    encoder.trainable = False
    inputs = tf.keras.Input((len(pixel_indices), cnn_config.n_channels), name="bmode_maps")
    features, cnn_prediction = encoder(inputs, training=False)
    correction = tf.keras.layers.Dense(
        3, kernel_initializer="zeros",
        bias_initializer=tf.keras.initializers.Constant([-3.0, 0.0, 0.0]),
        name="posterior_calibration",
    )(features)
    atom = correction[:, :1]
    mean = cnn_prediction[:, :1] + correction[:, 1:2]
    log_variance = cnn_prediction[:, 1:2] + correction[:, 2:3]
    outputs = tf.keras.layers.Concatenate(name="r_posterior_parameters")([
        atom, mean, log_variance,
    ])
    return tf.keras.Model(inputs, outputs, name="rnet_anchored_posterior")


def posterior_cdf(raw, r):
    """CDF including the point mass at r=0; r may be scalar or one per map."""
    atom, weights, means, scales = _np_parts(raw)
    values = np.broadcast_to(np.asarray(r, dtype=np.float64), atom.shape)
    scaled = values[:, None] / R_SCALE
    low = ndtr(-means / scales)
    high = ndtr((R_MAX_SCALED - means) / scales)
    normalization = np.maximum(high - low, 1.0e-12)
    conditional = np.sum(weights * np.clip((ndtr((scaled - means) / scales) - low) / normalization, 0.0, 1.0), axis=1)
    cdf = atom + (1.0 - atom) * conditional
    return np.where(values < 0.0, 0.0, np.where(values >= R_MAX_SCALED * R_SCALE, 1.0, cdf))


def posterior_slab_pdf(raw, r, *, weighted: bool = True):
    """Continuous density in physical r units; the zero atom is separate."""
    atom, weights, means, scales = _np_parts(raw)
    values = np.broadcast_to(np.asarray(r, dtype=np.float64), atom.shape)
    scaled = values[:, None] / R_SCALE
    low = ndtr(-means / scales)
    high = ndtr((R_MAX_SCALED - means) / scales)
    normalization = np.maximum(high - low, 1.0e-12)
    z = (scaled - means) / scales
    components = np.exp(-0.5 * z * z) / (_SQRT2PI * scales * normalization)
    pdf = np.sum(weights * components, axis=1) / R_SCALE
    if weighted:
        pdf *= 1.0 - atom
    return np.where((values < 0.0) | (values > R_MAX_SCALED * R_SCALE), 0.0, pdf)


def posterior_quantile(raw, probability, iterations: int = 48):
    """Numerically invert the bounded mixture CDF."""
    values = np.asarray(raw)
    p = np.broadcast_to(np.asarray(probability, dtype=np.float64), (len(values),))
    if np.any((p < 0.0) | (p > 1.0)):
        raise ValueError("Quantile probability must lie in [0, 1]")
    low = np.zeros(len(values), dtype=np.float64)
    high = np.full(len(values), R_MAX_SCALED * R_SCALE, dtype=np.float64)
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        left = posterior_cdf(values, middle) >= p
        high = np.where(left, middle, high)
        low = np.where(left, low, middle)
    return high


def posterior_summary(raw):
    """Return atom probability, posterior mean/std, and central intervals."""
    atom, weights, means, scales = _np_parts(raw)
    a = -means / scales
    b = (R_MAX_SCALED - means) / scales
    pdf_a = np.exp(-0.5 * a * a) / _SQRT2PI
    pdf_b = np.exp(-0.5 * b * b) / _SQRT2PI
    normalization = np.maximum(ndtr(b) - ndtr(a), 1.0e-12)
    shift = (pdf_a - pdf_b) / normalization
    component_mean = means + scales * shift
    component_variance = np.maximum(
        scales * scales * (1.0 + (a * pdf_a - b * pdf_b) / normalization - shift * shift),
        0.0,
    )
    mean = (1.0 - atom) * np.sum(weights * component_mean, axis=1)
    second = (1.0 - atom) * np.sum(weights * (component_variance + component_mean**2), axis=1)
    return {
        "p_zero": atom,
        "r_mean": mean * R_SCALE,
        "r_std": np.sqrt(np.maximum(second - mean**2, 0.0)) * R_SCALE,
        "r_q025": posterior_quantile(raw, 0.025),
        "r_q16": posterior_quantile(raw, 0.16),
        "r_median": posterior_quantile(raw, 0.5),
        "r_q84": posterior_quantile(raw, 0.84),
        "r_q975": posterior_quantile(raw, 0.975),
        "r_upper68": posterior_quantile(raw, 0.68),
        "r_upper95": posterior_quantile(raw, 0.95),
    }
