from __future__ import annotations

import numpy as np
import tensorflow as tf

from .config import BNNModelConfig


def gaussian_nll(y_true, y_pred, beta: float = 0.0, signed_null_penalty: float = 0.0):
    """Gaussian NLL for scaled r, optionally weighted and zero-centered."""
    mu = y_pred[..., :1]
    log_variance = tf.clip_by_value(y_pred[..., 1:2], -30.0, 5.0)
    nll = 0.5 * (log_variance + tf.square(y_true - mu) * tf.exp(-log_variance))
    if beta:
        nll *= tf.stop_gradient(tf.exp(float(beta) * log_variance))
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


class BayesianDense(tf.keras.layers.Layer):
    """Mean-field variational dense layer with an analytic Gaussian KL term."""

    def __init__(self, units: int, kl_weight: float, prior_scale: float = 1.0, activation=None, **kwargs):
        super().__init__(**kwargs)
        self.units = int(units)
        self.kl_weight = float(kl_weight)
        self.prior_scale = float(prior_scale)
        self.activation = tf.keras.activations.get(activation)

    def build(self, input_shape):
        input_dim = int(input_shape[-1])
        initializer = tf.keras.initializers.GlorotUniform()
        self.kernel_mu = self.add_weight(
            name="kernel_mu", shape=(input_dim, self.units), initializer=initializer, trainable=True
        )
        self.kernel_rho = self.add_weight(
            name="kernel_rho", shape=(input_dim, self.units), initializer=tf.keras.initializers.Constant(-5.0), trainable=True
        )
        self.bias_mu = self.add_weight(
            name="bias_mu", shape=(self.units,), initializer="zeros", trainable=True
        )
        self.bias_rho = self.add_weight(
            name="bias_rho", shape=(self.units,), initializer=tf.keras.initializers.Constant(-5.0), trainable=True
        )
        super().build(input_shape)

    def _kl_normal(self, mu, rho):
        sigma = tf.nn.softplus(rho) + 1.0e-6
        prior_var = tf.cast(self.prior_scale ** 2, sigma.dtype)
        return 0.5 * tf.reduce_sum((tf.square(sigma) + tf.square(mu)) / prior_var - 1.0 + tf.math.log(prior_var) - 2.0 * tf.math.log(sigma))

    def call(self, inputs):
        kernel_sigma = tf.nn.softplus(self.kernel_rho) + 1.0e-6
        bias_sigma = tf.nn.softplus(self.bias_rho) + 1.0e-6
        kernel = self.kernel_mu + kernel_sigma * tf.random.normal(tf.shape(self.kernel_mu))
        bias = self.bias_mu + bias_sigma * tf.random.normal(tf.shape(self.bias_mu))
        self.add_loss(self.kl_weight * (self._kl_normal(self.kernel_mu, self.kernel_rho) + self._kl_normal(self.bias_mu, self.bias_rho)))
        output = tf.linalg.matmul(inputs, kernel) + bias
        return self.activation(output) if self.activation is not None else output


def build_model(config: BNNModelConfig, pixel_indices: np.ndarray, kl_weight: float):
    """Build a DeepSphere CNN with a variational Bayesian regression head."""
    indices = np.asarray(pixel_indices, dtype=np.int64)
    regularizer = (
        tf.keras.regularizers.l2(config.l2_regularization)
        if config.l2_regularization > 0
        else None
    )

    layers = []
    for level, (order, features) in enumerate(config.conv_blocks):
        from deepsphere.healpy_layers import HealpyChebyshev, HealpyPseudoConv

        layers.append(
            HealpyChebyshev(
                K=int(order), Fout=int(features), activation="relu", use_bias=True,
                use_bn=True, regularizer=regularizer,
            )
        )
        if level < len(config.pool_features):
            layers.append(
                HealpyPseudoConv(
                    p=1, Fout=int(config.pool_features[level]), activation="relu", use_bias=True
                )
            )

    from deepsphere.healpy_networks import HealpyGCNN

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
        second = tf.keras.layers.Lambda(
            lambda value: tf.reduce_mean(tf.square(value), axis=1), name="spatial_second_moment"
        )(features)
        mean_square = tf.keras.layers.Lambda(
            lambda value: tf.square(value), name="spatial_mean_square"
        )(mean)
        pooled.append(tf.keras.layers.Subtract(name="spatial_variance")([second, mean_square]))
    if "max" in config.pooling:
        pooled.append(tf.keras.layers.GlobalMaxPooling1D(name="spatial_maximum")(features))
    features = tf.keras.layers.Concatenate(name="global_feature_statistics")(pooled)
    if config.dropout > 0:
        features = tf.keras.layers.Dropout(config.dropout)(features)

    for index, units in enumerate(config.bayesian_dense_features):
        features = BayesianDense(
            int(units), kl_weight=kl_weight, prior_scale=config.prior_scale,
            activation="relu", name=f"bayesian_dense_{index + 1}"
        )(features)
    outputs = BayesianDense(
        2, kl_weight=kl_weight, prior_scale=config.prior_scale,
        name="r_mean_log_variance"
    )(features)
    return tf.keras.Model(inputs, outputs, name="rnet_deepsphere_bnn")


def mc_prediction_to_r(
    predictions: np.ndarray,
    target_scale: float = 1.0e-2,
    calibration_scale: float = 1.0,
):
    """Combine MC samples into mean, aleatoric, epistemic, and total sigma(r)."""
    values = np.asarray(predictions, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 2:
        raise ValueError("predictions must have shape (mc_samples, batch, 2)")
    means = values[..., 0]
    aleatoric_variance = np.exp(0.5 * np.clip(values[..., 1], -30.0, 5.0) * 2.0)
    mean_scaled = np.mean(means, axis=0)
    epistemic_variance = np.var(means, axis=0)
    total_variance = np.mean(aleatoric_variance, axis=0) + epistemic_variance
    scale = float(target_scale)
    return {
        "r_predicted": mean_scaled * scale,
        "sigma_r": np.sqrt(np.maximum(total_variance, 1.0e-20)) * scale * float(calibration_scale),
        "sigma_aleatoric": np.sqrt(np.mean(aleatoric_variance, axis=0)) * scale,
        "sigma_epistemic": np.sqrt(np.maximum(epistemic_variance, 0.0)) * scale,
        "mean_scaled": mean_scaled,
        "log_variance_mean": np.mean(values[..., 1], axis=0),
    }
