from __future__ import annotations

import numpy as np
import tensorflow as tf

from .model import BayesianDense


def graph_backbone(model):
    from deepsphere.healpy_networks import HealpyGCNN

    layers = [layer for layer in model.layers if isinstance(layer, HealpyGCNN)]
    if len(layers) != 1:
        raise ValueError("Expected one DeepSphere graph backbone")
    return layers[0]


def initialize_from_deterministic(source, target, posterior_rho=-7.0):
    """Copy backbone and dense means; initialize narrow variational scales.

    The prior remains unchanged. This is an optimization initialization, not an
    empirical-Bayes prior. Source and target must use identical normalization.
    """
    src = [layer for layer in source.layers if isinstance(layer, tf.keras.layers.Dense)]
    dst = [layer for layer in target.layers if isinstance(layer, BayesianDense)]
    if len(src) != len(dst):
        raise ValueError("CNN and BNN dense depths differ")
    for a, b in zip(src, dst):
        if tuple(a.kernel.shape) != tuple(b.kernel_mu.shape) or tuple(a.bias.shape) != tuple(b.bias_mu.shape):
            raise ValueError("CNN and BNN dense shapes differ")
    graph_backbone(target).set_weights(graph_backbone(source).get_weights())
    for a, b in zip(src, dst):
        b.kernel_mu.assign(a.kernel)
        b.bias_mu.assign(a.bias)
        b.kernel_rho.assign(np.full(b.kernel_rho.shape, posterior_rho, dtype=np.float32))
        b.bias_rho.assign(np.full(b.bias_rho.shape, posterior_rho, dtype=np.float32))
