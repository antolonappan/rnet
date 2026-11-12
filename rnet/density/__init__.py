"""Bounded posterior-density inference for the six-band SAT maps."""

from .model import posterior_cdf, posterior_quantile, posterior_slab_pdf, posterior_summary

__all__ = ["posterior_cdf", "posterior_quantile", "posterior_slab_pdf", "posterior_summary"]
