"""Collapsed-latent MAP solve for the GP-DHP designs (used by the synthetic recovery experiments).

This module previously also drove a posterior sampler for a fully Bayesian GP-DHP variant that
has been removed from the paper.  Only `find_map` is used by live code
(synthetic_experiments.py), so only it is kept here.
"""
import os
import sys

import jax.numpy as jnp
import jax

jax.config.update("jax_enable_x64", True)
_HERE = os.path.dirname(os.path.abspath(__file__))                    # papercode/library
from .lbfgs_mt import minimize_lbfgs_mt   # on-device L-BFGS (Moré-Thuente)


def find_map(nlp, q, theta0):
    """MAP of the collapsed-latent negative log-posterior by on-device L-BFGS-MT.

    Autodiff gradients on the JAX `nlp`; the high-precision budget matches the former scipy
    L-BFGS-B settings without the per-iteration host<->device roundtrip.  Returns (theta*, nlp*).
    """
    res = minimize_lbfgs_mt(nlp, jnp.asarray(theta0), maxiter=20000, ftol=1e-14, gtol=1e-10)
    return res.x, float(res.fun)
