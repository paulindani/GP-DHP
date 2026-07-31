"""JAX re-implementation of the four neural NB baselines (MLP-NB, GRU-NB,
LSTM-NB, and the genuine DeepAR-NB) that mirror the PyTorch forward passes in
``neural_nb.py`` *exactly*, so that a PyTorch ``state_dict`` transfers 1:1 and the
two frameworks agree up to floating-point error (verified by
``scratchpad/verify_neural_jax.py`` to < 1e-6 in float64).

PyTorch conventions reproduced here
-----------------------------------
* ``nn.Linear``:  ``y = x @ W.T + b`` with ``W`` of shape ``(out, in)``.
* ``nn.LSTM``:    packed gate order ``[i, f, g, o]``;
                  ``c = f*c + i*g``,  ``h = o*tanh(c)``.
* ``nn.GRU``:     packed gate order ``[r, z, n]``;
                  ``n = tanh(x_n + r*(h_n))``,  ``h = (1-z)*n + z*h``.
* multi-layer stacking feeds each layer's full hidden sequence to the next; the
  model reads the last layer's final hidden state.  Dropout is identity (eval).

Model heads
-----------
* MLP-NB / GRU-NB / LSTM-NB: ``mu = softplus(head(.)) + EPS``; global scalar
  ``kappa = softplus(raw_kappa) + KAPPA_EPS``.
* DeepAR-NB (genuine DeepAR, Salinas et al. 2020): per-series scale ``nu`` with
  ``mu = nu*softplus(mu_head(.)) + EPS`` and *per-step*
  ``kappa = sqrt(nu)*softplus(kappa_head(.)) + KAPPA_EPS``.

Parameters are supplied as a dict keyed exactly like the PyTorch ``state_dict``
(e.g. ``rnn.weight_ih_l0``, ``net.0.weight``, ``head.3.bias``, ``mu_head.weight``,
``raw_kappa``); values may be numpy or jax arrays.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln

EPS = 1e-6
KAPPA_EPS = 1e-4


def _linear(x, W, b):
    """PyTorch nn.Linear: W has shape (out, in)."""
    return x @ W.T + b


def _dropout(x, rng, dropout):
    """Inverted dropout. Identity when dropout==0 (keep==1 -> all-ones mask), so it
    is safe to apply unconditionally inside a traced graph."""
    keep = 1.0 - dropout
    return x * jax.random.bernoulli(rng, keep, x.shape).astype(x.dtype) / keep


def _rnn_layers(sd, num_layers):
    return [
        dict(W_ih=sd[f"rnn.weight_ih_l{k}"], W_hh=sd[f"rnn.weight_hh_l{k}"],
             b_ih=sd[f"rnn.bias_ih_l{k}"], b_hh=sd[f"rnn.bias_hh_l{k}"])
        for k in range(num_layers)
    ]


def _lstm(layers, x_seq, rng=None, dropout=0.0, train=False):
    """PyTorch nn.LSTM (batch_first). x_seq: (B, T, in) -> last layer final hidden (B, H).
    With ``train=True`` applies dropout to each layer's output sequence except the
    last, matching nn.LSTM's inter-layer dropout.  ``train`` is a Python bool
    (static under jit); the eval path (default) is unchanged."""
    x = jnp.swapaxes(x_seq, 0, 1)                       # (T, B, in)
    B = x_seq.shape[0]
    n_layers = len(layers)
    h_last = None
    for li, lp in enumerate(layers):
        H = lp["W_hh"].shape[1]

        def step(carry, x_t, lp=lp):
            h, c = carry
            g = x_t @ lp["W_ih"].T + lp["b_ih"] + h @ lp["W_hh"].T + lp["b_hh"]
            gi, gf, gg, go = jnp.split(g, 4, axis=-1)   # PyTorch order: input, forget, cell, output
            i = jax.nn.sigmoid(gi)
            f = jax.nn.sigmoid(gf)
            gg = jnp.tanh(gg)
            o = jax.nn.sigmoid(go)
            c = f * c + i * gg
            h = o * jnp.tanh(c)
            return (h, c), h

        (h_T, _), h_seq = jax.lax.scan(step, (jnp.zeros((B, H)), jnp.zeros((B, H))), x)
        x = h_seq                                       # (T, B, H) feeds the next layer
        h_last = h_T
        if train and li < n_layers - 1:
            rng, k = jax.random.split(rng)
            x = _dropout(x, k, dropout)
    return h_last


def _gru(layers, x_seq, rng=None, dropout=0.0, train=False):
    """PyTorch nn.GRU (batch_first). x_seq: (B, T, in) -> last layer final hidden (B, H).
    Inter-layer dropout as in _lstm when ``train=True``."""
    x = jnp.swapaxes(x_seq, 0, 1)
    B = x_seq.shape[0]
    n_layers = len(layers)
    h_last = None
    for li, lp in enumerate(layers):
        H = lp["W_hh"].shape[1]

        def step(h, x_t, lp=lp):
            xg = x_t @ lp["W_ih"].T + lp["b_ih"]
            hg = h @ lp["W_hh"].T + lp["b_hh"]
            xr, xz, xn = jnp.split(xg, 3, axis=-1)      # PyTorch order: reset, update, new
            hr, hz, hn = jnp.split(hg, 3, axis=-1)
            r = jax.nn.sigmoid(xr + hr)
            z = jax.nn.sigmoid(xz + hz)
            n = jnp.tanh(xn + r * hn)
            h = (1.0 - z) * n + z * h
            return h, h

        h_T, h_seq = jax.lax.scan(step, jnp.zeros((B, H)), x)
        x = h_seq
        h_last = h_T
        if train and li < n_layers - 1:
            rng, k = jax.random.split(rng)
            x = _dropout(x, k, dropout)
    return h_last


def _linear_indices(sd, prefix):
    """Sorted numeric indices of the Linear sub-modules under a Sequential prefix."""
    ids = {int(k[len(prefix) + 1:].split(".")[0])
           for k in sd if k.startswith(prefix + ".") and k.endswith(".weight")}
    return sorted(ids)


def mlp_nb(sd, X_mlp, nu=1.0):
    idx = _linear_indices(sd, "net")                    # Linear layers, Tanh between (dropout is identity)
    x = X_mlp
    for j, i in enumerate(idx):
        x = _linear(x, sd[f"net.{i}.weight"], sd[f"net.{i}.bias"])
        if j < len(idx) - 1:
            x = jnp.tanh(x)
    mu = nu * jax.nn.softplus(x[..., 0]) + EPS          # nu=1.0 reproduces the unscaled torch model
    kappa = jax.nn.softplus(sd["raw_kappa"]) + KAPPA_EPS
    return mu, kappa


def rnn_nb(kind, sd, X_seq, target_cov, num_layers, nu=1.0):
    h = (_gru if kind == "GRU" else _lstm)(_rnn_layers(sd, num_layers), X_seq)
    z = jnp.concatenate([h, target_cov], axis=-1)
    idx = _linear_indices(sd, "head")                   # [Linear1, Linear2]; Tanh (+identity dropout) between
    z = jnp.tanh(_linear(z, sd[f"head.{idx[0]}.weight"], sd[f"head.{idx[0]}.bias"]))
    z = _linear(z, sd[f"head.{idx[1]}.weight"], sd[f"head.{idx[1]}.bias"])
    mu = nu * jax.nn.softplus(z[..., 0]) + EPS          # nu=1.0 reproduces the unscaled torch model
    kappa = jax.nn.softplus(sd["raw_kappa"]) + KAPPA_EPS
    return mu, kappa


def deepar_nb(sd, X_seq, target_cov, num_layers, nu):
    h = _lstm(_rnn_layers(sd, num_layers), X_seq)
    z = jnp.concatenate([h, target_cov], axis=-1)
    nu = jnp.asarray(nu)
    mu = nu * jax.nn.softplus(_linear(z, sd["mu_head.weight"], sd["mu_head.bias"])[..., 0]) + EPS
    kappa = jnp.sqrt(nu) * jax.nn.softplus(_linear(z, sd["kappa_head.weight"], sd["kappa_head.bias"])[..., 0]) + KAPPA_EPS
    return mu, kappa


@partial(jax.jit, static_argnums=(0, 5))                # model (0) and num_layers (5) are static
def apply_model(model, sd, X_mlp, X_seq, target_cov, num_layers, nu=1.0):
    """Dispatch to the model matching PyTorch's canonical name. Returns (mu, kappa).

    Jitted: ``model`` and ``num_layers`` are static -- they fix the architecture
    and the (compile-time-unrolled) 1-2 layer stack -- while ``sd`` is a parameter
    pytree and the inputs/``nu`` are traced.  The heavy per-timestep recurrence
    runs as a single ``lax.scan``; the only Python loops (over the small, static,
    heterogeneously-shaped layer stack) unroll once at trace time and cost nothing
    at run time.  Re-compiles only when the model, layer count, or input
    shapes/dtypes change -- not per weight update."""
    if model == "MLP-NB":
        return mlp_nb(sd, X_mlp, nu)
    if model == "GRU-NB":
        return rnn_nb("GRU", sd, X_seq, target_cov, num_layers, nu)
    if model == "LSTM-NB":
        return rnn_nb("LSTM", sd, X_seq, target_cov, num_layers, nu)
    if model == "DeepAR-NB":
        return deepar_nb(sd, X_seq, target_cov, num_layers, nu)
    raise ValueError(f"Unknown model: {model}")


@partial(jax.jit, static_argnums=(0, 5))                # model (0) and num_layers (5) are static
def apply_train(model, sd, X_mlp, X_seq, target_cov, num_layers, rng, dropout, nu=1.0):
    """Training-mode forward with dropout applied at the PyTorch sites (after every
    MLP/head Tanh, and between RNN layers).  Identical to ``apply_model`` when
    dropout==0.  ``rng`` seeds the dropout masks; ``dropout`` is traced."""
    if model == "MLP-NB":
        idx = _linear_indices(sd, "net")
        x = X_mlp
        for j, i in enumerate(idx):
            x = _linear(x, sd[f"net.{i}.weight"], sd[f"net.{i}.bias"])
            if j < len(idx) - 1:
                x = jnp.tanh(x)
                rng, k = jax.random.split(rng)
                x = _dropout(x, k, dropout)              # torch MLP: Dropout after each Tanh
        mu = nu * jax.nn.softplus(x[..., 0]) + EPS
        kappa = jax.nn.softplus(sd["raw_kappa"]) + KAPPA_EPS
        return mu, kappa
    if model == "DeepAR-NB":
        rng, kr = jax.random.split(rng)
        h = _lstm(_rnn_layers(sd, num_layers), X_seq, kr, dropout, train=True)
        z = jnp.concatenate([h, target_cov], axis=-1)
        nu = jnp.asarray(nu)
        mu = nu * jax.nn.softplus(_linear(z, sd["mu_head.weight"], sd["mu_head.bias"])[..., 0]) + EPS
        kappa = jnp.sqrt(nu) * jax.nn.softplus(_linear(z, sd["kappa_head.weight"], sd["kappa_head.bias"])[..., 0]) + KAPPA_EPS
        return mu, kappa
    kind = "GRU" if model == "GRU-NB" else "LSTM"
    rng, kr = jax.random.split(rng)
    h = (_gru if kind == "GRU" else _lstm)(_rnn_layers(sd, num_layers), X_seq, kr, dropout, train=True)
    z = jnp.concatenate([h, target_cov], axis=-1)
    idx = _linear_indices(sd, "head")
    z = jnp.tanh(_linear(z, sd[f"head.{idx[0]}.weight"], sd[f"head.{idx[0]}.bias"]))
    rng, k = jax.random.split(rng)
    z = _dropout(z, k, dropout)                          # torch RNN head: Dropout after the Tanh
    z = _linear(z, sd[f"head.{idx[1]}.weight"], sd[f"head.{idx[1]}.bias"])
    mu = nu * jax.nn.softplus(z[..., 0]) + EPS
    kappa = jax.nn.softplus(sd["raw_kappa"]) + KAPPA_EPS
    return mu, kappa


@jax.jit
def nb_logpmf(y, mu, kappa):
    """Negative-binomial log-pmf (mean mu, size kappa), matching neural_nb.nb_logpmf_np."""
    mu = jnp.maximum(mu, EPS)
    k = jnp.maximum(kappa, KAPPA_EPS)
    return (gammaln(y + k) - gammaln(k) - gammaln(y + 1.0)
            + k * (jnp.log(k) - jnp.log(k + mu)) + y * (jnp.log(mu) - jnp.log(k + mu)))
