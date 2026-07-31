"""All-JAX count-process baselines (parametric DHP family, Histogram DHP-NB,
Baseline-only, NB-INGARCH), numerically validated to machine precision under the
hard-ReLU link, then extended with a tunable softplus link selected by 40% held-out forward
validation (multi-start L-BFGS-MT on the analytic bilevel hypergradient),
consistent with GP-DHP and the neural models.

Design: the differentiable objectives (NB2 log-likelihood, kernels, baseline
designs, links) are JAX; the lag matrices are plain numpy (they depend only on
the counts, not the parameters); fitting is fully on-device L-BFGS / BFGS with a
Moré-Thuente line search (lbfgs_mt), interval constraints via smooth
reparametrization. Each solver jits once (problem structure static, data/hypers
traced) and is reused across the inner multi-starts and every FV evaluation. float64.

This module is a library; jax_baselines_runner.py drives it per (dataset, variant).
"""
import os
import sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.special import gammaln
from functools import partial
from typing import NamedTuple

from .lbfgs_mt import (minimize_lbfgs_mt_box, minimize_bfgs_mt,   # on-device L-BFGS/BFGS (Moré-Thuente)
                      minimize_lbfgs_mt, _box_transforms)         # + unconstrained solve & box reparam (FV bilevel)

FLOOR = 1e-6                                                    # fc_lambda_floor()


# --------------------------------------------------------------------------- #
#  shared primitives (NB2 scoring, kernels, baseline designs)      #
# --------------------------------------------------------------------------- #
@jax.jit
def nb2_logpmf(y, mu, kappa):
    """dnbinom(y, size=kappa, mu=mu, log=TRUE) with the mu>=floor guard."""
    mu = jnp.maximum(mu, FLOOR)
    return (gammaln(y + kappa) - gammaln(kappa) - gammaln(y + 1.0)
            + kappa * jnp.log(kappa / (kappa + mu)) + y * jnp.log(mu / (kappa + mu)))


def link_mu(eta, link="relu", link_scale=0.1):
    """fc_link_mu: hard rectifier (paper default) or smooth softplus (temperature
    link_scale); softplus(eta) -> max(eta,0) as link_scale -> 0. Adds the floor."""
    if link == "softplus":
        z = eta / link_scale
        sp = jnp.where(z > 30.0, eta, link_scale * jnp.log1p(jnp.exp(jnp.minimum(z, 30.0))))
        return sp + FLOOR
    return jnp.maximum(eta, 0.0) + FLOOR


def dhp_lag_matrix(y, D_max):
    """X[t,d] = y[t-d] for d<t else 0 (1-indexed t,d). Matches gpdhp_model.R."""
    y = np.asarray(y, float); T = len(y)
    X = np.zeros((T, D_max))
    for d in range(1, min(D_max, T - 1) + 1):
        X[d:, d - 1] = y[:T - d]
    return X


def dhp_lag_matrix_future(y_hist, y_fut, D_max):
    """Future row i (time t=n_hist+i): X[i,d] = y_all[t-d] when t-d>=1."""
    y_all = np.concatenate([np.asarray(y_hist, float), np.asarray(y_fut, float)])
    nh, nf = len(y_hist), len(y_fut)
    X = np.zeros((nf, D_max))
    for d0 in range(D_max):
        src = nh + np.arange(nf) - d0 - 1                       # 0-indexed source into y_all
        ok = src >= 0
        X[ok, d0] = y_all[src[ok]]
    return X


def fv_split(n, frac=0.4, min_train=0.6):
    """First 60% train / last 40% validation index split (indices into 0..n-1)."""
    ncut = int(np.floor((1.0 - frac) * n))
    return np.arange(ncut), np.arange(ncut, n)


# --------------------------------------------------------------------------- #
#  parametric DHP family                                                       #
# --------------------------------------------------------------------------- #
PARAM_NAMES = {"DHP": ["mu", "r", "p", "eta"],
               "LinearDHP": ["mu", "mu1", "r", "p", "eta"],
               "DHPsinusoidal": ["mu", "A1", "r", "p", "eta"],
               "DHPsinusoidalLinear": ["mu", "mu1", "A1", "r", "p", "eta"]}
PARAM_LABEL = {"DHP": "Discrete DHP", "LinearDHP": "Linear DHP",
               "DHPsinusoidal": "Sinusoidal DHP", "DHPsinusoidalLinear": "Linear + Sinusoidal DHP"}


def _param_bounds_starts(model, y_dev, ncov):
    m = max(float(np.mean(y_dev)), 1e-4)
    base = {"mu": (m, 1e-9, None), "mu1": (1e-4, None, None), "A1": (0.0, None, None),
            "r": (6.0, 1e-9, None), "p": (0.3, 1e-9, 1 - 1e-6), "eta": (0.5, 0.0, 1.25)}
    names = PARAM_NAMES[model]
    default = np.array([base[k][0] for k in names])
    lower = [base[k][1] for k in names]; upper = [base[k][2] for k in names]
    # multi-start (parametric_candidate_starts): default, low_exc, high_exc
    def mk(over):
        s = dict(zip(names, default))
        s.update(over); return np.array([s[k] for k in names])
    starts = {"default": default}
    if model == "DHP":
        starts["low_exc"] = mk({"mu": 0.8 * m, "r": 2, "p": 0.2, "eta": 0.2})
        starts["high_exc"] = mk({"mu": 0.3 * m, "r": 10, "p": 0.6, "eta": 0.9})
    elif model == "LinearDHP":
        starts["low_exc"] = mk({"mu": 0.8 * m, "mu1": 0, "r": 2, "p": 0.2, "eta": 0.2})
        starts["high_exc"] = mk({"mu": 0.3 * m, "mu1": 1e-4, "r": 10, "p": 0.6, "eta": 0.9})
    elif model == "DHPsinusoidal":
        starts["low_exc"] = mk({"mu": 0.8 * m, "A1": 0, "r": 2, "p": 0.2, "eta": 0.2})
        starts["high_exc"] = mk({"mu": 0.3 * m, "A1": 0, "r": 10, "p": 0.6, "eta": 0.9})
    else:
        starts["low_exc"] = mk({"mu": 0.8 * m, "mu1": 0, "A1": 0, "r": 2, "p": 0.2, "eta": 0.2})
        starts["high_exc"] = mk({"mu": 0.3 * m, "mu1": 1e-4, "A1": 0, "r": 10, "p": 0.6, "eta": 0.9})
    # covariate coeffs: free, init 0 (unregularized)
    if ncov:
        lower += [None] * ncov; upper += [None] * ncov
        for k in starts:
            starts[k] = np.concatenate([starts[k], np.zeros(ncov)])
    bounds = list(zip(lower, upper))
    return bounds, starts, names


@partial(jax.jit, static_argnums=(2, 3, 4))
def _param_mu(theta, kappa, model, ncov, is_future, X, t_idx, period, Zcov):
    names = PARAM_NAMES[model]
    par = {k: theta[i] for i, k in enumerate(names)}
    r, p, eta = par["r"], par["p"], par["eta"]
    d = jnp.arange(X.shape[1], dtype=float)
    logk = gammaln(d + r) - gammaln(r) - gammaln(d + 1.0) + r * jnp.log(1 - p) + d * jnp.log(p)
    kernel = eta * jnp.exp(logk)                               # eta * dnbinom(0:D-1, r, 1-p)
    base = par["mu"] * jnp.ones_like(t_idx)
    if "mu1" in par:
        base = base + par["mu1"] * t_idx
    if "A1" in par:
        base = base + par["A1"] * jnp.sin(2 * jnp.pi * t_idx / period)
    if ncov:
        base = base + Zcov @ theta[len(names):]
    return base + X @ kernel                                   # pre-link eta


@partial(jax.jit, static_argnums=(7, 8, 9, 10, 11, 12))
def _param_solve(x0, X, y, t_idx, Zcov, link_scale, kappa,
                 model, ncov, nmean, period, link, bounds):
    """One box-constrained L-BFGS-MT fit of the parametric mean/kernel params at a
    FIXED kappa (a forward-validation hyperparameter). Jitted (structure static, data/hypers traced)
    so it compiles once and is reused across multi-starts. Returns (theta*, nll*)."""
    def nll(par):
        eta = _param_mu(par, kappa, model, ncov, False, X, t_idx, period, Zcov)
        return -jnp.sum(nb2_logpmf(y, link_mu(eta, link, link_scale), kappa))
    res = minimize_lbfgs_mt_box(nll, x0, bounds, maxiter=2500, ftol=1e-10, gtol=1e-8)
    return res.x, res.fun


def fit_parametric(model, y_dev, period, D_max=100, Zcov_dev=None, kappa=1.0,
                   link="relu", link_scale=0.1):
    """Fit the mean/kernel params by box-constrained L-BFGS-MT from the multi-start set
    at a FIXED kappa (an forward-validation hyperparameter), keeping the highest development NB
    log-likelihood.  Used for the final refit-at-selected-hypers before scoring the test."""
    ncov = 0 if Zcov_dev is None else Zcov_dev.shape[1]
    y_dev = np.asarray(y_dev, float); T = len(y_dev)
    X = dhp_lag_matrix(y_dev, D_max); t_idx = np.arange(1, T + 1, dtype=float)
    bounds, starts, names = _param_bounds_starts(model, y_dev, ncov)
    nmean = len(names)
    bt = tuple((None if lo is None else float(lo), None if hi is None else float(hi)) for lo, hi in bounds)
    Xj, yj, tj = jnp.asarray(X), jnp.asarray(y_dev), jnp.asarray(t_idx)
    Zj = jnp.asarray(Zcov_dev) if ncov else jnp.zeros((T, 0))
    ls, kf = jnp.asarray(float(link_scale)), jnp.asarray(float(kappa))
    best = None
    for x0 in starts.values():
        xstar, fval = _param_solve(jnp.asarray(np.asarray(x0, float)), Xj, yj, tj, Zj, ls, kf,
                                   model, ncov, nmean, float(period), link, bt)
        if best is None or -float(fval) > best[0]:
            best = (-float(fval), np.asarray(xstar))
    dev_ll, theta_full = best
    return dict(model=model, theta=np.asarray(theta_full[:nmean + ncov]), kappa=float(kappa),
                names=names, ncov=ncov, link=link, link_scale=link_scale, period=period, D_max=D_max)


def score_parametric(fit, y_dev, y_test, Zcov_test=None):
    """Per-obs test NB log-scores for a fitted parametric model."""
    y_dev = np.asarray(y_dev, float); y_test = np.asarray(y_test, float)
    D_max, period, model, ncov = fit["D_max"], fit["period"], fit["model"], fit["ncov"]
    Xt = dhp_lag_matrix_future(y_dev, y_test, D_max)
    t_glob = len(y_dev) + np.arange(1, len(y_test) + 1, dtype=float)
    Zt = np.zeros((len(y_test), 0)) if ncov == 0 else np.asarray(Zcov_test, float)
    eta = _param_mu(jnp.asarray(fit["theta"]), fit["kappa"], model, ncov, True,
                    jnp.asarray(Xt), jnp.asarray(t_glob), period, jnp.asarray(Zt))
    mu = link_mu(eta, fit["link"], fit["link_scale"])
    return np.asarray(nb2_logpmf(jnp.asarray(y_test), mu, fit["kappa"]))


# --------------------------------------------------------------------------- #
#  Histogram DHP-NB                                                             #
# --------------------------------------------------------------------------- #
def hist_bin_lags(kind, D_max=100):
    if kind == "coarse_log":
        z = [[1], [2, 3], list(range(4, 8)), list(range(8, 15)), list(range(15, 31)),
             list(range(31, 61)), list(range(61, 101))]
    elif kind == "medium_log":
        z = [[1], [2], [3, 4], [5, 6, 7], list(range(8, 15)), list(range(15, 31)),
             list(range(31, 61)), list(range(61, 101))]
    elif kind == "equal_width":
        z = [list(range(a, a + 10)) for a in range(1, 92, 10)]
    else:
        raise ValueError(f"Unknown bin kind: {kind}")
    return [[d for d in v if 1 <= d <= D_max] for v in z]


def hist_make_sums(y, bins):
    """Xh_raw[t,m] = sum_{d in bin_m} y[t-d]. Matches hist_dhp_nb_make_sums."""
    y = np.asarray(y, float); n = len(y); X = np.zeros((n, len(bins)))
    for m, lags in enumerate(bins):
        for d in lags:
            if d < n:
                X[d:, m] += y[:n - d]
    return X


def hist_baseline_design(cal_index, period, form, dates=None):
    """intercept / trend=scale(t) / season / dow. Matches hist_dhp_nb_baseline_design."""
    t = np.asarray(cal_index, float); n = len(t)
    cols = [np.ones(n)]
    sd = t.std(ddof=1)
    trend = (t - t.mean()) / sd if (np.isfinite(sd) and sd > 0) else np.linspace(-1, 1, n)
    if not np.all(np.isfinite(trend)):
        trend = np.linspace(-1, 1, n)
    if form in ("linear", "linear_seasonal", "linear_seasonal_dow"):
        cols.append(trend)
    if form in ("seasonal", "linear_seasonal", "seasonal_dow", "linear_seasonal_dow"):
        cols += [np.sin(2 * np.pi * t / period), np.cos(2 * np.pi * t / period)]
    if form in ("seasonal_dow", "linear_seasonal_dow"):
        if dates is not None:
            dow = np.asarray(pd_wday(dates), float)
        else:
            dow = (np.arange(n)) % 7
        cols += [np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)]
    return np.column_stack(cols)


def pd_wday(dates):
    import pandas as pd
    return pd.to_datetime(pd.Series(dates)).dt.dayofweek.to_numpy()  # Mon=0..Sun=6


def _hist_candidate(Xb, Xh_raw, y, fit_idx, score_idx, pen, kappa, link="relu", link_scale=0.1):
    """One candidate at a FIXED kappa: recompute h_scale on fit_idx, scale Xh, fit
    (beta, theta_scaled) on fit_idx, score score_idx.  Returns (score_logscores, kappa, nll*)."""
    h_scale = Xh_raw[fit_idx].std(axis=0, ddof=1)
    h_scale[~np.isfinite(h_scale) | (h_scale <= 0)] = 1.0
    Xh = Xh_raw / h_scale
    params, kap, fval = _hist_fit_block(Xb, Xh, h_scale, y, fit_idx, pen, kappa, link, link_scale)
    ls = _hist_score(Xb, Xh, params, score_idx, y, kap, link, link_scale)
    return ls, kap, fval


@partial(jax.jit, static_argnums=(8, 9, 10, 11))
def _hist_solve(x0, Xbf, Xhf, yf, hs, pen, kappa, link_scale, nb, nh, link, bounds):
    """Fit (beta, theta_scaled) at a FIXED kappa; returns (params*, nll*)."""
    def nll(p):
        eta = Xbf @ p[:nb] + Xhf @ p[nb:nb + nh]
        return -jnp.sum(nb2_logpmf(yf, link_mu(eta, link, link_scale), kappa)) \
            + pen * jnp.sum((p[nb:nb + nh] / hs) ** 2)
    res = minimize_lbfgs_mt_box(nll, x0, bounds, maxiter=300, ftol=1e-8, gtol=1e-5)
    return res.x, res.fun


def _hist_fit_block(Xb, Xh, h_scale, y, fit_idx, pen, kappa, link, link_scale):
    """Box-constrained L-BFGS-MT fit of (beta, theta_scaled) on fit_idx at a FIXED
    kappa with ridge pen*sum(theta^2), theta=theta_scaled/h_scale. theta_scaled >= 0
    (nonneg excitation); beta free. Returns (params, kappa, nll*)."""
    nb, nh = Xb.shape[1], Xh.shape[1]
    Xbf, Xhf, yf = jnp.asarray(Xb[fit_idx]), jnp.asarray(Xh[fit_idx]), jnp.asarray(y[fit_idx])
    hs = jnp.asarray(h_scale)
    bounds = tuple([(None, None)] * nb + [(0.0, None)] * nh)
    fit_y = y[fit_idx]
    main = np.array([max(float(np.mean(fit_y)), 1e-3)] + [0.0] * (nb - 1) + [0.0] * nh)
    small = main.copy(); small[nb:nb + nh] = np.maximum(1e-4 * h_scale, 1e-8)
    pen_j, ls_j, kf = jnp.asarray(float(pen)), jnp.asarray(float(link_scale)), jnp.asarray(float(kappa))
    best = None
    for x0 in (main, small):
        xstar, fval = _hist_solve(jnp.asarray(x0), Xbf, Xhf, yf, hs, pen_j, kf, ls_j, nb, nh, link, bounds)
        if best is None or float(fval) < best[1]:
            best = (np.asarray(xstar), float(fval))
    p, fval = best
    return p, float(kappa), fval


def _hist_score(Xb, Xh, params, score_idx, y, kappa, link="relu", link_scale=0.1):
    nb, nh = Xb.shape[1], Xh.shape[1]
    beta = params[:nb]; th_s = params[nb:nb + nh]
    mu = link_mu(jnp.asarray(Xb[score_idx] @ beta + Xh[score_idx] @ th_s), link, link_scale)
    return np.asarray(nb2_logpmf(jnp.asarray(y[score_idx]), mu, kappa))


HIST_CFG = {False: dict(bin="medium_log", base="seasonal"),      # weekly
            True: dict(bin="equal_width", base="seasonal_dow")}  # daily
HIST_PEN_GRID = np.concatenate([[0.0], 2.0 ** np.arange(-20, 11)])   # 0, 2^-20..2^10 (32)


def _hist_design(y_dev, y_test, period, daily, cal_index, dates, Zcov_full, D_max):
    cfg = HIST_CFG[daily]
    y = np.concatenate([np.asarray(y_dev, float), np.asarray(y_test, float)])
    Xh_raw = hist_make_sums(y, hist_bin_lags(cfg["bin"], D_max))
    Xb = hist_baseline_design(cal_index, period, cfg["base"], dates)
    if Zcov_full is not None:
        Xb = np.column_stack([Xb, np.asarray(Zcov_full, float)])   # unregularized cov cols
    return y, Xb, Xh_raw


# --------------------------------------------------------------------------- #
#  Baseline-only (GP-DHP baseline design, no excitation)                        #
# --------------------------------------------------------------------------- #
def baseline_scaled_matrix(t_idx, fixed, period, Zcov=None, cov_scale=10.0):
    """level (sigma_level) + Fourier harmonics (sigma_fourier) + optional linear
    (sigma_lin) + optional scaled covariates. Matches gpdhp_model.R."""
    t = np.asarray(t_idx, float); n = len(t)
    parts = [np.full((n, 1), fixed["sigma_level"])]
    K = int(fixed["K_fourier"]); H = []
    for k in range(1, K + 1):
        H.append(np.sin(2 * np.pi * k * t / period)); H.append(np.cos(2 * np.pi * k * t / period))
    parts.append(fixed["sigma_fourier"] * np.column_stack(H))
    if fixed.get("sigma_lin", 0) > 0:
        parts.append((fixed["sigma_lin"] * t).reshape(-1, 1))
    if Zcov is not None:
        parts.append(cov_scale * np.asarray(Zcov, float))
    return np.column_stack(parts)


@partial(jax.jit, static_argnums=(3,))
def _bo_solve(x0, B, y, link, link_scale, kappa):
    """NB2 MAP over theta with unit ridge at a FIXED kappa; returns (theta*, nll*)."""
    def nll(th):
        return -jnp.sum(nb2_logpmf(y, link_mu(B @ th, link, link_scale), kappa)) + 0.5 * jnp.sum(th ** 2)
    res = minimize_bfgs_mt(nll, x0, maxiter=900, gtol=1e-8)
    return res.x, res.fun


def _bo_fit(B, y, kappa=1.0, link="relu", link_scale=0.1):
    """NB2 MAP with unit ridge 0.5*sum(theta^2), full BFGS-MT (unbounded, as R), at a
    FIXED kappa (an forward-validation hyperparameter). Returns (theta, kappa, nll*)."""
    q = B.shape[1]
    init = np.linalg.solve(B.T @ B + np.eye(q), B.T @ np.maximum(y, 1e-3))
    xstar, fval = _bo_solve(jnp.asarray(init), jnp.asarray(B), jnp.asarray(y), link,
                            jnp.asarray(float(link_scale)), jnp.asarray(float(kappa)))
    return np.asarray(xstar), float(kappa), float(fval)


def fit_score_baseline_only(y_dev, y_test, period, fixed, kappa, Zcov_dev=None, Zcov_test=None,
                            link="relu", link_scale=0.1, cov_scale=10.0):
    """Fit baseline-only at FIXED design hypers + kappa on full dev; return (per-obs test
    log-scores, kappa, nll_at_map). Used for the final refit-at-selected-hypers scoring."""
    y_dev = np.asarray(y_dev, float); y_test = np.asarray(y_test, float); T = len(y_dev)
    B_fit = baseline_scaled_matrix(np.arange(1, T + 1), fixed, period, Zcov_dev, cov_scale)
    theta, kap, fval = _bo_fit(B_fit, y_dev, kappa, link, link_scale)
    B_test = baseline_scaled_matrix(T + np.arange(1, len(y_test) + 1), fixed, period, Zcov_test, cov_scale)
    mu = link_mu(jnp.asarray(B_test @ theta), link, link_scale)
    return np.asarray(nb2_logpmf(jnp.asarray(y_test), mu, kap)), kap, fval


# --------------------------------------------------------------------------- #
#  NB-INGARCH (clean JAX port; comparable to tscount::tsglm, not bit-identical) #
# --------------------------------------------------------------------------- #
# tscount uses QMLE for the mean params + a separate dispersion estimate and a
# package-specific recursion initialisation; this port instead fits the whole
# INGARCH recursion (identity link) by JOINT NB2 maximum likelihood via lax.scan.
# Same model family and internal covariates, validated to give comparable pLL.
ING_CFG = {False: dict(past_obs=[1, 4], past_mean=[1], cov="trend"),        # weekly
           True: dict(past_obs=[1, 7], past_mean=[1], cov="annual_dow")}    # daily


def ingarch_covariates(cal_index, period, kind, dates=None):
    """Internal xreg, matching nb_ingarch_covariates (trend / annual / annual_dow;
    shifted so the seasonal terms are nonneg for the identity link)."""
    t = np.asarray(cal_index, float); n = len(t); mats = []
    if kind == "trend":
        mats.append((t - t.min()) / max(1.0, t.max() - t.min()))
    if kind in ("annual", "annual_dow"):
        mats += [1 + np.sin(2 * np.pi * t / period), 1 + np.cos(2 * np.pi * t / period)]
    if kind == "annual_dow":
        dow = pd_wday(dates) if dates is not None else np.arange(n) % 7
        mats += [1 + np.sin(2 * np.pi * dow / 7), 1 + np.cos(2 * np.pi * dow / 7)]
    return np.column_stack(mats) if mats else None


@partial(jax.jit, static_argnums=(4, 5))
def _ingarch_lambda(params, y, X, mu0, po, pm):
    """lambda_t = beta0 + sum_i beta_i y_{t-i} + sum_j alpha_j lambda_{t-j} + eta'X_t
    (identity link); pre-sample y and lambda set to mu0. po/pm are static tuples."""
    npo, npm, nx = len(po), len(pm), X.shape[1]
    beta0 = params[0]
    betas = params[1:1 + npo]
    alphas = params[1 + npo:1 + npo + npm]
    etas = params[1 + npo + npm:1 + npo + npm + nx]
    c = beta0 + (X @ etas if nx else 0.0)
    for bi, lag in zip(betas, po):
        yl = jnp.concatenate([jnp.full(lag, mu0), y[:-lag]])   # y_{t-lag}, presample mu0
        c = c + bi * yl
    qmax = max(pm)

    def step(carry, ct):                                       # carry = [lam_{t-1},...,lam_{t-qmax}]
        lam = ct + sum(alphas[k] * carry[pm[k] - 1] for k in range(npm))
        return jnp.concatenate([lam[None], carry[:-1]]), lam
    _, lam = jax.lax.scan(step, jnp.full(qmax, mu0), c)
    return lam


def fit_score_ingarch(y_dev, y_test, period, daily, cal_index, dates=None, Zcov_full=None, kappa=1.0):
    """Fit the NB-INGARCH mean params on the dev period at a FIXED kappa (a forward-validation
    hyperparameter), score one-step-ahead (true newobs) on the test period.
    cal_index/dates cover the full series. Used for the refit-at-selected-hypers scoring."""
    cfg = ING_CFG[daily]; po, pm, kind = cfg["past_obs"], cfg["past_mean"], cfg["cov"]
    y = np.concatenate([np.asarray(y_dev, float), np.asarray(y_test, float)])
    T = len(y_dev); mu0 = float(np.mean(y_dev))
    Xint = ingarch_covariates(cal_index, period, kind, dates)
    X = Xint if Zcov_full is None else (np.asarray(Zcov_full, float) if Xint is None
                                        else np.column_stack([Xint, np.asarray(Zcov_full, float)]))
    if X is None:
        X = np.zeros((len(y), 0))
    npo, npm, nx = len(po), len(pm), X.shape[1]
    y_fit = jnp.asarray(y[:T]); X_fit = jnp.asarray(X[:T])
    po_t, pm_t = tuple(po), tuple(pm); kap = jnp.asarray(float(kappa))

    def eta_fn(par):
        return _ingarch_lambda(par, y_fit, X_fit, mu0, po_t, pm_t)
    def nll(par):
        return -jnp.sum(nb2_logpmf(y_fit, jnp.maximum(eta_fn(par), FLOOR), kap))

    x0 = np.concatenate([[0.5 * mu0], np.full(npo, 0.2), np.full(npm, 0.2), np.zeros(nx)])
    bounds = tuple([(1e-6, None)] + [(0.0, None)] * npo + [(0.0, 0.999)] * npm + [(None, None)] * nx)
    r = minimize_lbfgs_mt_box(nll, jnp.asarray(x0), bounds, maxiter=500, ftol=1e-10, gtol=1e-7)
    lam_full = np.asarray(_ingarch_lambda(r.x, jnp.asarray(y), jnp.asarray(X), mu0, po_t, pm_t))
    ls = np.asarray(nb2_logpmf(jnp.asarray(y[T:]), jnp.asarray(np.maximum(lam_full[T:], FLOOR)), float(kappa)))
    return ls, dict(kappa=float(kappa), model="NB-INGARCH", dev_ll=-float(r.fun))


# --------------------------------------------------------------------------- #
#  Hyperparameter search ranges (shared by the FV selectors below)              #
# --------------------------------------------------------------------------- #
# All non-neural baselines select their hyperparameters by particle-swarm
# optimization of the 40% held-out FORWARD-VALIDATION pLL (fit_score_*_fv), then a
# box-L-BFGS-MT refine on the ANALYTIC bilevel hypergradient.  kappa (dispersion),
# s (softplus bandwidth; s->0 = hard ReLU), the ridge penalty and Fourier scales
# are the forward-validation hyperparameters; the latent coefficients are inner-fit on the 60%
# train at fixed hypers.
LOG_KAPPA = (np.log(0.1), np.log(1e4))          # dispersion search range
LOG_S = (np.log(1e-3), np.log(3.0))             # softplus bandwidth (small -> ReLU)
LOG_PEN = (np.log(1e-8), np.log(1e3))           # histogram ridge penalty
LOG_SLEVEL, LOG_SFOUR, LOG_SLIN = (np.log(0.5), np.log(20)), (np.log(0.01), np.log(20)), (np.log(1e-6), np.log(1e-2))


# --------------------------------------------------------------------------- #
#  Forward-validation (FV) selection with an ANALYTIC bilevel hypergradient     #
# --------------------------------------------------------------------------- #
# The sole selection path for the non-neural baselines: choose the hyperparameters
# h (kappa, softplus s, ridge penalty / Fourier design scales) by maximizing the
# 40% held-out VALIDATION pLL, where for each h the latent parameters are inner-fit
# on the 60% TRAINING portion.  The hypergradient dpLL_val/dh is computed by
# IMPLICIT differentiation of the inner argmin (no autodiff through the optimizer):
#
#     z*(h)  = argmin_z F_z(z; h)          inner train MAP objective (z unconstrained)
#     G(h)   = NLL_val(z*(h); h)           outer validation negative log-likelihood
#     dG/dh  = dG/dh|_z*  -  lam^T d/dh grad_z F_z ,   lam = [grad^2_z F_z]^{-1} dG/dz
#
# Box constraints on the latent parameters are handled by running the inner
# problem in the UNCONSTRAINED reparametrized coordinate z (theta = to_c(z), the
# same smooth box map lbfgs_mt uses), so grad_z F_z(z*) = 0 holds exactly and the
# IFT is valid; near-active bounds appear as flat directions in grad^2_z F_z and
# are absorbed by a small Levenberg damping (they contribute ~0 to the
# hypergradient).  kappa is one of the hyperparameters h everywhere (never fitted
# with the latent).  Multi-start box-L-BFGS-MT maximizes pLL_val on the analytic
# hypergradient (a jax.custom_vjp), keeping the best of the random starts.  This
# mirrors the GP-DHP evidence adjoint but swaps the outer objective (evidence
# log-det -> validation NLL) and the inner objective's data window (dev -> train).


class _BilevelFV(NamedTuple):
    fv_pll: object      # h -> validation pLL scalar (maximize)
    neg_fv: object      # custom_vjp: h -> validation NLL with the analytic IFT gradient (minimize)
    F_z: object         # (z, h) -> train MAP objective (inner; minimize over z)
    G_z: object         # (z, h) -> validation NLL (outer)
    solve: object       # h -> z* at the best inner start (for verification/introspection)


def _nb_nll(y, eta, kappa, s, identity=False):
    """NB2 negative log-likelihood at pre-link eta (softplus link, or identity for INGARCH)."""
    mu = jnp.maximum(eta, FLOOR) if identity else link_mu(eta, "softplus", s)
    return -jnp.sum(nb2_logpmf(y, mu, kappa))


def _damped(H, damp_rel=1e-8, damp_abs=1e-10):
    """Levenberg-damped inner Hessian: absorbs the near-zero (near-active-bound) directions
    of the reparametrized Hessian without perturbing the well-conditioned ones."""
    p = H.shape[0]
    return H + (damp_rel * jnp.mean(jnp.abs(jnp.diag(H))) + damp_abs) * jnp.eye(p)


def _make_bilevel(F_z, G_z, z0_fn, inner_maxiter=100, inner_gtol=1e-9, n_polish=4):
    """Assemble the FV bilevel objective from a train objective F_z(z,h), a validation
    objective G_z(z,h) and an inner-start factory z0_fn(h)->(S,p).  Returns a _BilevelFV
    whose neg_fv carries the analytic IFT hypergradient (no autodiff through the solver).
    The inner solve is L-BFGS multi-start + a few damped-Newton polish steps that nail
    grad_z F_z(z*) = 0 (tight stationarity, so the IFT adjoint is exact)."""
    def solve(h):
        z0s = z0_fn(h)
        def one(z0):
            r = minimize_lbfgs_mt(lambda z: F_z(z, h), z0, maxiter=inner_maxiter,
                                  gtol=inner_gtol, ftol=1e-12)
            return r.x, r.fun
        zs, funs = jax.vmap(one)(z0s)
        z = zs[jnp.argmin(funs)]
        for _ in range(n_polish):                  # damped-Newton polish to true stationarity,
            g = jax.grad(F_z, 0)(z, h)             # GUARDED: reject any step that doesn't shrink ||grad_z F||
            z_new = z - jnp.linalg.solve(_damped(jax.hessian(F_z, 0)(z, h)), g)
            g_new = jax.grad(F_z, 0)(z_new, h)     # (so an overshoot on a stiff/ill-conditioned inner
            ok = jnp.isfinite(g_new).all() & (jnp.dot(g_new, g_new) < jnp.dot(g, g))   # Hessian never diverges)
            z = jnp.where(ok, z_new, z)
        return z

    @jax.jit
    def fv_pll(h):
        return -G_z(solve(h), h)                    # validation pLL (maximize)

    @jax.custom_vjp
    def neg_fv(h):
        return G_z(solve(h), h)                     # validation NLL (refine minimizes)

    def _fwd(h):
        z = solve(h)
        return G_z(z, h), (h, z)

    def _bwd(res, g):
        h, z = res
        gGz = jax.grad(G_z, 0)(z, h)               # dG/dz  at z*
        gGh = jax.grad(G_z, 1)(z, h)               # dG/dh  (explicit)
        lam = jnp.linalg.solve(_damped(jax.hessian(F_z, 0)(z, h)), gGz)   # adjoint H^{-1} dG/dz
        mixed = jax.grad(lambda hh: lam @ jax.grad(F_z, 0)(z, hh))(h)     # lam^T d/dh grad_z F_z
        return (jnp.asarray(g) * (gGh - mixed),)

    neg_fv.defvjp(_fwd, _bwd)
    return _BilevelFV(fv_pll, neg_fv, F_z, G_z, solve)


def verify_fv_grad(bl, h0, n_newton=6):
    """Analytic IFT hypergradient vs autodiff through n_newton damped-Newton steps taken
    FROM the converged z* (the gold-standard check; NOT finite differences, which fail on
    the flat/saturated directions).  Returns (analytic, unrolled) gradient arrays."""
    h0 = jnp.asarray(h0, float)
    z_star = jax.lax.stop_gradient(bl.solve(h0))
    ana = np.asarray(jax.grad(bl.neg_fv)(h0))

    def unrolled(h):
        z = z_star
        for _ in range(n_newton):
            z = z - jnp.linalg.solve(_damped(jax.hessian(bl.F_z, 0)(z, h)), jax.grad(bl.F_z, 0)(z, h))
        return bl.G_z(z, h)
    ref = np.asarray(jax.grad(unrolled)(h0))
    return ana, ref


# ---- per-model FV builders (train/val row splits + F_z, G_z, inner starts) -------------- #
def _param_fv_eval(model, y_dev, period, D_max, Zcov_dev):
    ncov = 0 if Zcov_dev is None else Zcov_dev.shape[1]
    y_dev = np.asarray(y_dev, float); T = len(y_dev)
    X = dhp_lag_matrix(y_dev, D_max); t_idx = np.arange(1, T + 1, dtype=float)
    tr, va = fv_split(T)
    bounds, starts, names = _param_bounds_starts(model, y_dev, ncov)
    to_c, to_raw = _box_transforms([b[0] for b in bounds], [b[1] for b in bounds])
    z0 = jnp.stack([to_raw(jnp.asarray(np.asarray(x0, float))) for x0 in starts.values()])
    Xtr, ttr, ytr = jnp.asarray(X[tr]), jnp.asarray(t_idx[tr]), jnp.asarray(y_dev[tr])
    Xva, tva, yva = jnp.asarray(X[va]), jnp.asarray(t_idx[va]), jnp.asarray(y_dev[va])
    Ztr = jnp.asarray(Zcov_dev[tr]) if ncov else jnp.zeros((len(tr), 0))
    Zva = jnp.asarray(Zcov_dev[va]) if ncov else jnp.zeros((len(va), 0))
    per = float(period)

    def F_z(z, h):
        kappa, s = jnp.exp(h[0]), jnp.exp(h[1]); th = to_c(z)
        return _nb_nll(ytr, _param_mu(th, kappa, model, ncov, False, Xtr, ttr, per, Ztr), kappa, s)

    def G_z(z, h):
        kappa, s = jnp.exp(h[0]), jnp.exp(h[1]); th = to_c(z)
        return _nb_nll(yva, _param_mu(th, kappa, model, ncov, False, Xva, tva, per, Zva), kappa, s)

    return _make_bilevel(F_z, G_z, lambda h: z0)


def _hist_fv_eval(y_dev, y_test, period, daily, cal_index, dates, Zcov_full, D_max):
    y, Xb, Xh_raw = _hist_design(y_dev, y_test, period, daily, cal_index, dates, Zcov_full, D_max)
    T = len(y_dev); tr, va = fv_split(T); nb, nh = Xb.shape[1], Xh_raw.shape[1]
    hsc = Xh_raw[tr].std(axis=0, ddof=1)                # h_scale on TRAIN rows only
    hsc[~np.isfinite(hsc) | (hsc <= 0)] = 1.0
    Xh = Xh_raw / hsc
    Xbtr, Xhtr, ytr = jnp.asarray(Xb[tr]), jnp.asarray(Xh[tr]), jnp.asarray(y[tr])
    Xbva, Xhva, yva = jnp.asarray(Xb[va]), jnp.asarray(Xh[va]), jnp.asarray(y[va])
    hs = jnp.asarray(hsc)
    to_c, to_raw = _box_transforms([None] * nb + [0.0] * nh, [None] * (nb + nh))
    main = np.array([max(float(np.mean(y[tr])), 1e-3)] + [0.0] * (nb - 1) + [0.0] * nh)
    small = main.copy(); small[nb:nb + nh] = np.maximum(1e-4 * hsc, 1e-8)
    z0 = jnp.stack([to_raw(jnp.asarray(main)), to_raw(jnp.asarray(small))])

    def F_z(z, h):
        pen, kappa, s = jnp.exp(h[0]), jnp.exp(h[1]), jnp.exp(h[2])
        p = to_c(z); eta = Xbtr @ p[:nb] + Xhtr @ p[nb:nb + nh]
        return _nb_nll(ytr, eta, kappa, s) + pen * jnp.sum((p[nb:nb + nh] / hs) ** 2)

    def G_z(z, h):
        kappa, s = jnp.exp(h[1]), jnp.exp(h[2])
        p = to_c(z); eta = Xbva @ p[:nb] + Xhva @ p[nb:nb + nh]
        return _nb_nll(yva, eta, kappa, s)

    return _make_bilevel(F_z, G_z, lambda h: z0)


def _bo_fv_eval(y_dev, period, K, Zcov_dev, cov_scale=10.0):
    y_dev = np.asarray(y_dev, float); T = len(y_dev); t = np.arange(1, T + 1, dtype=float)
    tr, va = fv_split(T); ncov = 0 if Zcov_dev is None else Zcov_dev.shape[1]
    Hc = []
    for k in range(1, K + 1):
        Hc.append(np.sin(2 * np.pi * k * t / period)); Hc.append(np.cos(2 * np.pi * k * t / period))
    Four = np.column_stack(Hc); onescol = np.ones((T, 1)); tcol = t.reshape(-1, 1)
    Zc = cov_scale * np.asarray(Zcov_dev, float) if ncov else np.zeros((T, 0))
    onestr, onesva = jnp.asarray(onescol[tr]), jnp.asarray(onescol[va])
    Ftr, Fva = jnp.asarray(Four[tr]), jnp.asarray(Four[va])
    ttr, tva = jnp.asarray(tcol[tr]), jnp.asarray(tcol[va])
    Ztr, Zva = jnp.asarray(Zc[tr]), jnp.asarray(Zc[va])
    ytr, yva = jnp.asarray(y_dev[tr]), jnp.asarray(y_dev[va])
    q = 1 + 2 * K + 1 + ncov; ymaxtr = jnp.maximum(ytr, 1e-3)

    def _B(h, ones, Fo, tc, Zz):
        return jnp.concatenate([jnp.exp(h[0]) * ones, jnp.exp(h[1]) * Fo,
                                jnp.exp(h[2]) * tc, Zz], axis=1)

    def F_z(z, h):
        kappa, s = jnp.exp(h[3]), jnp.exp(h[4])
        return _nb_nll(ytr, _B(h, onestr, Ftr, ttr, Ztr) @ z, kappa, s) + 0.5 * jnp.sum(z ** 2)

    def G_z(z, h):
        kappa, s = jnp.exp(h[3]), jnp.exp(h[4])
        return _nb_nll(yva, _B(h, onesva, Fva, tva, Zva) @ z, kappa, s)

    def z0_fn(h):
        B = _B(h, onestr, Ftr, ttr, Ztr)
        return jnp.linalg.solve(B.T @ B + jnp.eye(q), B.T @ ymaxtr)[None, :]   # ridge init (1 start)

    return _make_bilevel(F_z, G_z, z0_fn)


def _ingarch_fv_eval(y_dev, period, daily, cal_index, dates, Zcov_full):
    cfg = ING_CFG[daily]; po, pm, kind = cfg["past_obs"], cfg["past_mean"], cfg["cov"]
    y_dev = np.asarray(y_dev, float); T = len(y_dev); tr, va = fv_split(T)
    cal = np.asarray(cal_index)[:T]; dts = None if dates is None else np.asarray(dates)[:T]
    Xint = ingarch_covariates(cal, period, kind, dts)
    if Zcov_full is not None:
        Zc = np.asarray(Zcov_full, float)[:T]
        X = Zc if Xint is None else np.column_stack([Xint, Zc])
    else:
        X = Xint if Xint is not None else np.zeros((T, 0))
    npo, npm, nx = len(po), len(pm), X.shape[1]
    mu0 = float(np.mean(y_dev[tr]))                    # presample = TRAIN mean
    ydev_j, X_j = jnp.asarray(y_dev), jnp.asarray(X); po_t, pm_t = tuple(po), tuple(pm)
    tr_j, va_j = jnp.asarray(tr), jnp.asarray(va)
    to_c, to_raw = _box_transforms([1e-6] + [0.0] * npo + [0.0] * npm + [None] * nx,
                                   [None] + [None] * npo + [0.999] * npm + [None] * nx)
    # The INGARCH recursion is NON-CONVEX in (beta, alpha): a single inner start can land in
    # different basins near a bifurcation (FP-sensitive), making the FV objective two-valued.
    # Multi-start the inner solve (default / low-persistence / high-persistence) and keep the
    # lowest train NLL, so theta*(h) is the GLOBAL inner optimum and neg_fv is single-valued.
    x0s = [np.concatenate([[0.5 * mu0], np.full(npo, 0.2), np.full(npm, 0.2), np.zeros(nx)]),
           np.concatenate([[0.9 * mu0], np.full(npo, 0.05), np.full(npm, 0.05), np.zeros(nx)]),
           np.concatenate([[0.2 * mu0], np.full(npo, 0.3), np.full(npm, 0.6), np.zeros(nx)])]
    z0 = jnp.stack([to_raw(jnp.asarray(x)) for x in x0s])

    def _lam(z):
        return _ingarch_lambda(to_c(z), ydev_j, X_j, mu0, po_t, pm_t)

    def F_z(z, h):
        return _nb_nll(ydev_j[tr_j], _lam(z)[tr_j], jnp.exp(h[0]), 1.0, identity=True)

    def G_z(z, h):
        return _nb_nll(ydev_j[va_j], _lam(z)[va_j], jnp.exp(h[0]), 1.0, identity=True)

    return _make_bilevel(F_z, G_z, lambda h: z0)


# --------------------------------------------------------------------------- #
#  Multi-start L-BFGS-MT selection (the FV hyperparameter selector)             #
# --------------------------------------------------------------------------- #
# Sample n_starts RANDOM hyperparameter initializations and refine EACH by
# box-L-BFGS-MT (<=maxiter) on the analytic FV bilevel hypergradient, all in
# parallel across a thread pool of n_devices CPU cores (each L-BFGS is a jitted
# trajectory, compiled once then dispatched, that releases the GIL during XLA so N
# run on N cores; NOT vmap), then keep the start whose refined point has the best
# validation pLL.  Random inits: uniform in the (log-scaled) box for bounded/scale
# hypers -- the baselines' h vector is already all log-coordinates, GP-DHP maps a
# uniform unit cube through _unit_to_nat (log for sd/variance/rate, linear for
# bounded).  (An unbounded, non-positive hyper would be drawn from a wide Gaussian;
# none exist in the current models.)  Discrete choices (K_fourier) round-robin.
# Multi-start L-BFGS-MT forward-validation selection driver for the baselines.
def multistart_lbfgs(neg_fvs, keys, bounds, sample_fn, n_starts=20, maxiter=300, seed=0, n_devices=10):
    """neg_fvs: {key: neg_fv callable} (the FV bilevel validation-NLL custom_vjp for each discrete choice);
    keys: discrete choices assigned round-robin across the n_starts; bounds: [(lo,hi)] per hyper;
    sample_fn(rng)->h0 (natural units the objective expects).  Each start is refined by
    minimize_lbfgs_mt_box(neg_fvs[key], h0, bounds, maxiter) -- a jitted trajectory (compiled once per key,
    then fast dispatch) run across a ThreadPoolExecutor of n_devices cores.  val pLL at the refined point
    is -res.fun (the L-BFGS already evaluated neg_fv there, no re-solve).  Returns (h_best, val_pll_best, key_best)."""
    from concurrent.futures import ThreadPoolExecutor
    rng = np.random.RandomState(seed)
    bnds = list(bounds)
    starts = [(keys[i % len(keys)], np.asarray(sample_fn(rng), float)) for i in range(n_starts)]

    def _mk(nf):
        @jax.jit
        def _one(h0):
            res = minimize_lbfgs_mt_box(nf, h0, bnds, maxiter=maxiter)
            return res.x, -res.fun                         # (refined hyper, validation pLL)
        return _one
    solvers = {k: _mk(neg_fvs[k]) for k in keys}
    # Pre-compile each solver single-threaded BEFORE the pool: JAX *tracing* is not thread-safe,
    # so concurrent first-calls would race and return a corrupted (res.x, res.fun).  After warm-up
    # the pool only DISPATCHES the cached executable, which is thread-safe.
    for k in keys:
        h0 = next((h for (kk, h) in starts if kk == k), None)
        if h0 is not None:
            jax.block_until_ready(solvers[k](jnp.asarray(h0)))

    def run(item):
        k, h0 = item
        hx, negpll = solvers[k](jnp.asarray(h0))
        p = float(negpll)
        return np.asarray(hx), (p if np.isfinite(p) else -1e18), k

    if n_devices > 1:
        with ThreadPoolExecutor(max_workers=int(n_devices)) as ex:
            results = list(ex.map(run, starts))
    else:
        results = [run(s) for s in starts]
    return max(results, key=lambda r: r[1])


def _uniform_box_sampler(lo, hi):
    """Uniform in [lo, hi] per coordinate (the baselines' h is all log-coordinates, so this is
    log-uniform in kappa / softplus s / ridge penalty / Fourier scales)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return lambda rng: lo + rng.random(len(lo)) * (hi - lo)


def fit_score_parametric_fv(model, y_dev, y_test, period, Zcov_dev=None, Zcov_test=None,
                            n_starts=20, maxiter=300, seed=0, n_devices=10):
    """Multi-start L-BFGS-MT selection of (kappa, softplus s) on the analytic FV hypergradient."""
    bl = _param_fv_eval(model, y_dev, period, 100, Zcov_dev)
    lo, hi = [LOG_KAPPA[0], LOG_S[0]], [LOG_KAPPA[1], LOG_S[1]]
    h, pll, _ = multistart_lbfgs({None: bl.neg_fv}, [None], list(zip(lo, hi)), _uniform_box_sampler(lo, hi),
                                 n_starts, maxiter, seed, n_devices)
    k, s = float(np.exp(h[0])), float(np.exp(h[1]))
    fit = fit_parametric(model, y_dev, period, Zcov_dev=Zcov_dev, kappa=k, link="softplus", link_scale=s)
    ls = score_parametric(fit, y_dev, y_test, Zcov_test=Zcov_test)
    return ls, dict(model=PARAM_LABEL[model], kappa=k, link_scale=s, val_pll=pll)


def fit_score_histogram_fv(y_dev, y_test, period, daily, cal_index, dates=None, Zcov_full=None,
                           D_max=100, n_starts=20, maxiter=300, seed=0, n_devices=10):
    """Multi-start L-BFGS-MT selection of (ridge penalty, kappa, softplus s)."""
    bl = _hist_fv_eval(y_dev, y_test, period, daily, cal_index, dates, Zcov_full, D_max)
    lo, hi = [LOG_PEN[0], LOG_KAPPA[0], LOG_S[0]], [LOG_PEN[1], LOG_KAPPA[1], LOG_S[1]]
    h, pll, _ = multistart_lbfgs({None: bl.neg_fv}, [None], list(zip(lo, hi)), _uniform_box_sampler(lo, hi),
                                 n_starts, maxiter, seed, n_devices)
    pen, k, s = float(np.exp(h[0])), float(np.exp(h[1])), float(np.exp(h[2]))
    y, Xb, Xh_raw = _hist_design(y_dev, y_test, period, daily, cal_index, dates, Zcov_full, D_max)
    T = len(y_dev); fi = np.arange(T); si = np.arange(T, len(y))
    ls, kap, _ = _hist_candidate(Xb, Xh_raw, y, fi, si, pen, k, "softplus", s)
    return ls, dict(model="Histogram DHP-NB", kappa=k, penalty=pen, link_scale=s, val_pll=pll)


def fit_score_baseline_only_fv(y_dev, y_test, period, Zcov_dev=None, Zcov_test=None,
                               K_fourier_grid=(1, 2, 3), n_starts=20, maxiter=300, seed=0, n_devices=10):
    """Multi-start L-BFGS-MT selection of (sigma_level, sigma_fourier, sigma_lin, kappa, softplus s);
    K_fourier assigned round-robin across the random starts."""
    lo = [LOG_SLEVEL[0], LOG_SFOUR[0], LOG_SLIN[0], LOG_KAPPA[0], LOG_S[0]]
    hi = [LOG_SLEVEL[1], LOG_SFOUR[1], LOG_SLIN[1], LOG_KAPPA[1], LOG_S[1]]
    neg_fvs = {K: _bo_fv_eval(y_dev, period, K, Zcov_dev).neg_fv for K in K_fourier_grid}
    h, pll, K = multistart_lbfgs(neg_fvs, list(K_fourier_grid), list(zip(lo, hi)),
                                 _uniform_box_sampler(lo, hi), n_starts, maxiter, seed, n_devices)
    fixed = dict(K_fourier=K, sigma_level=float(np.exp(h[0])), sigma_fourier=float(np.exp(h[1])),
                 sigma_lin=float(np.exp(h[2])))
    k, s = float(np.exp(h[3])), float(np.exp(h[4]))
    ls, kap, _ = fit_score_baseline_only(y_dev, y_test, period, fixed, k, Zcov_dev, Zcov_test, "softplus", s)
    return ls, dict(model="Baseline only", kappa=k, link_scale=s, val_pll=pll, **fixed)


def fit_score_ingarch_fv(y_dev, y_test, period, daily, cal_index, dates=None, Zcov_full=None,
                         n_starts=20, maxiter=300, seed=0, n_devices=10):
    """Multi-start L-BFGS-MT selection of kappa (identity link)."""
    bl = _ingarch_fv_eval(y_dev, period, daily, cal_index, dates, Zcov_full)
    lo, hi = [LOG_KAPPA[0]], [LOG_KAPPA[1]]
    h, pll, _ = multistart_lbfgs({None: bl.neg_fv}, [None], list(zip(lo, hi)), _uniform_box_sampler(lo, hi),
                                 n_starts, maxiter, seed, n_devices)
    k = float(np.exp(h[0]))
    ls, _ = fit_score_ingarch(y_dev, y_test, period, daily, cal_index, dates, Zcov_full, kappa=k)
    return ls, dict(model="NB-INGARCH", kappa=k, val_pll=pll)


if __name__ == "__main__":
    # quick self-test: NB2 log-pmf vs jax.scipy.stats, link sanity
    from jax.scipy.stats import nbinom
    y = np.array([0., 1, 3, 10]); mu = np.array([0.5, 2.0, 3.0, 8.0]); kap = 4.0
    j = np.asarray(nb2_logpmf(jnp.asarray(y), jnp.asarray(mu), kap))
    s = np.asarray(nbinom.logpmf(jnp.asarray(y), kap, kap / (kap + jnp.asarray(mu))))
    print("nb2_logpmf max|JAX-jax.scipy| =", np.max(np.abs(j - s)))
    e = np.array([-1.0, 0.0, 2.0])
    print("relu link:", np.asarray(link_mu(jnp.asarray(e), "relu")))
    print("softplus(s=0.1):", np.asarray(link_mu(jnp.asarray(e), "softplus", 0.1)))
