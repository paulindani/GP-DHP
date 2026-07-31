"""High-level constrained GP-DHP fitting for the paper-reproduction experiments.

Two helpers that used to be copy-pasted across paper_tables / supp_tables / pit_check / the
constrained diagnostic runner:

  fit_constrained_cell(ds, cov, ...)  -> the ALM stability-constrained FV selection + full-period
        constrained MAP refit for one (dataset, covariate) cell; returns the per-obs held-out NB
        log-scores, achieved R_+, the selection dict, and the data needed downstream.

  constrained_predictive(res, ...)    -> re-derive the constrained theta* (identical ALM rho-ramp as
        the refit) to expose the one-step predictive MEAN lambda(t) on the test window and the fitted
        excitation kernel f_hat; asserts the re-derived per-obs log-scores reproduce res['ls'] to 1e-8.

The settled protocol (delta=1e-4, 2x budget, 20 starts, seed 0) is encoded as the defaults.
"""
import numpy as np
import jax.numpy as jnp

from . import datasets
from . import covariates
from . import gpdhp_cfv_grad as CFV
from ._gpdhp_common import (build_design, _baseline_cols, softplus_link,
                            nb2_loglik, nb2_logpmf_vec, lag_matrix_future)
from .lbfgs_mt import minimize_bfgs_mt

FLOOR = 1e-6
RHO_RAMP = (30., 100., 300., 1e3, 3e3, 1e4, 3e4, 1e5, 3e5)   # ALM penalty ramp (matches CFV refit)


def fit_constrained_cell(ds, cov, *, delta=1e-4, mult=2.0, n_starts=20, seed=0, n_devices=10,
                         root=None, verbose=False):
    """Constrained (ALM) forward-validation selection + full-period constrained MAP refit for one cell.

    Returns a dict: ls (per-obs held-out NB log-scores), Rp (achieved R_+), sel (selection dict), and
    the data (y_dev, y_test, period, daily, C_dev, C_test) + budgets that constrained_predictive needs.
    Asserts the fitted kernel is subcritical (R_+ < 1)."""
    maxit = int(300 * mult); inner = int(500 * mult); refit = int(800 * mult)
    y_dev, y_test, period, daily = datasets.load_split(ds)
    C_dev = C_test = None
    if cov:
        C_dev, C_test, _ = covariates.build_covariates(ds, root=root)
        groups = [np.array([0]), np.arange(1, C_dev.shape[1])]
        sel = CFV.select_constrained_cov(y_dev, C_dev, period, daily, groups, delta=delta,
                                         n_starts=n_starts, maxiter=maxit, seed=seed, n_devices=n_devices,
                                         use_fft=True, verbose=verbose, inner_maxiter=inner)
        ls, Rp = CFV.constrained_refit_score_cov(y_dev, C_dev, C_test, y_test, sel["h"], sel["kappa"],
                                                 sel["link_scale"], sel["col_sigmas"], delta=delta,
                                                 refit_maxiter=refit)
    else:
        sel = CFV.select_constrained(y_dev, period, daily, delta=delta, n_starts=n_starts, maxiter=maxit,
                                     seed=seed, n_devices=n_devices, use_fft=True, verbose=False,
                                     inner_maxiter=inner)
        ls, Rp = CFV.constrained_refit_score(y_dev, y_test, sel["h"], sel["kappa"],
                                             sel["link_scale"], delta=delta, refit_maxiter=refit)
    Rp = float(Rp)
    assert Rp < 1.0, f"{ds}/{'cov' if cov else 'nocov'}: NOT SUBCRITICAL R_+={Rp}"
    return dict(ls=np.asarray(ls, float), Rp=Rp, sel=sel, cov=cov, ds=ds,
                y_dev=y_dev, y_test=y_test, period=period, daily=daily,
                C_dev=C_dev, C_test=C_test, delta=delta, refit=refit)


def constrained_predictive(res, *, floor=FLOOR):
    """Re-derive the constrained theta* (same ALM rho-ramp as the refit) to expose the one-step
    predictive mean lambda(t) on the test window and the fitted excitation f_hat.

    Returns dict(y_test, lam, kappa, f_hat, theta). The end-to-end check that the re-derived per-obs
    log-scores match res['ls'] guards against silent design mismatches (e.g. a dropped NB offset)."""
    sel, cov, delta, refit = res["sel"], res["cov"], res["delta"], res["refit"]
    h, kappa, lk = sel["h"], sel["kappa"], sel["link_scale"]
    y_devn = np.asarray(res["y_dev"], float); y_testn = np.asarray(res["y_test"], float)
    T = len(y_devn); n_test = len(y_testn); budget = 1.0 - delta; D = int(h["D_max"])
    des = build_design(y_devn, h)
    B, X, L, m_NB, q_b, q_g = des["B"], des["X"], des["L"], des["m_NB"], des["q_b"], des["q_g"]
    mj, Lj = jnp.asarray(m_NB), jnp.asarray(L)
    offset = jnp.asarray(des["offset"])            # X @ m_NB -- fixed NB excitation (both branches)
    if cov:
        cs = jnp.asarray(sel["col_sigmas"], float); ncol = int(cs.shape[0])
        Ctr = jnp.asarray(res["C_dev"]) * cs
        XL = jnp.asarray(X) @ Lj if q_g else jnp.zeros((T, 0))
        A = jnp.concatenate([jnp.asarray(B), Ctr, XL], axis=1); gsl = q_b + ncol
    else:
        A, ncol, gsl = jnp.asarray(des["A"]), 0, q_b
    ydev = jnp.asarray(y_devn); dim = A.shape[1]
    Rplus = lambda th: jnp.sum(jnp.maximum(mj + (Lj @ th[gsl:] if q_g else 0.0), 0.0))

    def F(th):
        mu = softplus_link(offset + A @ th, lk, floor)
        return -nb2_loglik(ydev, mu, kappa) + 0.5 * jnp.sum(th ** 2)

    th = jnp.zeros(dim); gam = 0.0
    for rho in RHO_RAMP:
        def Lrho(z, g=gam, r=rho):
            hinge = jnp.maximum(0.0, g + r * (Rplus(z) - budget))
            return F(z) + (0.5 / r) * (hinge ** 2 - g ** 2)
        th = minimize_bfgs_mt(Lrho, th, maxiter=refit, gtol=1e-8, ftol=1e-14).x
        gam = jnp.maximum(0.0, gam + rho * (Rplus(th) - budget))
    f_hat = np.asarray(mj + (Lj @ th[gsl:] if q_g else 0.0), float)

    tt = jnp.arange(T + 1, T + n_test + 1, dtype=jnp.float64)
    B_te = _baseline_cols(tt, h["period"], h); X_te = lag_matrix_future(y_devn, y_testn, D)
    eta = B_te @ th[:q_b] + X_te @ jnp.asarray(f_hat)
    if cov:
        eta = eta + (jnp.asarray(res["C_test"]) * cs) @ th[q_b:q_b + ncol]
    lam = np.asarray(softplus_link(eta, lk, floor), float)          # one-step predictive mean

    ls_mine = np.asarray(nb2_logpmf_vec(jnp.asarray(y_testn), jnp.asarray(lam), float(kappa)), float)
    d_ls = float(np.max(np.abs(ls_mine - res["ls"])))
    assert d_ls < 1e-8, f"re-derived predictive != refit (max|dlogscore|={d_ls:.2e})"
    return dict(y_test=y_testn, lam=lam, kappa=float(kappa), f_hat=f_hat, theta=np.asarray(th, float))
