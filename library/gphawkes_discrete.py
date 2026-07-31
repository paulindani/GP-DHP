"""Discrete-time Gaussian-process-modulated Hawkes process benchmark: the discrete analogue of the
continuous-time GP-modulated Hawkes processes of \\citet{zhou2020} (sigmoid Gaussian Hawkes) and
\\citet{zhang2020} (sparse squared-GP Hawkes).  The triggering kernel over the discrete lags is a
squared Gaussian process---a nonnegative, fully nonparametric kernel with no parametric backbone---
and the intensity uses the SAME softplus link as GP-DHP:

    lambda_t = h_s( B_t beta + sum_{d=1..D} N_{t-d} phi(d) ),   phi(d) = ( sigma (L0(ell) z) )_d^2,
    z ~ N(0, I),   L0(ell) = chol( RBF(ell) ),   N_t | H_{t-1} ~ NegBin(mean=lambda_t, size=kappa),

with h_s the softplus link of scale s (bandwidth).  The baseline/covariate design B and the
negative-binomial observation layer match the fixed-bin Histogram DHP-NB.

Hyperparameter selection is the SAME protocol as GP-DHP: the hyperparameters psi=(sigma, ell, s, kappa)
are chosen by 40% held-out forward validation of the one-step predictive log-likelihood, optimized by a
MULTI-START (20-start) L-BFGS-MT search on the analytic implicit-function-theorem BILEVEL HYPERGRADIENT
(no autodiff through the inner solver), exactly as gpdhp_fv_grad.make_fv_fns does for GP-DHP.  The inner
variables theta=(z, beta) --- the squared-GP latent (unit-ridge prior) and the free baseline
coefficients --- are the MAP fit at fixed psi:

    theta*(psi) = argmin_theta F(theta, psi),   F = -loglik_train + 0.5||z||^2,   grad_theta F(theta*) = 0
    G(psi)      = -loglik_val(theta*(psi), psi)                       (validation NLL; pLL_val = -G)
    dG/dpsi     = dG/dpsi|_theta*  -  lambda . d/dpsi grad_theta F,   lambda = [grad^2_theta F]^{-1} dG/dtheta.

The squared-GP kernel is nonlinear in z (unlike the collapsed GP-DHP latent), so the inner objective is
non-convex; the IFT adjoint remains valid at any inner stationary point, and the inner Hessian is
Levenberg-damped for the adjoint solve.  The final model is refit on the whole fitting period at the
selected psi, then scored one-step-ahead on the test window.
"""
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(_HERE))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from .jax_baselines import (dhp_lag_matrix, dhp_lag_matrix_future, hist_baseline_design,   # noqa: E402
                           HIST_CFG, multistart_lbfgs, FLOOR)
from ._gpdhp_common import softplus_link, nb2_loglik, nb2_logpmf_vec       # noqa: E402
from .lbfgs_mt import minimize_bfgs_mt                                                       # noqa: E402

# forward-validation search box for psi = (sigma, ell, s, kappa); log-uniform starts, box-L-BFGS-MT refine
BOUNDS = dict(sigma=(0.01, 5.0), ell=(0.5, 50.0), s=(0.02, 2.0), kappa=(0.25, 1.0e6))
_ZSCALE = 0.3            # scale of the fixed nonzero inner-latent start (z=0 is a saddle of phi=f^2)


def _damp_H(H, damp_rel=1e-8, damp_abs=1e-10):
    return H + (damp_rel * jnp.mean(jnp.abs(jnp.diagonal(H))) + damp_abs) * jnp.eye(H.shape[0])


def _standardize(Bfull, T):
    B = np.asarray(Bfull, float).copy()
    m = B[:T].mean(0); sd = B[:T].std(0, ddof=0)
    for j in range(B.shape[1]):
        if sd[j] > 1e-8 and not np.allclose(B[:, j], B[0, j]):
            B[:, j] = (B[:, j] - m[j]) / sd[j]
    return B


def _matched_design(y_dev, y_test, period, daily, cal, dates, Zfull, D_max):
    """Lag design + standardized histogram-matched baseline/covariate design (dev + test rows)."""
    T = len(y_dev)
    X_dev = dhp_lag_matrix(np.asarray(y_dev, float), D_max)
    X_test = dhp_lag_matrix_future(np.asarray(y_dev, float), np.asarray(y_test, float), D_max)
    Bfull = np.asarray(hist_baseline_design(cal, period, HIST_CFG[bool(daily)]["base"], dates), float)
    if Zfull is not None:
        Bfull = np.column_stack([Bfull, np.asarray(Zfull, float)])
    Bfull = _standardize(Bfull, T)
    return X_dev, X_test, Bfull[:T], Bfull[T:]


def _kernel_ops(D):
    d = np.arange(1, D + 1, dtype=float)
    DD = jnp.asarray((d[:, None] - d[None, :]) ** 2)
    eye = jnp.eye(D)

    def L0(ell):                                            # chol of the RBF CORRELATION (sigma factored out)
        return jnp.linalg.cholesky(jnp.exp(-0.5 * DD / (ell * ell)) + 1e-6 * eye)
    return L0


def make_fv_fns(y_dev, B_dev, X_dev, min_train=0.6, inner_maxiter=500, inner_gtol=1e-8,
                inner_ftol=1e-14, seed=0):
    """FV bilevel value + analytic-hypergradient callables over psi=(sigma,ell,s,kappa).  theta=(z,beta)
    is inner-fit on the first `min_train` of the dev window; the tail is the validation score."""
    y = jnp.asarray(np.asarray(y_dev, float)); T = len(np.asarray(y_dev))
    D = X_dev.shape[1]; p = B_dev.shape[1]; dim = D + p
    ncut = int(np.floor(min_train * T)); tr = jnp.arange(ncut); va = jnp.arange(ncut, T)
    Xj = jnp.asarray(np.asarray(X_dev, float)); Bj = jnp.asarray(np.asarray(B_dev, float))
    ytr, yva = y[tr], y[va]
    L0 = _kernel_ops(D)
    th_init = jnp.concatenate([jnp.asarray(np.random.default_rng(seed).standard_normal(D) * _ZSCALE),
                               jnp.zeros(p)])

    def _eta(theta, nat):
        sigma, ell = nat[0], nat[1]
        z, beta = theta[:D], theta[D:]
        phi = (sigma * (L0(ell) @ z)) ** 2                  # nonnegative squared-GP triggering kernel
        return Bj @ beta + Xj @ phi

    def F_z(theta, nat):                                    # train MAP objective (inner); ridge on z only
        s, kappa = nat[2], nat[3]
        mu = softplus_link(_eta(theta, nat)[tr], s, FLOOR)
        return -nb2_loglik(ytr, mu, kappa) + 0.5 * jnp.sum(theta[:D] ** 2)

    def G_z(theta, nat):                                    # validation NLL (outer)
        s, kappa = nat[2], nat[3]
        mu = softplus_link(_eta(theta, nat)[va], s, FLOOR)
        return -nb2_loglik(yva, mu, kappa)

    @jax.jit
    def inner_map(nat):
        res = minimize_bfgs_mt(lambda th: F_z(th, nat), th_init,
                               maxiter=inner_maxiter, gtol=inner_gtol, ftol=inner_ftol)
        return res.x

    @jax.jit
    def analytic_fv_grad(theta, nat):
        gGz = jax.grad(lambda th: G_z(th, nat))(theta)
        gGh = jax.grad(lambda nt: G_z(theta, nt))(nat)
        H = jax.hessian(lambda th: F_z(th, nat))(theta)
        lam = jnp.linalg.solve(_damp_H(H), gGz)
        c = jax.grad(lambda nt: jnp.dot(lam, jax.grad(lambda th: F_z(th, nt))(theta)))(nat)
        return -(gGh - c)

    @jax.custom_vjp
    def neg_fv(nat):
        return G_z(inner_map(nat), nat)

    def _fwd(nat):
        th = inner_map(nat)
        return G_z(th, nat), (nat, th)

    def _bwd(res, gbar):
        nat, th = res
        return (jnp.asarray(gbar) * (-analytic_fv_grad(th, nat)),)

    neg_fv.defvjp(_fwd, _bwd)
    return dict(F_z=F_z, G_z=G_z, inner_map=inner_map, analytic_fv_grad=analytic_fv_grad,
                neg_fv=neg_fv, dim=dim, D=D, p=p)


def select_fv(y_dev, B_dev, X_dev, n_starts=20, maxiter=300, seed=0, n_devices=10,
              min_train=0.6, verbose=False):
    """20-start L-BFGS-MT on the analytic bilevel hypergradient; returns (psi=[sigma,ell,s,kappa], val_pLL)."""
    fns = make_fv_fns(y_dev, B_dev, X_dev, min_train=min_train, seed=seed)
    names = ["sigma", "ell", "s", "kappa"]
    bnds = [BOUNDS[n] for n in names]
    lo = np.array([b[0] for b in bnds]); hi = np.array([b[1] for b in bnds])

    def sample_fn(rng):                                     # log-uniform starts over the box (all positive)
        u = rng.random(len(names))
        return np.exp(np.log(lo) + u * (np.log(hi) - np.log(lo)))

    nat, pll, _ = multistart_lbfgs({0: fns["neg_fv"]}, [0], bnds, sample_fn,
                                   n_starts=n_starts, maxiter=maxiter, seed=seed, n_devices=n_devices)
    if verbose:
        print(f"  FV multistart ({n_starts}x{maxiter}): sigma={nat[0]:.3g} ell={nat[1]:.3g} "
              f"s={nat[2]:.3g} kappa={nat[3]:.3g}  val_pLL={float(pll):.2f}", flush=True)
    return np.asarray(nat, float), float(pll)


def _refit_score(y_dev, B_dev, X_dev, y_test, B_test, X_test, nat, seed=0, inner_maxiter=800):
    """Refit theta=(z,beta) on the WHOLE dev window at the selected psi, then score the test window."""
    D = X_dev.shape[1]; p = B_dev.shape[1]
    sigma, ell, s, kappa = [float(v) for v in nat]
    L0 = _kernel_ops(D)(ell)
    Bd = jnp.asarray(np.asarray(B_dev, float)); Xd = jnp.asarray(np.asarray(X_dev, float))
    yd = jnp.asarray(np.asarray(y_dev, float))

    def F(theta):
        z, beta = theta[:D], theta[D:]
        eta = Bd @ beta + Xd @ ((sigma * (L0 @ z)) ** 2)
        return -nb2_loglik(yd, softplus_link(eta, s, FLOOR), kappa) + 0.5 * jnp.sum(z ** 2)

    th0 = jnp.concatenate([jnp.asarray(np.random.default_rng(seed).standard_normal(D) * _ZSCALE), jnp.zeros(p)])
    th = minimize_bfgs_mt(F, th0, maxiter=inner_maxiter, gtol=1e-8, ftol=1e-14).x
    z, beta = th[:D], th[D:]
    phi = (sigma * (L0 @ z)) ** 2
    eta_te = jnp.asarray(np.asarray(B_test, float)) @ beta + jnp.asarray(np.asarray(X_test, float)) @ phi
    ls = np.asarray(nb2_logpmf_vec(jnp.asarray(np.asarray(y_test, float)),
                                   softplus_link(eta_te, s, FLOOR), kappa))
    phi_np = np.asarray(phi)
    return ls, dict(sigma=round(sigma, 4), ell=round(ell, 3), s=round(s, 4), kappa=round(kappa, 3),
                    peak_lag=int(np.argmax(phi_np) + 1), kernel_mass=round(float(phi_np.sum()), 4))


def fit_score_gphawkes(y_dev, y_test, period, daily=False, cal=None, dates=None, Zfull=None,
                       D_max=100, n_starts=20, maxiter=300, seed=0, n_devices=10, verbose=True):
    """Select psi=(sigma,ell,s,kappa) by 20-start L-BFGS-MT on the analytic bilevel FV hypergradient
    (softplus link, bandwidth s an FV hyperparameter), refit on the dev window, and score the test window."""
    y_dev = np.asarray(y_dev, float); y_test = np.asarray(y_test, float)
    X_dev, X_test, B_dev, B_test = _matched_design(y_dev, y_test, period, daily, cal, dates, Zfull, D_max)
    nat, val_pll = select_fv(y_dev, B_dev, X_dev, n_starts=n_starts, maxiter=maxiter, seed=seed,
                             n_devices=n_devices, verbose=verbose)
    ls, info = _refit_score(y_dev, B_dev, X_dev, y_test, B_test, X_test, nat, seed=seed)
    info["val_pll"] = round(val_pll, 2)
    return np.asarray(ls, float), info
