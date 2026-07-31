"""Analytic FORWARD-VALIDATION bilevel hypergradient for the softplus GP-DHP -- the
derivative of the 40% held-out validation pLL w.r.t. the (continuous) hyperparameters,
WITHOUT differentiating through the inner MAP optimizer.

Split the dev period into a 60% train / 40% validation window.  For a hyperparameter
vector nat (natural units):

    theta*(nat) = argmin_theta F(theta, nat),   F = -loglik_train(theta, nat) + 0.5||theta||^2
    G(nat)      = -loglik_val(theta*(nat), nat)          (validation NLL; pLL_val = -G)

theta is the whitened GP-DHP latent (unit-ridge prior), fit on the 60% train by lbfgs_mt's
on-device BFGS and treated as a constant.  kappa is one of the hyperparameters nat (never
fitted with the latent).  The hypergradient is the IMPLICIT-FUNCTION-THEOREM adjoint of the
inner argmin (one linear solve with the exact inner Hessian; no autodiff through the solver):

    dG/dnat = dG/dnat|_{theta*}  -  lambda . d/dnat grad_theta F,
    lambda  = [grad^2_theta F]^{-1} (dG/dtheta|_{theta*}),
    lambda . d/dnat grad_theta F = d/dnat [ lambda . grad_theta F ]  (a VJP, lambda frozen).

Because theta is unconstrained and F is unit-ridge-regularized, grad_theta F(theta*) = 0
holds by BFGS stationarity and grad^2_theta F is PD -- no reparametrization, no Newton polish.
A `jax.custom_vjp` wrapper (`neg_fv`) exposes the validation NLL with this analytic gradient,
so minimize_lbfgs_mt_box refines the hyperparameters (multi-start L-BFGS-MT) without ever
autodiffing the inner solve.  Verified against autodiff-through-Newton in verify_fv_grad
(NOT finite differences, which fail on flat/saturated directions).
"""
import os
import sys

import numpy as np
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(_HERE))

from .lbfgs_mt import minimize_bfgs_mt, minimize_lbfgs_mt_box
from ._gpdhp_common import (
    _fourier_design, nb_kernel, softplus_link, nb2_loglik)
from ._gpdhp_fft import _causal_conv, _next_fast_len   # O(T log T) excitation
from . import gpdhp_select as GS               # BOUNDS, LOG, names_for, build_h, helpers

FLOOR = GS.FLOOR


def _damp_H(H, damp_rel=1e-8, damp_abs=1e-10):
    """Levenberg-damped Hessian for the adjoint solve (harmless insurance; the whitened
    unit-ridge GP-DHP inner Hessian is already PD, so the damping is negligible)."""
    return H + (damp_rel * jnp.mean(jnp.abs(jnp.diagonal(H))) + damp_abs) * jnp.eye(H.shape[0])


def make_fv_fns(y_fit, period, daily, Kf, D_max=100, C=None, cov_groups=None, min_train=0.6,
                inner_maxiter=500, inner_gtol=1e-8, inner_ftol=1e-14, use_fft=True):
    """Build the FORWARD-VALIDATION bilevel value + analytic-hypergradient callables for
    one fixed K_fourier.  The whitened latent theta is inner-fit on the first `min_train`
    of y_fit (train MAP objective F = -loglik_train + 0.5||theta||^2); the held-out tail
    is scored (validation NLL G = -loglik_val).  The hypergradient dpLL_val/dnat is the
    implicit-function-theorem adjoint of the inner argmin -- NO autodiff through the MAP
    solver -- exactly as the evidence gradient, but with the outer objective swapped
    (evidence log-det -> validation NLL) and the inner data window (dev -> train):

        theta*(nat) = argmin_theta F(theta, nat),          grad_theta F(theta*) = 0
        dG/dnat = dG/dnat|_theta*  -  lambda . d/dnat grad_theta F,
        lambda  = [grad^2_theta F]^{-1} dG/dtheta.

    theta is unconstrained (unit-ridge whitened prior), so grad_theta F = 0 holds by BFGS
    stationarity and the exact inner Hessian grad^2_theta F is PD -- no reparametrization,
    no Newton polish.  kappa is one of the hyperparameters `nat` (never fitted with the
    latent).  Row splits of the full-dev design give the correct causal one-step-ahead
    validation lags (train history + earlier validation).

    Returns a dict of callables over `nat` (natural units):
      fv_pll(nat)   -> scalar validation pLL   (runs the inner train MAP fit)
      neg_fv(nat)   -> scalar validation NLL    (jax.custom_vjp; grad = analytic adjoint)
      grad_fv(nat)  -> (grad, pll, theta*)      (analytic; numpy grad)
      inner_map, F_z, G_z, analytic_fv_grad, dim, p, ncol.
    """
    y = jnp.asarray(np.asarray(y_fit, float))
    T = len(np.asarray(y_fit))
    ncut = int(np.floor(min_train * T))
    tr = jnp.arange(ncut); va = jnp.arange(ncut, T)
    X = GS._lag_matrix(y_fit, D_max)                     # (T, D) full-dev lag design
    D = D_max
    t = jnp.arange(1, T + 1, dtype=jnp.float64)
    q_b = 1 + 2 * Kf + (2 * 3 if daily else 0) + 1
    p_core = 11 + (1 if daily else 0)
    has_cov = C is not None
    if has_cov:
        Cj = jnp.asarray(np.asarray(C, float)); ncol = int(Cj.shape[1])
        gidx = [jnp.asarray(np.asarray(ix)) for ix in cov_groups]; n_groups = len(gidx)
    else:
        Cj, ncol, gidx, n_groups = None, 0, [], 0
    dim = q_b + ncol + D
    p = p_core + n_groups
    ytr, yva = y[tr], y[va]

    def _baseline(s_level, s_fou, s_week, s_lin):
        parts = [s_level * jnp.ones((T, 1)), s_fou * _fourier_design(t, period, Kf)]
        if daily:
            parts.append(s_week * _fourier_design(t, 7.0, 3))
        parts.append((s_lin * t)[:, None])
        return jnp.concatenate(parts, axis=1)

    def _unpack(nat):
        v = [nat[i] for i in range(11)]
        s_week = nat[11] if daily else 0.0
        return (*v, s_week)

    def _col_sigmas(cov_scales):
        cs = jnp.zeros(ncol)
        for gi in range(n_groups):
            cs = cs.at[gidx[gi]].set(cov_scales[gi])
        return cs

    def _design(nat):
        (s_level, s_fou, s_lin, K_NB, mean_lag, size, kappa,
         sigma_u, ell_u, beta, lk, s_week) = _unpack(nat)
        B = _baseline(s_level, s_fou, s_week, s_lin)
        m_NB = K_NB * nb_kernel(D, mean_lag, size)
        Lc = GS._warped_rbf_chol(D, beta, sigma_u, ell_u)
        XL = X @ Lc
        if has_cov:
            A = jnp.concatenate([B, Cj * _col_sigmas(nat[p_core:]), XL], axis=1)
        else:
            A = jnp.concatenate([B, XL], axis=1)
        return A, X @ m_NB, lk, kappa

    # -- linear predictor eta(theta, nat) over the full fitting period ------------------------- #
    # Dense forms A = [B | C*cs | X L] and returns offset + A theta at O(T D) per eval (plus an
    # O(T D^2) X L build).  The FFT path computes the excitation X(m_NB + L theta_g) as a causal
    # convolution of the counts with the length-D kernel (m_NB + L theta_g) -- O(T log T + D^2),
    # never materialising X L -- and is bit-for-bit the same predictor (verified to ~1e-9).  Both
    # feed identical F_z/G_z, so the IFT hypergradient / custom_vjp below are unchanged.
    if use_fft:
        Lfft = _next_fast_len(T + D + 1)
        y_rfft = jnp.fft.rfft(y, n=Lfft)                 # precomputed once (y fixed)

        def _eta(theta, nat):
            (s_level, s_fou, s_lin, K_NB, mean_lag, size, kappa,
             sigma_u, ell_u, beta, lk, s_week) = _unpack(nat)
            B = _baseline(s_level, s_fou, s_week, s_lin)
            m_NB = K_NB * nb_kernel(D, mean_lag, size)
            Lc = GS._warped_rbf_chol(D, beta, sigma_u, ell_u)
            tb = theta[:q_b]
            if has_cov:
                tc = theta[q_b:q_b + ncol]; tg = theta[q_b + ncol:]
                base = B @ tb + (Cj * _col_sigmas(nat[p_core:])) @ tc
            else:
                tg = theta[q_b:]; base = B @ tb
            f = m_NB + Lc @ tg
            exc = _causal_conv(y_rfft, f, Lfft, T)                  # GP excitation X @ (m_NB + L theta_g)
            return base + exc, lk, kappa, 0.0
    else:
        def _eta(theta, nat):
            A, offset, lk, kappa = _design(nat)
            return offset + A @ theta, lk, kappa, 0.0

    def F_z(theta, nat):                                 # train MAP objective (inner)
        eta, lk, kappa, reg_pen = _eta(theta, nat)
        mu = softplus_link(eta[tr], lk, FLOOR)
        return -nb2_loglik(ytr, mu, kappa) + 0.5 * jnp.sum(theta ** 2) + reg_pen

    def G_z(theta, nat):                                 # validation NLL (outer)
        eta, lk, kappa, _ = _eta(theta, nat)
        mu = softplus_link(eta[va], lk, FLOOR)
        return -nb2_loglik(yva, mu, kappa)

    @jax.jit
    def inner_map(nat):
        res = minimize_bfgs_mt(lambda th: F_z(th, nat), jnp.zeros(dim),
                               maxiter=inner_maxiter, gtol=inner_gtol, ftol=inner_ftol)
        return res.x

    @jax.jit
    def fv_pll(nat):
        return -G_z(inner_map(nat), nat)                 # validation pLL (maximize)

    @jax.jit
    def analytic_fv_grad(theta, nat):
        """dpLL_val/dnat at the fixed theta* -- IFT adjoint of the inner argmin."""
        gGz = jax.grad(lambda th: G_z(th, nat))(theta)   # dG/dtheta
        gGh = jax.grad(lambda nt: G_z(theta, nt))(nat)   # dG/dnat (explicit)
        H = jax.hessian(lambda th: F_z(th, nat))(theta)  # exact inner Hessian
        lam = jnp.linalg.solve(_damp_H(H), gGz)
        c = jax.grad(lambda nt: jnp.dot(lam, jax.grad(lambda th: F_z(th, nt))(theta)))(nat)
        return -(gGh - c)                                # d(pLL_val)/dnat = -d(NLL_val)/dnat

    def grad_fv(nat):
        nat = jnp.asarray(np.asarray(nat, float))
        th = inner_map(nat)
        return np.asarray(analytic_fv_grad(th, nat)), float(-G_z(th, nat)), th

    @jax.custom_vjp
    def neg_fv(nat):
        return G_z(inner_map(nat), nat)                  # validation NLL (refine minimizes)

    def _fwd(nat):
        th = inner_map(nat)
        return G_z(th, nat), (nat, th)

    def _bwd(res, gbar):
        nat, th = res
        return (jnp.asarray(gbar) * (-analytic_fv_grad(th, nat)),)   # d(NLL)/dnat = -d(pLL)/dnat

    neg_fv.defvjp(_fwd, _bwd)

    return dict(F_z=F_z, G_z=G_z, inner_map=inner_map, fv_pll=fv_pll, neg_fv=neg_fv,
                analytic_fv_grad=analytic_fv_grad, grad_fv=grad_fv, dim=dim, p=p, ncol=ncol)


def verify_fv_grad(y_fit, period, daily, Kf, nat0, D_max=100, n_newton=6, min_train=0.6):
    """Analytic FV hypergradient vs autodiff through n_newton damped-Newton steps taken FROM
    the converged theta* (gold standard; not finite differences).  Returns (analytic, ref)."""
    fns = make_fv_fns(y_fit, period, daily, Kf, D_max, min_train=min_train)
    F_z, G_z = fns["F_z"], fns["G_z"]
    nat0 = jnp.asarray(np.asarray(nat0, float))
    th_star = jax.lax.stop_gradient(fns["inner_map"](nat0))
    ana = np.asarray(fns["analytic_fv_grad"](th_star, nat0))    # d(pLL)/dnat

    def unrolled(nat):
        th = th_star
        for _ in range(n_newton):
            th = th - jnp.linalg.solve(_damp_H(jax.hessian(lambda z: F_z(z, nat))(th)),
                                       jax.grad(lambda z: F_z(z, nat))(th))
        return -G_z(th, nat)                                     # pLL_val
    ref = np.asarray(jax.grad(unrolled)(nat0))
    return ana, ref


def select_fv_cov(y_fit, C_fit, period, daily, cov_groups, D_max=100,
                  n_starts=20, maxiter=300, seed=0, cov_bounds=(1e-3, 20.0),
                  verbose=False, n_devices=10, use_fft=True):
    """Covariate GP-DHP MULTI-START L-BFGS-MT FV selection (the cov analogue of gpdhp_select.select):
    search (core hypers + one prior scale per covariate group) JOINTLY by the 40% held-out validation
    pLL -- n_starts random inits refined by box-L-BFGS-MT on the analytic bilevel hypergradient,
    thread-pool parallel over n_devices cores, keep best.  make_fv_fns builds A=[B | C*col_sigmas | X L]
    (the collapsed covariate design); K_fourier round-robin.  cov_groups: list of column-index
    arrays into C_fit.  Returns a dict compatible with gpdhp_select.select plus col_sigmas."""
    from .jax_baselines import multistart_lbfgs
    y_fit = np.asarray(y_fit, float); C_fit = np.asarray(C_fit, float)
    ncol = C_fit.shape[1]
    names_core = GS.names_for(daily); p_core = len(names_core)
    n_groups = len(cov_groups); dims = p_core + n_groups
    bnds = [GS.BOUNDS[nm] for nm in names_core] + [tuple(cov_bounds)] * n_groups
    logf = np.array([GS.LOG[nm] for nm in names_core] + [1] * n_groups, bool)
    lo = np.array([b[0] for b in bnds]); hi = np.array([b[1] for b in bnds])
    lo_log = np.where(logf, lo, 1.0); hi_log = np.where(logf, hi, 1.0)

    def sample_fn(rng):                                               # log/linear per-coord random start
        u = rng.random(dims)
        return np.where(logf, np.exp(np.log(lo_log) + u * (np.log(hi_log) - np.log(lo_log))), lo + u * (hi - lo))

    neg_fvs = {Kf: make_fv_fns(y_fit, period, daily, Kf, D_max, C=C_fit, cov_groups=cov_groups,
                               use_fft=use_fft)["neg_fv"]
               for Kf in (1, 2, 3)}
    nat1, pll1, Kf = multistart_lbfgs(neg_fvs, [1, 2, 3], bnds, sample_fn,
                                      n_starts=n_starts, maxiter=maxiter, seed=seed, n_devices=n_devices)
    if verbose:
        print(f"  multi-start L-BFGS-MT (cov): {n_starts} starts, best Kf={int(Kf)} val_pLL={float(pll1):.3f}", flush=True)
    x_core = dict(zip(names_core, [float(v) for v in nat1[:p_core]]))
    h = GS.build_h(x_core, int(Kf), daily, period, D_max, threshold=True)
    cov_scales = nat1[p_core:]
    col_sigmas = np.zeros(ncol)
    for gi, idx in enumerate(cov_groups):
        col_sigmas[np.asarray(idx)] = float(cov_scales[gi])
    return dict(h=h, kappa=float(x_core["kappa"]), link_scale=float(x_core["link_scale"]),
                K_fourier=int(Kf), val_pll=float(pll1), score=float(pll1), objective="fv_multistart",
                x=x_core, col_sigmas=col_sigmas, cov_scales=[float(s) for s in cov_scales],
                refine=None, n_evals=int(n_starts))
