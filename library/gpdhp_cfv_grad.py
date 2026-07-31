"""CONSTRAINED forward-validation bilevel hypergradient for softplus GP-DHP under the Costa stability
constraint  R_+(theta) = sum_d ( m_NB(psi)_d + (L(psi) theta_g)_d )_+ <= 1 - delta.

Inner problem (lower level):   theta*(psi) = argmin_theta F_tr(theta; psi)  s.t. R_+(theta,psi) <= 1-delta
solved to the KKT point by an AUGMENTED LAGRANGIAN (a few ramped subproblems, each an unconstrained
softplus-GP-DHP MAP by minimize_bfgs_mt); returns (theta*, gamma*) with gamma* the budget multiplier.

Outer hypergradient (the BORDERED-KKT adjoint -- the constrained analogue of gpdhp_fv_grad):
KKT stationarity  grad_theta F + gamma grad_theta R_+ = 0  with  R_+ = 1-delta  (when active) gives,
because R_+ is piecewise-LINEAR in theta (no curvature),

    [ H   g ] [lambda]   [ dG/dtheta ]        H = grad^2_theta F_tr  (UNCHANGED from the unconstrained case)
    [ g^T 0 ] [  mu  ] = [     0     ] ,       g = grad_theta R_+  (subgradient over the positive lags),

    dG/dpsi = dG/dpsi|_theta*  -  lambda^T d/dpsi grad_theta(F + gamma R_+)  -  mu d/dpsi R_+ .

Implemented as the FINITE-RHO IFT of the last ALM subproblem: theta* is stationary for
F + (1/2 rho)max(0, gamma_prev + rho c)^2, i.e.  grad F + gamma* grad R_+ = 0  with Hessian
H + a rho_last g g^T, where a = 1{gamma_prev + rho_last c > 0} is the hinge-branch indicator
(R_+ piecewise-linear => no curvature term beyond the active-branch rank-one), solved by
Sherman-Morrison on the same two H-solves.  As rho_last -> inf this recovers the bordered projection at feasible pinned
points; when the cap is slack (gamma*=0) it reduces EXACTLY to the unconstrained adjoint of
gpdhp_fv_grad; and it stays EXACT where the inner fit cannot reach the budget -- the regime the
bordered projection gets wrong (it enforces dR_+/dpsi = 0, which fails off the constraint surface).

OUTER FEASIBILITY PENALTY:  the outer value is  G~ = val NLL + pen_mu * max(0, R_+(theta*) - budget)^2.
val alone is BLIND to infeasibility, and infeasible-in-practice hypers exist: R_+(m_NB) = K_NB
exactly (nb_kernel is normalised) with K_NB's bound 1.25 > 1, and a near-off GP needs
theta_g ~ 1/sigma_u to compensate, which the unit ridge prohibits -- that is how nyc/nocov once
selected a supercritical R_+=1.2491 fit.  The hinge steers the multi-start away from that region
(this is why the finite-rho adjoint matters: the bordered projection would cancel the penalty's
gradient exactly), vanishes identically on feasible fits, and the final guard remains asserting
achieved R_+ < 1 on the refit.  A jax.custom_vjp (neg_fv) exposes G~ with this analytic gradient
for the same multi-start L-BFGS-MT outer search.  softplus link (smooth F, clean Hessian); the
constraint alone gives the Costa guarantee (h_s(x) <= x_+ + s log2).

Nocov + FFT excitation; single K_fourier per builder (round-robin outside).  Guarantee caveats:
strict delta>0 margin, bounded baseline, finite kappa -> geometric ergodicity (constant/periodic
baseline) or uniform-in-t moment bounds (time-varying covariate baseline).
"""
import os, sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

_HERE = os.path.dirname(os.path.abspath(__file__))
_HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(_HERE))

from .lbfgs_mt import minimize_bfgs_mt
from ._gpdhp_common import _fourier_design, nb_kernel, softplus_link, nb2_loglik
from ._gpdhp_fft import _causal_conv, _next_fast_len
from . import gpdhp_select as GS
from .gpdhp_fv_grad import _damp_H

FLOOR = GS.FLOOR


def make_cfv_fns(y_fit, period, daily, Kf, D_max=100, min_train=0.6, delta=0.02, use_fft=True,
                 C=None, cov_groups=None, inner_maxiter=500,
                 alm_rhos=(30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5), pen_mu=1e5):
    """Constrained-FV callables for one K_fourier.  budget = 1 - delta.  Optional covariate block C
    (columns scaled by per-group prior scales appended to `nat`), mirroring gpdhp_fv_grad.make_fv_fns.
    `pen_mu` weights the OUTER feasibility penalty (module docstring): neg_fv returns
    G~ = val NLL + pen_mu * max(0, R_+(theta*) - budget)^2, zero on feasible fits."""
    y = jnp.asarray(np.asarray(y_fit, float)); T = len(np.asarray(y_fit))
    ncut = int(np.floor(min_train * T)); tr = jnp.arange(ncut); va = jnp.arange(ncut, T)
    X = GS._lag_matrix(y_fit, D_max); D = D_max
    t = jnp.arange(1, T + 1, dtype=jnp.float64)
    q_b = 1 + 2 * Kf + (2 * 3 if daily else 0) + 1
    p_core = 11 + (1 if daily else 0)
    has_cov = C is not None
    if has_cov:
        Cj = jnp.asarray(np.asarray(C, float)); ncol = int(Cj.shape[1])
        gidx = [jnp.asarray(np.asarray(ix)) for ix in cov_groups]; n_groups = len(gidx)
    else:
        Cj, ncol, gidx, n_groups = None, 0, [], 0
    dim = q_b + ncol + D; budget = 1.0 - delta
    ytr, yva = y[tr], y[va]
    Lfft = _next_fast_len(T + D + 1)
    y_rfft = jnp.fft.rfft(y, n=Lfft)

    def _col_sigmas(cov_scales):
        cs = jnp.zeros(ncol)
        for gi in range(n_groups):
            cs = cs.at[gidx[gi]].set(cov_scales[gi])
        return cs

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

    def _kernel_pieces(nat):
        (s_level, s_fou, s_lin, K_NB, mean_lag, size, kappa,
         sigma_u, ell_u, beta, lk, s_week) = _unpack(nat)
        m_NB = K_NB * nb_kernel(D, mean_lag, size)
        Lc = GS._warped_rbf_chol(D, beta, sigma_u, ell_u)
        B = _baseline(s_level, s_fou, s_week, s_lin)
        return B, m_NB, Lc, lk, kappa

    def kernel(theta, nat):
        _, m_NB, Lc, _, _ = _kernel_pieces(nat)
        return m_NB + Lc @ theta[q_b + ncol:]             # f(d) = m_NB + (L theta_g)(d)

    def Rplus(theta, nat):
        return jnp.sum(jnp.maximum(kernel(theta, nat), 0.0))

    def _eta(theta, nat):
        B, m_NB, Lc, lk, kappa = _kernel_pieces(nat)
        tb = theta[:q_b]; tg = theta[q_b + ncol:]
        base = B @ tb
        if has_cov:
            base = base + (Cj * _col_sigmas(nat[p_core:])) @ theta[q_b:q_b + ncol]
        if use_fft:
            exc = _causal_conv(y_rfft, m_NB + Lc @ tg, Lfft, T)
        else:
            exc = X @ (m_NB + Lc @ tg)
        return base + exc, lk, kappa

    def F_z(theta, nat):                                  # train MAP objective (softplus link)
        eta, lk, kappa = _eta(theta, nat)
        mu = softplus_link(eta[tr], lk, FLOOR)
        return -nb2_loglik(ytr, mu, kappa) + 0.5 * jnp.sum(theta ** 2)

    def G_z(theta, nat):                                  # validation NLL
        eta, lk, kappa = _eta(theta, nat)
        mu = softplus_link(eta[va], lk, FLOOR)
        return -nb2_loglik(yva, mu, kappa)

    @jax.jit
    def inner_map_c(nat):
        """Augmented-Lagrangian constrained MAP -> (theta*, gamma*)."""
        theta = jnp.zeros(dim); gamma = 0.0
        for rho in alm_rhos:
            def Lrho(th, gam=gamma, r=rho):
                c = Rplus(th, nat) - budget
                hinge = jnp.maximum(0.0, gam + r * c)
                return F_z(th, nat) + (0.5 / r) * (hinge ** 2 - gam ** 2)
            theta = minimize_bfgs_mt(Lrho, theta, maxiter=inner_maxiter, gtol=1e-8, ftol=1e-14).x
            gamma = jnp.maximum(0.0, gamma + rho * (Rplus(theta, nat) - budget))
        return theta, gamma

    @jax.jit
    def fv_pll(nat):
        th, _ = inner_map_c(nat)
        return -G_z(th, nat)

    rho_last = float(alm_rhos[-1])

    def pen_viol(theta, nat):
        return jnp.maximum(0.0, Rplus(theta, nat) - budget)

    def G_pen(theta, nat):                                # outer value: val NLL + feasibility hinge
        v = pen_viol(theta, nat)
        return G_z(theta, nat) + pen_mu * v * v

    @jax.jit
    def analytic_cfv_grad(theta, gamma, nat):
        """d(penalised pLL_val)/dnat -- FINITE-RHO IFT adjoint of the last ALM subproblem.

        theta* satisfies grad F + gamma* grad R_+ = 0 with gamma* = max(0, gamma_prev + rho_last c)
        (the ALM update), i.e. it is stationary for the rho_last subproblem whose Hessian is
        H + a rho_last g g^T with a = 1{gamma* > 0} the hinge-branch indicator (R_+ piecewise-
        linear); every rho_last term below is gated on `active` = a, so on the slack branch the
        adjoint IS the unconstrained one.  Sherman-Morrison on the two existing H-solves;
        the cross term a rho_last (lam.g) dR_+/dnat carries the constraint-surface motion.  As
        rho_last -> inf this recovers the bordered-KKT projection at feasible pinned points, and --
        unlike the projection, which enforces dR_+/dpsi = 0 and so cancels any outer function of
        R_+ exactly -- it keeps the feasibility penalty's gradient alive in the infeasible region,
        where R_+*(psi) genuinely moves with psi (the signal that steers the search back)."""
        viol = pen_viol(theta, nat)
        g = jax.grad(lambda th: Rplus(th, nat))(theta)          # subgradient of R_+ (positive lags)
        gRh = jax.grad(lambda nt: Rplus(theta, nt))(nat)        # dR_+/dnat (explicit, theta frozen)
        gGz = jax.grad(lambda th: G_z(th, nat))(theta) + 2.0 * pen_mu * viol * g
        gGh = jax.grad(lambda nt: G_z(theta, nt))(nat) + 2.0 * pen_mu * viol * gRh
        H = _damp_H(jax.hessian(lambda th: F_z(th, nat))(theta))
        lam0 = jnp.linalg.solve(H, gGz)
        hg = jnp.linalg.solve(H, g)
        active = gamma > 1e-9                                   # last subproblem's hinge was active
        denom = 1.0 + rho_last * jnp.dot(g, hg)                 # >= 1 (H PD): well-conditioned
        lam = jnp.where(active, lam0 - (rho_last * jnp.dot(g, lam0) / denom) * hg, lam0)
        gam_eff = jnp.where(active, gamma, 0.0)
        c1 = jax.grad(lambda nt: jnp.dot(
            lam, jax.grad(lambda th: F_z(th, nt) + gam_eff * Rplus(th, nt))(theta)))(nat)
        c2 = jnp.where(active, rho_last * jnp.dot(lam, g), 0.0) * gRh
        return -(gGh - c1 - c2)                                 # d(pLL~)/dnat = -d(G~)/dnat

    def grad_fv(nat):
        nat = jnp.asarray(np.asarray(nat, float))
        th, gam = inner_map_c(nat)
        return np.asarray(analytic_cfv_grad(th, gam, nat)), float(-G_z(th, nat)), th, float(gam)

    @jax.custom_vjp
    def neg_fv(nat):
        th, _ = inner_map_c(nat)
        return G_pen(th, nat)

    def _fwd(nat):
        th, gam = inner_map_c(nat)
        return G_pen(th, nat), (nat, th, gam)

    def _bwd(res, gbar):
        nat, th, gam = res
        return (jnp.asarray(gbar) * (-analytic_cfv_grad(th, gam, nat)),)

    neg_fv.defvjp(_fwd, _bwd)

    return dict(F_z=F_z, G_z=G_z, G_pen=G_pen, pen_viol=pen_viol, Rplus=Rplus, kernel=kernel,
                inner_map_c=inner_map_c, fv_pll=fv_pll, analytic_cfv_grad=analytic_cfv_grad,
                grad_fv=grad_fv, neg_fv=neg_fv, dim=dim, q_b=q_b, budget=budget, pen_mu=pen_mu)


def constrained_refit_score(y_dev, y_test, h, kappa, link_scale, delta=0.02,
                            alm_rhos=(30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5),
                            refit_maxiter=800):
    """Refit the constrained softplus MAP on the FULL dev period at fixed hypers (R_+ <= 1-delta,
    augmented Lagrangian), then score the held-out test set one-step-ahead (combined-history
    excitation).  Returns (test_logscores, R_+).  `refit_maxiter` is the per-rho inner BFGS budget."""
    from ._gpdhp_common import build_design, _baseline_cols, lag_matrix_future
    y_dev = np.asarray(y_dev, float); y_test = np.asarray(y_test, float)
    T = len(y_dev); n_test = len(y_test); D = int(h["D_max"]); budget = 1.0 - delta
    des = build_design(y_dev, h)
    A, offset, B, L, m_NB, q_b, q_g = (des["A"], des["offset"], des["B"], des["L"], des["m_NB"],
                                       des["q_b"], des["q_g"])
    Aj, offj, mj, Lj = map(jnp.asarray, (A, offset, m_NB, L)); ydev = jnp.asarray(y_dev)
    dim = A.shape[1]

    def Rplus(theta):
        return jnp.sum(jnp.maximum(mj + (Lj @ theta[q_b:] if q_g else 0.0), 0.0))

    def F(theta):
        mu = softplus_link(offj + Aj @ theta, link_scale, FLOOR)
        return -nb2_loglik(ydev, mu, kappa) + 0.5 * jnp.sum(theta ** 2)

    theta = jnp.zeros(dim); gamma = 0.0
    for rho in alm_rhos:
        def Lrho(th, gam=gamma, r=rho):
            hinge = jnp.maximum(0.0, gam + r * (Rplus(th) - budget))
            return F(th) + (0.5 / r) * (hinge ** 2 - gam ** 2)
        theta = minimize_bfgs_mt(Lrho, theta, maxiter=refit_maxiter, gtol=1e-8, ftol=1e-14).x
        gamma = jnp.maximum(0.0, gamma + rho * (Rplus(theta) - budget))
    rplus = float(Rplus(theta))
    # one-step test score
    tt = jnp.arange(T + 1, T + n_test + 1, dtype=jnp.float64)
    B_te = _baseline_cols(tt, h["period"], h); X_te = lag_matrix_future(y_dev, y_test, D)
    tb = theta[:q_b]; f = mj + (Lj @ theta[q_b:] if q_g else 0.0)
    from ._gpdhp_common import nb2_logpmf_vec
    eta = B_te @ tb + X_te @ f
    ls = np.asarray(nb2_logpmf_vec(jnp.asarray(y_test), softplus_link(eta, link_scale, FLOOR), kappa), float)
    return ls, rplus


# Feasibility across hypers: R_+(m_NB) = K_NB exactly (nb_kernel is normalised) and K_NB's bound
# runs to 1.25 > 1, so hypers exist whose constrained inner fit cannot reach the budget in practice
# (theta_g ~ 1/sigma_u is ridge-prohibitive when the GP is near-off) -- and plain val_pLL is blind
# to the violation (nyc/nocov once selected a supercritical R_+=1.2491 fit that way).  The outer
# feasibility penalty (pen_mu; module docstring) steers the multi-start away from that region.
# Capping K_NB at the budget was tried and REVERTED: sufficient but not necessary -- it excluded
# dengue's valid K_NB>budget + inhibitory-GP fit, costing ~4 nats and 2 DM wins.
# ALWAYS assert achieved R_+ < 1 on the final refit.


def select_constrained(y_fit, period, daily, D_max=100, delta=0.02, n_starts=20, maxiter=300,
                       seed=0, n_devices=10, use_fft=True, verbose=False, inner_maxiter=500,
                       pen_mu=1e5):
    """Constrained GP-DHP FV selection: multi-start L-BFGS-MT on the finite-rho constrained
    hypergradient with the outer feasibility penalty (pen_mu).  Same protocol/return as
    gpdhp_select.select.  `maxiter` is the OUTER multistart budget; `inner_maxiter` the per-rho
    ALM inner BFGS budget.  Returned val_pll is the CLEAN validation pLL (penalty excluded)."""
    from .jax_baselines import multistart_lbfgs
    y_fit = np.asarray(y_fit, float)
    names = GS.names_for(daily); dims = len(names)
    cfv = {Kf: make_cfv_fns(y_fit, period, daily, Kf, D_max, delta=delta, use_fft=use_fft,
                            inner_maxiter=inner_maxiter, pen_mu=pen_mu)
           for Kf in (1, 2, 3)}
    neg_fvs = {Kf: cfv[Kf]["neg_fv"] for Kf in (1, 2, 3)}
    bounds = [GS.BOUNDS[nm] for nm in names]
    sample_fn = lambda rng: GS._unit_to_nat(rng.random(dims), names)
    nat, pll, Kf = multistart_lbfgs(neg_fvs, [1, 2, 3], bounds, sample_fn,
                                    n_starts=n_starts, maxiter=maxiter, seed=seed, n_devices=n_devices)
    x = dict(zip(names, [float(v) for v in nat]))
    h = GS.build_h(x, int(Kf), daily, period, D_max, threshold=False)   # keep GP block for the constraint
    natj = jnp.asarray(nat)
    th, gam = cfv[int(Kf)]["inner_map_c"](natj)
    rplus = float(cfv[int(Kf)]["Rplus"](th, natj))
    val_clean = -float(cfv[int(Kf)]["G_z"](th, natj))
    viol = max(0.0, rplus - (1.0 - delta))
    if verbose:
        print(f"  constrained multi-start: best Kf={int(Kf)} val_pLL={val_clean:.3f} "
              f"(penalised {float(pll):.3f}) R_+={rplus:.7f} (budget {1-delta:.7f}) "
              f"gamma={float(gam):.3g} viol={viol:.2e}", flush=True)
    return dict(h=h, kappa=float(x["kappa"]), link_scale=float(x["link_scale"]), K_fourier=int(Kf),
                val_pll=val_clean, val_pll_pen=float(pll), R_plus=rplus, gamma=float(gam),
                pen_viol=viol, x=x, delta=delta, objective="cfv_multistart", n_evals=int(n_starts))


def select_constrained_cov(y_fit, C_fit, period, daily, cov_groups, D_max=100, delta=0.02,
                           n_starts=20, maxiter=300, seed=0, n_devices=10, cov_bounds=(1e-3, 20.0),
                           use_fft=True, verbose=False, inner_maxiter=500, pen_mu=1e5):
    """Covariate constrained GP-DHP FV selection (cov analogue of select_constrained; mirrors
    gpdhp_fv_grad.select_fv_cov): core hypers + one prior scale per covariate group, with the outer
    feasibility penalty (pen_mu).  `maxiter` is the OUTER multistart budget; `inner_maxiter` the
    per-rho ALM inner BFGS budget (the ramp runs one solve per rho in alm_rhos).  Returned val_pll
    is the CLEAN validation pLL (penalty excluded)."""
    from .jax_baselines import multistart_lbfgs
    y_fit = np.asarray(y_fit, float); C_fit = np.asarray(C_fit, float); ncol = C_fit.shape[1]
    names_core = GS.names_for(daily); p_core = len(names_core); n_groups = len(cov_groups)
    dims = p_core + n_groups
    bnds = [GS.BOUNDS[nm] for nm in names_core] + [tuple(cov_bounds)] * n_groups
    logf = np.array([GS.LOG[nm] for nm in names_core] + [1] * n_groups, bool)
    lo = np.array([b[0] for b in bnds]); hi = np.array([b[1] for b in bnds])
    lo_l = np.where(logf, lo, 1.0); hi_l = np.where(logf, hi, 1.0)
    sample_fn = lambda rng: np.where(logf, np.exp(np.log(lo_l) + rng.random(dims) * (np.log(hi_l) - np.log(lo_l))),
                                     lo + rng.random(dims) * (hi - lo))
    cfv = {Kf: make_cfv_fns(y_fit, period, daily, Kf, D_max, delta=delta, use_fft=use_fft,
                            C=C_fit, cov_groups=cov_groups, inner_maxiter=inner_maxiter,
                            pen_mu=pen_mu)
           for Kf in (1, 2, 3)}
    neg_fvs = {Kf: cfv[Kf]["neg_fv"] for Kf in (1, 2, 3)}
    nat1, pll1, Kf = multistart_lbfgs(neg_fvs, [1, 2, 3], bnds, sample_fn,
                                      n_starts=n_starts, maxiter=maxiter, seed=seed, n_devices=n_devices)
    x_core = dict(zip(names_core, [float(v) for v in nat1[:p_core]]))
    h = GS.build_h(x_core, int(Kf), daily, period, D_max, threshold=False)   # keep GP block for the constraint
    cov_scales = nat1[p_core:]; col_sigmas = np.zeros(ncol)
    for gi, idx in enumerate(cov_groups):
        col_sigmas[np.asarray(idx)] = float(cov_scales[gi])
    natj = jnp.asarray(nat1)
    th, gam = cfv[int(Kf)]["inner_map_c"](natj)
    rplus = float(cfv[int(Kf)]["Rplus"](th, natj))
    val_clean = -float(cfv[int(Kf)]["G_z"](th, natj))
    viol = max(0.0, rplus - (1.0 - delta))
    if verbose:
        print(f"  constrained-cov multi-start: Kf={int(Kf)} val_pLL={val_clean:.3f} "
              f"(penalised {float(pll1):.3f}) R_+={rplus:.7f} (budget {1-delta:.7f}) "
              f"gamma={float(gam):.3g} viol={viol:.2e}", flush=True)
    return dict(h=h, kappa=float(x_core["kappa"]), link_scale=float(x_core["link_scale"]),
                K_fourier=int(Kf), val_pll=val_clean, val_pll_pen=float(pll1), R_plus=rplus,
                gamma=float(gam), pen_viol=viol, col_sigmas=col_sigmas,
                cov_scales=[float(s) for s in cov_scales], delta=delta,
                x=x_core, objective="cfv_cov_multistart", n_evals=int(n_starts))


def constrained_refit_score_cov(y_dev, C_dev, C_score, y_test, h, kappa, link_scale, col_sigmas,
                                delta=0.02, alm_rhos=(30.0, 100.0, 300.0, 1e3, 3e3, 1e4, 3e4, 1e5, 3e5),
                                refit_maxiter=800):
    """Covariate constrained softplus MAP on the full dev period (R_+ <= 1-delta, augmented
    Lagrangian), then one-step held-out test score.  Returns (test_logscores, R_+).
    `refit_maxiter` is the per-rho inner BFGS budget of the ALM ramp."""
    from ._gpdhp_common import (build_design, _baseline_cols, lag_matrix_future,
                                                 nb2_logpmf_vec)
    y_dev = np.asarray(y_dev, float); y_test = np.asarray(y_test, float)
    T = len(y_dev); n_test = len(y_test); D = int(h["D_max"]); budget = 1.0 - delta
    des = build_design(y_dev, h)
    B, X, L, m_NB, q_b, q_g = des["B"], des["X"], des["L"], des["m_NB"], des["q_b"], des["q_g"]
    cs = jnp.asarray(col_sigmas, float); ncol = int(cs.shape[0])
    Ctr = jnp.asarray(C_dev) * cs
    XL = jnp.asarray(X) @ jnp.asarray(L) if q_g else jnp.zeros((T, 0))
    A = jnp.concatenate([jnp.asarray(B), Ctr, XL], axis=1)
    offset = jnp.asarray(des["offset"]); mj = jnp.asarray(m_NB); Lj = jnp.asarray(L); ydev = jnp.asarray(y_dev)
    dim = A.shape[1]

    def Rplus(theta):
        return jnp.sum(jnp.maximum(mj + (Lj @ theta[q_b + ncol:] if q_g else 0.0), 0.0))

    def F(theta):
        mu = softplus_link(offset + A @ theta, link_scale, FLOOR)
        return -nb2_loglik(ydev, mu, kappa) + 0.5 * jnp.sum(theta ** 2)

    theta = jnp.zeros(dim); gamma = 0.0
    for rho in alm_rhos:
        def Lrho(th, gam=gamma, r=rho):
            hinge = jnp.maximum(0.0, gam + r * (Rplus(th) - budget))
            return F(th) + (0.5 / r) * (hinge ** 2 - gam ** 2)
        theta = minimize_bfgs_mt(Lrho, theta, maxiter=refit_maxiter, gtol=1e-8, ftol=1e-14).x
        gamma = jnp.maximum(0.0, gamma + rho * (Rplus(theta) - budget))
    rplus = float(Rplus(theta))
    tt = jnp.arange(T + 1, T + n_test + 1, dtype=jnp.float64)
    B_te = _baseline_cols(tt, h["period"], h); X_te = lag_matrix_future(y_dev, y_test, D)
    tb = theta[:q_b]; tc = theta[q_b:q_b + ncol]; f = mj + (Lj @ theta[q_b + ncol:] if q_g else 0.0)
    eta = B_te @ tb + (jnp.asarray(C_score) * cs) @ tc + X_te @ f
    ls = np.asarray(nb2_logpmf_vec(jnp.asarray(y_test), softplus_link(eta, link_scale, FLOOR), kappa), float)
    return ls, rplus
