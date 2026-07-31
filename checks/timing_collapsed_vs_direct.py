"""Collapsed vs direct full-latent optimization -- the timing check behind Table 1
(tab:collapsed_uncollapsed, Section 5.1.3).

Simulates the Section-5.1.3 DGP -- linear-seasonal baseline b(t) = 0.5 + 0.001 t
+ 0.05 cos(2 pi t/365) + 0.25 sin(2 pi t/365), excitation f(d) = 0.35 q_NB(d; mean 4, size 10)
+ 0.20 q_NB(d; mean 12, size 25) over D_max = 200 lags, NB2 size kappa = 100 -- and fits GP-DHP
twice at the SAME fixed hyperparameters on the first 80% of each series:

  * COLLAPSED : minimize the whitened objective over theta in R^(q_b + D)      (the paper's method)
  * DIRECT    : minimize over the full latent baseline b in R^T_dev and lag component u in R^D,
                with the same NB2 likelihood, softplus link, K_b / K_g priors (via precomputed
                Cholesky factors, built outside the timed optimization), and lag matrix X.

Both use the same L-BFGS-MT optimizer.  Reports, per T in {2000, 5000}: median wall time over
3 replicates (after a JIT warm-up replicate), the speedup, the max RMSE between the two fitted
dev-period intensities, and the max held-out |pLL difference| / n_test (direct-fit baselines are
extended to the test window by the GP conditional mean under the same K_b).

Wall times are machine-specific: the numbers printed in Table 1 came from the paper's reference
laptop, and a rerun reproduces the ORDER (about an order-of-magnitude speedup at machine-precision
agreement), not the exact seconds.  Runtime ~2 min.  Usage:  python timing_collapsed_vs_direct.py
"""
import os, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

import sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from library._gpdhp_common import build_design, _baseline_cols, softplus_link, nb2_loglik, nb2_logpmf_vec, nb_kernel
from library.lbfgs_mt import minimize_bfgs_mt

FLOOR = 1e-6
D_MAX = 200
# fixed hyperparameters for both fits (the check is at FIXED hypers; no selection involved)
H = dict(D_max=D_MAX, period=365.0, K_fourier=1, sigma_level=1.0, sigma_fourier=0.5,
         sigma_lin=1e-3, sigma_weekly=0.0, K_NB=0.35, mean_lag=4.0, size=10.0,
         beta=0.1, sigma_u=0.5, ell_u=10.0)
KAPPA, LK = 100.0, 1.0


def simulate(T, seed):
    rng = np.random.RandomState(seed)
    t = np.arange(1, T + 1, dtype=float)
    b = 0.5 + 0.001 * t + 0.05 * np.cos(2 * np.pi * t / 365) + 0.25 * np.sin(2 * np.pi * t / 365)
    f = 0.35 * np.asarray(nb_kernel(D_MAX, 4.0, 10.0)) + 0.20 * np.asarray(nb_kernel(D_MAX, 12.0, 25.0))
    y = np.zeros(T)
    for i in range(T):
        md = min(D_MAX, i)
        exc = float(np.dot(y[i - md:i][::-1], f[:md])) if md else 0.0
        lam = max(b[i] + exc, 1e-6)
        y[i] = rng.negative_binomial(KAPPA, KAPPA / (KAPPA + lam))
    return y


def lag_matrix(y, D):
    T = len(y)
    X = np.zeros((T, D))
    for d in range(1, D + 1):
        X[d:, d - 1] = y[:T - d]
    return X


def run_T(T, n_reps=3):
    Tdev = int(0.8 * T)
    des0 = build_design(simulate(T, 0)[:Tdev], H)     # shapes/keys only; rebuilt per replicate
    q_b = des0["q_b"]

    tc, td, rmses, dplls = [], [], [], []
    for rep in range(n_reps + 1):                     # rep 0 = JIT warm-up (untimed)
        y = simulate(T, 100 + rep)
        y_dev, y_test = y[:Tdev], y[Tdev:]
        des = build_design(y_dev, H)
        A, offset = jnp.asarray(des["A"]), jnp.asarray(des["offset"])
        B_sc, L, m_NB = np.asarray(des["B"]), np.asarray(des["L"]), np.asarray(des["m_NB"])
        X = lag_matrix(y_dev, D_MAX)
        ydj = jnp.asarray(y_dev)

        # ---------------- collapsed: theta in R^(q_b + D) ----------------------------------- #
        def F_col(th):
            mu = softplus_link(offset + A @ th, LK, FLOOR)
            return -nb2_loglik(ydj, mu, KAPPA) + 0.5 * jnp.sum(th ** 2)
        t0 = time.perf_counter()
        th = minimize_bfgs_mt(F_col, jnp.zeros(A.shape[1]), maxiter=800, gtol=1e-8, ftol=1e-14).x
        jax.block_until_ready(th)
        t_col = time.perf_counter() - t0

        # ---------------- direct: full latent (b, u), same priors via Cholesky -------------- #
        Kb = B_sc @ B_sc.T + 1e-6 * np.eye(Tdev)                    # K_b^num (basis + nugget)
        cKb = jax.scipy.linalg.cho_factor(jnp.asarray(Kb), lower=True)
        Lg = jnp.asarray(L)                                          # chol(K_g) from build_design
        Xj, mj = jnp.asarray(X), jnp.asarray(m_NB)

        def F_dir(z):
            b, u = z[:Tdev], z[Tdev:]
            mu = softplus_link(b + Xj @ (mj + u), LK, FLOOR)
            pb = 0.5 * jnp.dot(b, jax.scipy.linalg.cho_solve(cKb, b))
            pu = 0.5 * jnp.sum(jax.scipy.linalg.solve_triangular(Lg, u, lower=True) ** 2)
            return -nb2_loglik(ydj, mu, KAPPA) + pb + pu
        t0 = time.perf_counter()
        z = minimize_bfgs_mt(F_dir, jnp.zeros(Tdev + D_MAX), maxiter=2000, gtol=1e-8, ftol=1e-14).x
        jax.block_until_ready(z)
        t_dir = time.perf_counter() - t0

        # ---------------- agreement: fitted dev intensity + held-out pLL -------------------- #
        lam_col = np.asarray(softplus_link(offset + A @ th, LK, FLOOR))
        b_hat, u_hat = np.asarray(z[:Tdev]), np.asarray(z[Tdev:])
        lam_dir = np.asarray(softplus_link(jnp.asarray(b_hat + X @ (m_NB + u_hat)), LK, FLOOR))
        rmse = float(np.sqrt(np.mean((lam_col - lam_dir) ** 2)))

        tt = jnp.arange(Tdev + 1, T + 1, dtype=jnp.float64)
        B_te = np.asarray(_baseline_cols(tt, H["period"], H))
        X_te = np.zeros((len(y_test), D_MAX))
        yfull = np.concatenate([y_dev, y_test])
        for d in range(1, D_MAX + 1):
            X_te[:, d - 1] = yfull[Tdev - d:T - d]
        f_col = m_NB + L @ np.asarray(th[q_b:])
        eta_col = B_te @ np.asarray(th[:q_b]) + X_te @ f_col
        # direct baseline extended by the GP conditional mean under the same K_b
        b_te = (B_te @ B_sc.T) @ np.asarray(jax.scipy.linalg.cho_solve(cKb, jnp.asarray(b_hat)))
        eta_dir = b_te + X_te @ (m_NB + u_hat)
        pll = lambda eta: float(np.sum(np.asarray(nb2_logpmf_vec(
            jnp.asarray(y_test), softplus_link(jnp.asarray(eta), LK, FLOOR), KAPPA))))
        dpll = abs(pll(eta_col) - pll(eta_dir)) / len(y_test)

        if rep > 0:                                   # skip the compile-warm-up replicate
            tc.append(t_col); td.append(t_dir); rmses.append(rmse); dplls.append(dpll)
    return (float(np.median(tc)), float(np.median(td)), max(rmses), max(dplls))


if __name__ == "__main__":
    print(f"{'T':>6} {'D_max':>6} {'reps':>5} {'Collapsed(s)':>13} {'Direct(s)':>10} "
          f"{'Speedup':>8} {'MaxRMSE':>10} {'Max|dpLL|/n':>12}")
    for T in (2000, 5000):
        c, d, r, p = run_T(T)
        print(f"{T:>6} {D_MAX:>6} {'3/3':>5} {c:>13.3f} {d:>10.3f} {d/c:>7.2f}x {r:>10.2e} {p:>12.2e}",
              flush=True)
    print("done_marker  (Table 1 reports the reference machine's times; the speedup order and the "
          "machine-precision agreement are the reproducible quantities)")
