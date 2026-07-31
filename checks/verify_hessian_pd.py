"""Hessian-PD + drift-rate probe for the referee fixes.

For all 7 constrained cells (deterministic protocol: delta=1e-4, 2x, 20 starts, seed 0):
  (1) smallest eigenvalue of the EXACT inner Hessian  H = hessian_theta F  at
      (a) the selection winner's inner train-window optimum theta_hat (the H used by the
          bilevel adjoint), and
      (b) the full-period constrained refit theta* (the H in Appendix C's formulas).
      The IFT hypergradient and the Sherman-Morrison denominator claims need H positive
      definite at these points -- the objective is NOT globally convex (NB likelihood), so
      this is the numerical verification the paper will now cite.
  (2) the infimal guaranteed drift rate rho0 = root of  sum_d (f_hat(d))_+ rho^{-d} = 1
      at the fitted kernel (bisection). At binding fits R_+ ~ 0.9999 this shows how close
      to one the guaranteed contraction is (the certificate is qualitative at the boundary).
"""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HAWKES = os.environ.get("HAWKES_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HAWKES_ROOT"] = HAWKES
sys.path.insert(0, HAWKES)
from library import datasets
from library import covariates as CV
from library import gpdhp_cfv_grad as CFV
from library import gpdhp_select as GS
from library.lbfgs_mt import minimize_bfgs_mt
from library._gpdhp_common import build_design, softplus_link, nb2_loglik

DELTA = 1e-4; MULT = 2.0; NST = 20; SEED = 0; ND = 10
MAXIT = int(300 * MULT); INNER = int(500 * MULT); REFIT = int(800 * MULT)
CELLS = ["dengue", "crypto", "rand", "nyc", "nyc/cov", "gva", "gva/cov"]


def rho0_of(fpos):
    """Root of sum_d fpos_d * rho^{-d} = 1 in (0,1) -- infimal guaranteed drift rate."""
    d = np.arange(1, len(fpos) + 1)
    phi = lambda r: np.sum(fpos * r ** (-d.astype(float)))
    lo, hi = 1e-6, 1.0
    if phi(1.0) >= 1.0:
        return 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if phi(mid) > 1 else (lo, mid)
    return 0.5 * (lo + hi)


print(f"{'cell':10} {'eigmin(sel H)':>14} {'eigmin(refit H)':>16} {'R_+':>9} {'rho0':>10} {'1-rho0':>10}")
allmins = []
for cell in CELLS:
    t0 = time.time()
    ds = cell.split("/")[0]; cov = cell.endswith("/cov")
    y_dev, y_test, period, daily = datasets.load_split(ds)
    names = GS.names_for(daily)
    if cov:
        C_dev, C_test, _ = CV.build_covariates(ds, root=HAWKES)
        groups = [np.array([0]), np.arange(1, C_dev.shape[1])]
        sel = CFV.select_constrained_cov(y_dev, C_dev, period, daily, groups, delta=DELTA,
                                         n_starts=NST, maxiter=MAXIT, seed=SEED, n_devices=ND,
                                         use_fft=True, verbose=False, inner_maxiter=INNER)
        nat = np.array([sel["x"][nm] for nm in names] + list(sel["cov_scales"]))
        cf = CFV.make_cfv_fns(np.asarray(y_dev, float), period, daily, sel["K_fourier"],
                              delta=DELTA, use_fft=True, C=np.asarray(C_dev, float),
                              cov_groups=groups, inner_maxiter=INNER)
    else:
        sel = CFV.select_constrained(y_dev, period, daily, delta=DELTA, n_starts=NST,
                                     maxiter=MAXIT, seed=SEED, n_devices=ND, use_fft=True,
                                     verbose=False, inner_maxiter=INNER)
        nat = np.array([sel["x"][nm] for nm in names])
        cf = CFV.make_cfv_fns(np.asarray(y_dev, float), period, daily, sel["K_fourier"],
                              delta=DELTA, use_fft=True, inner_maxiter=INNER)
    natj = jnp.asarray(nat)

    # (a) selection-window inner optimum + exact Hessian of F_z (the adjoint's H)
    th_sel, gam_sel = cf["inner_map_c"](natj)
    H_sel = np.asarray(jax.hessian(lambda z: cf["F_z"](z, natj))(th_sel))
    emin_sel = float(np.linalg.eigvalsh(0.5 * (H_sel + H_sel.T))[0])

    # (b) full-period constrained refit theta* + Hessian of the full-period F (Appendix C's H)
    h = sel["h"]; y_devn = np.asarray(y_dev, float); T = len(y_devn); budget = 1.0 - DELTA
    des = build_design(y_devn, h)
    q_b, q_g = des["q_b"], des["q_g"]
    mj, Lj = jnp.asarray(des["m_NB"]), jnp.asarray(des["L"])
    offset = jnp.asarray(des["offset"])
    if cov:
        cs = jnp.asarray(sel["col_sigmas"], float); ncol = int(cs.shape[0])
        XL = jnp.asarray(des["X"]) @ Lj if q_g else jnp.zeros((T, 0))
        A = jnp.concatenate([jnp.asarray(des["B"]), jnp.asarray(C_dev) * cs, XL], axis=1)
        gsl = q_b + ncol
    else:
        A, gsl = jnp.asarray(des["A"]), q_b
    ydev = jnp.asarray(y_devn); dim = A.shape[1]
    kappa, lk = sel["kappa"], sel["link_scale"]
    Rplus = lambda th: jnp.sum(jnp.maximum(mj + (Lj @ th[gsl:] if q_g else 0.0), 0.0))
    def F(th):
        mu = softplus_link(offset + A @ th, lk, 1e-6)
        return -nb2_loglik(ydev, mu, kappa) + 0.5 * jnp.sum(th ** 2)
    th = jnp.zeros(dim); gam = 0.0
    for rho in (30., 100., 300., 1e3, 3e3, 1e4, 3e4, 1e5, 3e5):
        def Lr(z, g=gam, r=rho):
            hinge = jnp.maximum(0.0, g + r * (Rplus(z) - budget))
            return F(z) + (0.5 / r) * (hinge ** 2 - g ** 2)
        th = minimize_bfgs_mt(Lr, th, maxiter=REFIT, gtol=1e-8, ftol=1e-14).x
        gam = jnp.maximum(0.0, gam + rho * (Rplus(th) - budget))
    H_ref = np.asarray(jax.hessian(F)(th))
    emin_ref = float(np.linalg.eigvalsh(0.5 * (H_ref + H_ref.T))[0])

    f_hat = np.asarray(mj + (Lj @ th[gsl:] if q_g else 0.0), float)
    Rp = float(np.maximum(f_hat, 0).sum())
    r0 = rho0_of(np.maximum(f_hat, 0))
    allmins += [emin_sel, emin_ref]
    print(f"{cell:10} {emin_sel:16.9f} {emin_ref:18.9f} {Rp:9.5f} {r0:10.7f} {1-r0:10.2e}"
          f"   ({time.time()-t0:.0f}s)", flush=True)

print(f"\nMIN eigenvalue across all 14 Hessians: {min(allmins):.9f}")
print("done_marker")
