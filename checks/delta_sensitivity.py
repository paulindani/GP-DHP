"""Stability-margin sensitivity check behind the Section 5.2 claim that "on the most binding
series (GVA with covariates) moving delta across 1e-5 -- 1e-3 changes the held-out
log-likelihood by less than 2.5 nats".

Reruns the full constrained selection + refit (identical protocol/budgets/seed to
supp_tables.py) on the GVA covariate cell at delta in {1e-5, 1e-4, 1e-3} and reports the
held-out pLL per delta plus the max spread.  delta=1e-4 is the paper's production setting
(pLL = -2985.0).  Runtime ~6 min.  Usage:  python delta_sensitivity.py
"""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from library import datasets
from library import covariates as CV
from library import gpdhp_cfv_grad as CFV

MULT = 2.0; NST = 20; SEED = 0; ND = 10
MAXIT = int(300 * MULT); INNER = int(500 * MULT); REFIT = int(800 * MULT)
DELTAS = (1e-5, 1e-4, 1e-3)

if __name__ == "__main__":
    y_dev, y_test, period, daily = datasets.load_split("gva")
    C_dev, C_test, _ = CV.build_covariates("gva", root=os.environ.get("HAWKES_ROOT"))
    groups = [np.array([0]), np.arange(1, C_dev.shape[1])]
    plls = {}
    print(f"{'delta':>8} {'R_+':>10} {'pLL':>10}")
    for delta in DELTAS:
        t0 = time.time()
        sel = CFV.select_constrained_cov(y_dev, C_dev, period, daily, groups, delta=delta,
                                         n_starts=NST, maxiter=MAXIT, seed=SEED, n_devices=ND,
                                         use_fft=True, verbose=False, inner_maxiter=INNER)
        ls, Rp = CFV.constrained_refit_score_cov(y_dev, C_dev, C_test, y_test, sel["h"], sel["kappa"],
                                                 sel["link_scale"], sel["col_sigmas"], delta=delta,
                                                 refit_maxiter=REFIT)
        assert Rp < 1.0
        plls[delta] = float(np.asarray(ls).sum())
        print(f"{delta:>8.0e} {Rp:>10.6f} {plls[delta]:>10.1f}   ({time.time()-t0:.0f}s)", flush=True)
    spread = max(plls.values()) - min(plls.values())
    print(f"\nmax pLL spread across delta in 1e-5..1e-3: {spread:.2f} nats "
          f"(paper Section 5.2 claims < 2.5)")
    print("done_marker")
