"""Standalone GP-DHP runner (one dataset, one covariate variant); the run_nonneural
orchestrator launches it as a single sequential JAX subprocess (one GPU tenant at a time).

  python gpdhp_runner.py --dataset nyc --variant cov   --out /path/nyc_cov_gpdhp.npz
  python gpdhp_runner.py --dataset nyc --variant nocov --out /path/nyc_nocov_gpdhp.npz
  python gpdhp_runner.py --dataset dengue --variant nocov --out ...      # weekly, no cov

Selection: multi-start L-BFGS-MT forward-validation on a single last-40% window (gpdhp_select).
Then a final collapsed-latent MAP fit on the full fitting period; writes the per-obs NB log-scores
+ observed test counts + selected hypers to an npz.

(A fully Bayesian sampling variant has been removed from the paper; only the MAP fit remains.)"""
import argparse
import json
import os
import sys


def _setup_n_devices():
    """Under --n-devices>1 the multi-start L-BFGS-MT trajectories are evaluated across a thread
    pool; disable XLA intra-op (Eigen) + BLAS threading BEFORE `import jax` so the N worker
    threads each use ~1 core instead of oversubscribing."""
    nd = None
    for i, tok in enumerate(sys.argv):
        if tok == "--n-devices" and i + 1 < len(sys.argv):
            nd = sys.argv[i + 1]
        elif tok.startswith("--n-devices="):
            nd = tok.split("=", 1)[1]
    nd = nd or os.environ.get("GPDHP_N_DEVICES")
    if nd and int(nd) > 1:
        if "xla_cpu_multi_thread_eigen" not in os.environ.get("XLA_FLAGS", ""):
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                                       + " --xla_cpu_multi_thread_eigen=false").strip()
        os.environ.setdefault("OMP_NUM_THREADS", "1")


_setup_n_devices()

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

HERE = os.path.dirname(os.path.abspath(__file__))                       # papercode/experiments
HAWKES = os.environ.get("HAWKES_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(HERE))

from library import gpdhp_select
from library._gpdhp_fft import build_fold_model_fft, build_cov_model_fft   # FFT excitation (default)
from library.lbfgs_mt import minimize_bfgs_mt
from library import covariates as CV
from library import datasets                        # DATA + load_split (shared)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(datasets.DATA))
    ap.add_argument("--variant", default="nocov", choices=["nocov", "cov"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-starts", type=int, default=20,
                    help="number of random hyperparameter initializations refined by L-BFGS-MT (multi-start)")
    ap.add_argument("--maxiter", type=int, default=300,
                    help="max L-BFGS-MT steps per random start (on the analytic FV bilevel hypergradient). "
                         "kappa is a sampled hyperparameter.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-devices", type=int, default=10,
                    help="number of CPU cores to run the multi-start L-BFGS-MT trajectories over via a "
                         "thread pool; 1 = serial. Intra-op (Eigen) threading is disabled automatically "
                         "so the N workers don't oversubscribe.")
    a = ap.parse_args()
    if a.variant == "cov" and not CV.has_covariates(a.dataset):
        sys.exit(f"{a.dataset} has no covariates; use --variant nocov")
    if a.n_devices > 1:
        print(f"[gpdhp] multi-start L-BFGS-MT threaded over {a.n_devices} CPU cores", flush=True)

    y_dev, y_test, period, daily = datasets.load_split(a.dataset)
    # Selection is MULTI-START L-BFGS-MT on the 40% held-out forward-validation pLL (kappa a sampled
    # hyperparameter). For the cov variant the covariate group prior scales are selected JOINTLY in the
    # same bilevel (select_fv_cov) -- no grid search.
    covariates_label = "none"
    col_sigmas = None
    group_sigmas = None
    if a.variant == "cov":
        cov_dev, cov_test, _ = CV.build_covariates(a.dataset, root=HAWKES)
        covariates_label = "temp+holidays"
        from library import gpdhp_fv_grad as FG
        cov_groups = [np.array([0]), np.arange(1, cov_dev.shape[1])]      # temp | holidays
        sel = FG.select_fv_cov(y_dev, cov_dev, period, daily, cov_groups,
                               n_starts=a.n_starts, maxiter=a.maxiter, seed=a.seed, n_devices=a.n_devices)
        col_sigmas = sel["col_sigmas"]                       # per-COLUMN prior scales (group scale broadcast)
        group_sigmas = sel.get("cov_scales")                 # per-GROUP selected scales [temp, holidays]
    else:
        sel = gpdhp_select.select(y_dev, period, daily=daily, min_train=0.6,
                                  n_starts=a.n_starts, maxiter=a.maxiter, seed=a.seed, n_devices=a.n_devices)
    h, kappa, lk = sel["h"], sel["kappa"], sel["link_scale"]
    print(f"[gpdhp {a.dataset}/{a.variant}] selected Kf={sel['K_fourier']} val_pLL={sel['val_pll']:.2f} "
          f"({sel['n_evals']} starts)", flush=True)

    if a.variant == "cov":
        fm = build_cov_model_fft(y_dev, cov_dev, cov_test, y_test, h, kappa, lk, col_sigmas)
        latent_dim = fm["dim"]
    else:
        fm = build_fold_model_fft(y_dev, y_test, h, kappa, lk)
        latent_dim = fm["q"]

    theta = minimize_bfgs_mt(fm["nlp"], jnp.zeros(latent_dim), maxiter=800, gtol=1e-7, ftol=1e-13).x
    mls = np.asarray(fm["test_logscore"](theta), float)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    np.savez_compressed(a.out, dataset=a.dataset, variant=a.variant, covariates=covariates_label,
        map_logscores=mls, observed=y_test, pll_map=float(mls.sum()),
        sel_hypers=json.dumps(h), sel_kappa=float(kappa), sel_link=float(lk), sel_Kf=int(sel["K_fourier"]),
        val_pll=float(sel["val_pll"]),
        sel_col_sigmas=json.dumps([float(s) for s in col_sigmas]) if col_sigmas is not None else "[]",
        sel_group_sigmas=json.dumps([float(s) for s in group_sigmas]) if group_sigmas is not None else "[]")
    print(f"[gpdhp {a.dataset}/{a.variant}] MAP {mls.sum():.2f} "
          f"| n_test={len(mls)} | val_pLL={sel['val_pll']:.2f} -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
