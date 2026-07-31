"""Run the all-JAX count-process baselines for one (dataset, variant) and write
per-observation held-out NB log-scores as a CSV (columns model, observed, nb_log_score,
one block per baseline, test points in the orchestrator's order), consumed by the
run_nonneural orchestrator.

All baselines select their hyperparameters (kappa, softplus bandwidth s, ridge
penalty, Fourier scales) by 40% held-out forward validation -- MULTI-START L-BFGS-MT
on the analytic FV bilevel hypergradient (n_starts random inits, keep best); there
is no grid search. JAX is pinned to the CPU here so several of these can run in
parallel while the GPU is busy with GP-DHP / the neural grid.

  python jax_baselines_runner.py --dataset nyc --variant cov --out /path/nyc_cov_jax.csv
"""
import argparse
import json
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")          # keep the GPU free for the neural grid


def _setup_n_devices():
    """Under --n-devices>1 the multi-start L-BFGS-MT trajectories run across a thread pool
    (jax_baselines.multistart_lbfgs); disable XLA intra-op (Eigen) + BLAS threading BEFORE
    importing jax so the N worker threads each use ~1 core rather than oversubscribing."""
    nd = None
    for i, tok in enumerate(sys.argv):
        if tok == "--n-devices" and i + 1 < len(sys.argv):
            nd = sys.argv[i + 1]
        elif tok.startswith("--n-devices="):
            nd = tok.split("=", 1)[1]
    if nd and int(nd) > 1:
        if "xla_cpu_multi_thread_eigen" not in os.environ.get("XLA_FLAGS", ""):
            os.environ["XLA_FLAGS"] = (os.environ.get("XLA_FLAGS", "")
                                       + " --xla_cpu_multi_thread_eigen=false").strip()
        os.environ.setdefault("OMP_NUM_THREADS", "1")


_setup_n_devices()

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))   # papercode
sys.path.insert(0, os.path.dirname(HERE))
from library import jax_baselines as JB
from library import covariates as CV
from library import datasets                        # DATA + load_split_cal (shared)

PARAM_MODELS = ["DHP", "LinearDHP", "DHPsinusoidal", "DHPsinusoidalLinear"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(datasets.DATA))
    ap.add_argument("--variant", default="nocov", choices=["nocov", "cov"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--models", default="parametric,histogram,baseline_only,ingarch",
                    help="which baseline families to run")
    ap.add_argument("--n-starts", type=int, default=20,
                    help="number of random hyperparameter initializations refined by L-BFGS-MT (multi-start)")
    ap.add_argument("--maxiter", type=int, default=300,
                    help="max L-BFGS-MT steps per random start (on the analytic FV bilevel hypergradient)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-devices", type=int, default=10,
                    help="run each baseline's multi-start L-BFGS-MT trajectories across N CPU cores via a "
                         "thread pool; 1 = serial. Intra-op (Eigen) threading is disabled automatically so "
                         "the N workers don't oversubscribe. Selection is the 40%% held-out forward-validation "
                         "pLL (kappa a sampled hyperparameter).")
    a = ap.parse_args()
    fams = [m.strip() for m in a.models.split(",") if m.strip()]
    nstarts, maxit, seed = a.n_starts, a.maxiter, a.seed
    par_fn, hist_fn, bo_fn, ing_fn = (JB.fit_score_parametric_fv, JB.fit_score_histogram_fv,
                                      JB.fit_score_baseline_only_fv, JB.fit_score_ingarch_fv)

    y_dev, y_test, period, daily, cal, dates = datasets.load_split_cal(a.dataset)
    Zdev = Ztest = Zfull = None
    if a.variant == "cov":
        if not CV.has_covariates(a.dataset):
            sys.exit(f"{a.dataset} has no covariates; use --variant nocov")
        Zdev, Ztest, _ = CV.build_covariates(a.dataset, root=HAWKES)
        Zfull = np.vstack([Zdev, Ztest])

    rows, hypers = [], {}

    def emit(logscores, info):
        m = info["model"]; hypers[m] = {k: v for k, v in info.items() if k != "model"}
        rows.append(pd.DataFrame(dict(model=m, observed=y_test, nb_log_score=np.asarray(logscores, float))))
        print(f"  [{a.dataset}/{a.variant}] {m:24} pLL {float(np.sum(logscores)):10.2f} "
              f"| {hypers[m]}", flush=True)

    if a.n_devices > 1:
        print(f"  [{a.dataset}/{a.variant}] multi-start L-BFGS-MT threaded over {a.n_devices} CPU cores", flush=True)
    if "parametric" in fams:
        for mk in PARAM_MODELS:
            ls, info = par_fn(mk, y_dev, y_test, period, Zdev, Ztest, nstarts, maxit, seed,
                              n_devices=a.n_devices)
            emit(ls, info)
    if "histogram" in fams:
        ls, info = hist_fn(y_dev, y_test, period, daily, cal, dates, Zfull,
                           n_starts=nstarts, maxiter=maxit, seed=seed, n_devices=a.n_devices)
        emit(ls, info)
    if "baseline_only" in fams:
        ls, info = bo_fn(y_dev, y_test, period, Zdev, Ztest, n_starts=nstarts, maxiter=maxit, seed=seed,
                         n_devices=a.n_devices)
        emit(ls, info)
    if "ingarch" in fams:
        ls, info = ing_fn(y_dev, y_test, period, daily, cal, dates, Zfull,
                          n_starts=nstarts, maxiter=maxit, seed=seed, n_devices=a.n_devices)
        emit(ls, info)

    out = pd.concat(rows, ignore_index=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out.to_csv(a.out, index=False)
    with open(os.path.splitext(a.out)[0] + "_hypers.json", "w") as fh:
        json.dump(dict(dataset=a.dataset, variant=a.variant, hypers=hypers), fh, indent=2)
    print(f"[jax-baselines {a.dataset}/{a.variant}] wrote {len(rows)} baselines x {len(y_test)} test pts -> {a.out}",
          flush=True)


if __name__ == "__main__":
    main()
