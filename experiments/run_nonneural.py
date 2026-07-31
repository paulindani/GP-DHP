"""Non-neural benchmark orchestrator: GP-DHP (collapsed-latent MAP) and the JAX count-process
baselines (Discrete/Linear/Sinusoidal/Linear+Sinusoidal DHP, Histogram DHP-NB, Baseline-only,
NB-INGARCH), plus two nonparametric-kernel Hawkes benchmarks -- the random-histogram DHP of Browning
(reversible-jump MCMC) and a discrete GP-modulated Hawkes (squared-GP kernel, MAP) -- over ALL datasets
and BOTH covariate variants where covariates exist (nyc, gva).

Execution model (what makes this different from the old run_master.py):
  * STRICTLY SERIAL per (dataset, variant, model) -- no cross-model / cross-cell parallelism.  The
    ONLY parallelism is the multi-start L-BFGS-MT thread pool (--n-devices) INSIDE each fit.
  * GP-DHP is fit FIRST in every cell (it is the DM reference + supplies the observed series).
  * Results are folded into ONE npz AFTER EVERY SINGLE MODEL FIT (bench_shared.fold_model), so the
    file is valid, incremental, and Diebold-Mariano-ready (DM vs GP-DHP(MAP) + Holm) throughout.

Selection is 20-start x 300-iter multi-start L-BFGS-MT on the 40% held-out FV pLL (the defaults),
for GP-DHP and every baseline.  Baselines only -- the neural NB models are a separate orchestrator
(run_neural.py) writing a separate npz.

  python run_nonneural.py                                  # all datasets, full budget
  python run_nonneural.py --datasets dengue,nyc --n-starts 4 --maxiter 50            # smoke
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))              # papercode/experiments
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))  # papercode (local, self-contained)
PYFUN = HERE  # benchmark subprocess targets are co-located in experiments
sys.path.insert(0, os.path.dirname(HERE))
from library import covariates as CV
from library import bench_shared as BS

ORDER = ["dengue", "crypto", "rand", "nyc", "gva"]
BASELINES = ["Discrete DHP", "Linear DHP", "Sinusoidal DHP", "Linear + Sinusoidal DHP",
             "Baseline only", "Histogram DHP-NB", "NB-INGARCH"]
GP_REF = BS.GP_REF                                             # "GP-DHP (MAP)"


def _env(extra=None):
    e = dict(os.environ, HAWKES_ROOT=HAWKES, VECLIB_MAXIMUM_THREADS="2", OMP_NUM_THREADS="2")
    if extra:
        e.update(extra)
    return e


def variants_for(ds):
    return ["nocov", "cov"] if CV.has_covariates(ds) else ["nocov"]


def run_gpdhp(ds, variant, tmp, a):
    """GP-DHP MAP for one cell (subprocess, its own JAX process). Returns the npz path."""
    out = os.path.join(tmp, f"{ds}_{variant}_gpdhp.npz")
    cmd = [sys.executable, os.path.join(HERE, "gpdhp_runner.py"),
           "--dataset", ds, "--variant", variant, "--out", out,
           "--n-starts", str(a.n_starts), "--maxiter", str(a.maxiter),
           "--seed", str(a.seed), "--n-devices", str(a.n_devices)]
    log = os.path.join(tmp, f"gpdhp_{ds}_{variant}.log")
    with open(log, "w") as lf:
        r = subprocess.run(cmd, env=_env(), cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0 or not os.path.exists(out):
        raise RuntimeError(f"GP-DHP failed for {ds}/{variant} (see {log}):\n" + open(log).read()[-1500:])
    return out


def run_baselines(ds, variant, tmp, a):
    """All 7 JAX multistart baselines for one cell (subprocess, CPU-pinned; fits the 7 serially,
    each with the --n-devices thread pool). Returns (csv_path, hypers_dict)."""
    csv = os.path.join(tmp, f"{ds}_{variant}_baselines.csv")
    cmd = [sys.executable, os.path.join(PYFUN, "jax_baselines_runner.py"),
           "--dataset", ds, "--variant", variant, "--out", csv,
           "--n-starts", str(a.n_starts), "--maxiter", str(a.maxiter),
           "--seed", str(a.seed), "--n-devices", str(a.n_devices)]
    log = os.path.join(tmp, f"jaxbl_{ds}_{variant}.log")
    t0 = time.time()
    with open(log, "w") as lf:
        subprocess.run(cmd, env=_env({"JAX_PLATFORMS": "cpu"}), cwd=PYFUN, stdout=lf, stderr=subprocess.STDOUT)
    if not os.path.exists(csv) or os.path.getmtime(csv) < t0 - 1:
        raise RuntimeError(f"JAX baselines failed for {ds}/{variant} (see {log}):\n" + open(log).read()[-1500:])
    hyp = {}
    hjson = os.path.splitext(csv)[0] + "_hypers.json"
    if os.path.exists(hjson):
        hyp = json.load(open(hjson)).get("hypers", {})
    return csv, hyp


def run_extra_kernels(ds, variant, tmp, a):
    """Fold the two nonparametric-kernel Hawkes benchmarks for one cell by invoking their runners as
    subprocesses (each folds its row into a.out and recomputes DM vs GP-DHP + Holm): the discrete
    GP-Hawkes (squared-GP kernel, MAP; fast) and, unless --no-browning, the random-histogram DHP
    (Browning, reversible-jump MCMC; slow on the long daily series)."""
    jobs = [("GP-Hawkes (discrete)", "run_gphawkes.py", [])]
    if not a.no_browning:
        jobs.append(("Random-histogram DHP-NB", "run_browning.py", []))   # full Browning RJMCMC budget
    for name, script, extra in jobs:
        log = os.path.join(tmp, f"{ds}_{variant}_extra_{script}.log")
        cmd = [sys.executable, os.path.join(HERE, script), "--out", a.out,
               "--cells", f"{ds}/{variant}", "--seed", str(a.seed)] + extra
        with open(log, "w") as lf:
            r = subprocess.run(cmd, env=_env({"JAX_PLATFORMS": "cpu"}), cwd=HERE, stdout=lf, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            print(f"[{ds}/{variant}]   WARN extra kernel '{name}' failed (see {log})", flush=True)
        else:
            print(f"[{ds}/{variant}]   folded {name}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ORDER))
    ap.add_argument("--out", default=os.path.join(HERE, "results_hawkes", "nonneural_models.npz"),
                    help="the single accumulating npz (written after every model fit)")
    ap.add_argument("--tmp", default=os.path.join(HERE, "results_hawkes", "_nonneural_tmp"))
    ap.add_argument("--n-starts", type=int, default=20, help="multistart random hyper inits (FV bilevel)")
    ap.add_argument("--maxiter", type=int, default=300, help="max L-BFGS-MT steps per start")
    ap.add_argument("--n-devices", type=int, default=10, help="CPU cores for the multistart thread pool")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-baselines", action="store_true",
                    help="fit ONLY GP-DHP (MAP) per cell and re-fold its row; leave the existing "
                         "baseline logscores in --out untouched (their DM vs GP-DHP auto-recomputes on re-fold)")
    ap.add_argument("--extra-kernels", action=argparse.BooleanOptionalAction, default=True,
                    help="also fit+fold the random-histogram DHP (Browning, RJMCMC) and the discrete "
                         "GP-Hawkes (squared-GP kernel) per cell (default: ON; --no-extra-kernels to skip both)")
    ap.add_argument("--no-browning", action="store_true",
                    help="within --extra-kernels, skip only the (slow) Browning RJMCMC and keep GP-Hawkes")
    ap.add_argument("--keep-tmp", action="store_true", help="keep per-cell aux files/logs")
    a = ap.parse_args()
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    os.makedirs(a.tmp, exist_ok=True)
    print(f"[nonneural] serial per model | multistart {a.n_starts}x{a.maxiter}, "
          f"n-devices {a.n_devices} | -> {a.out}", flush=True)
    t_all = time.time()
    for ds in datasets:
        for v in variants_for(ds):
            # ---- GP-DHP first (MAP is the DM anchor; supplies observed) ----
            print(f"[{ds}/{v}] GP-DHP (MAP) ...", flush=True)
            gp = np.load(run_gpdhp(ds, v, a.tmp, a), allow_pickle=True)
            observed = np.asarray(gp["observed"], float)
            gmeta = dict(val_pll=float(gp["val_pll"]), sel_hypers=str(gp["sel_hypers"]),
                         kappa=float(gp["sel_kappa"]), link_scale=float(gp["sel_link"]), Kf=int(gp["sel_Kf"]),
                         col_sigmas=str(gp["sel_col_sigmas"]) if "sel_col_sigmas" in gp.files else "[]",
                         group_sigmas=str(gp["sel_group_sigmas"]) if "sel_group_sigmas" in gp.files else "[]")
            BS.fold_model(a.out, ds, v, GP_REF, np.asarray(gp["map_logscores"], float), observed, gmeta)
            print(f"[{ds}/{v}]   folded {GP_REF} {float(gp['pll_map']):.1f}", flush=True)
            # ---- JAX multistart baselines (serial internally), fold each of the 7 one at a time ----
            if a.skip_baselines:
                print(f"[{ds}/{v}] baselines SKIPPED (--skip-baselines): existing rows kept, "
                      f"DM vs GP-DHP recomputed on the GP-DHP re-fold", flush=True)
            else:
                print(f"[{ds}/{v}] JAX multistart baselines ...", flush=True)
                import pandas as pd
                csv, hyp = run_baselines(ds, v, a.tmp, a)
                bdf = pd.read_csv(csv)
                for mdl in BASELINES:
                    rows = bdf[bdf.model == mdl]
                    ls = rows["nb_log_score"].to_numpy(float)
                    if len(ls) != len(observed) or not np.allclose(rows["observed"].to_numpy(float), observed):
                        print(f"[{ds}/{v}]   WARN baseline '{mdl}' skipped (n/observed mismatch vs GP-DHP)", flush=True)
                        continue
                    h = hyp.get(mdl, {})
                    meta = dict(val_pll=float(h["val_pll"])) if "val_pll" in h else {}
                    if h:
                        meta["sel_hypers"] = json.dumps({k: v2 for k, v2 in h.items()})
                    rec = BS.fold_model(a.out, ds, v, mdl, ls, observed, meta)
                    print(f"[{ds}/{v}]   folded {mdl:24} pLL {rec['pll']:.1f}", flush=True)
                if a.extra_kernels:
                    print(f"[{ds}/{v}] extra nonparametric-kernel Hawkes benchmarks ...", flush=True)
                    run_extra_kernels(ds, v, a.tmp, a)
            if not a.keep_tmp:
                for f in glob.glob(os.path.join(a.tmp, f"{ds}_{v}_*")) + glob.glob(os.path.join(a.tmp, f"*_{ds}_{v}.log")):
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        print(f"=== {ds} done ({time.time()-t_all:.0f}s elapsed) ===\n", flush=True)
    if not a.keep_tmp:
        try:
            os.rmdir(a.tmp)
        except OSError:
            pass
    print(f"=== ALL DONE ({time.time()-t_all:.0f}s) -> {a.out} ===", flush=True)
    BS.print_summary(a.out)


if __name__ == "__main__":
    main()
