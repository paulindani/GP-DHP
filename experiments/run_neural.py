"""Neural benchmark orchestrator: the four JAX neural NB models (MLP-NB, GRU-NB, LSTM-NB, DeepAR-NB)
over ALL datasets and BOTH covariate variants where covariates exist (nyc, gva).  Covariates were OFF
in the previous neural run; this runs each covariate dataset TWICE (nocov + cov).

Serial per (dataset, variant): runs the 144-combo grid (neural_grid_jax.py, 40% forward-validation)
and folds the 4 per-run npz into ONE separate npz -- same flat-key + manifest format as the non-neural
orchestrator, with DM vs GP-DHP(MAP) + Holm.  The GP-DHP(MAP) reference per-obs log-scores are read
from the NON-NEURAL npz (--gpdhp-npz), so run run_nonneural.py first for the DM tests to populate.
A final idempotent recompute_dm() pass guarantees DM+Holm land on every cell; --recompute-dm-only adds
them to an existing npz (e.g. a GPU run that stored only pLL) without re-running the grid.

  python run_neural.py                                       # all datasets/variants, full grid
  python run_neural.py --datasets dengue,nyc --limit-combos 2 --neural-max-epochs 10   # smoke
  python run_neural.py --recompute-dm-only                   # add DM to an existing npz, no retraining
"""
import argparse
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
NEURAL = ["MLP-NB", "GRU-NB", "LSTM-NB", "DeepAR-NB"]
GP_REF = BS.GP_REF                                             # "GP-DHP (MAP)"


def _env(extra=None):
    e = dict(os.environ, HAWKES_ROOT=HAWKES, VECLIB_MAXIMUM_THREADS="2", OMP_NUM_THREADS="2")
    if extra:
        e.update(extra)
    return e


def variants_for(ds):
    return ["nocov", "cov"] if CV.has_covariates(ds) else ["nocov"]


def gpdhp_ref(gpdhp_npz, ds, variant):
    """(observed, GP-DHP(MAP) logscores) for (ds, variant) from the non-neural npz, or (None, None)."""
    if not gpdhp_npz or not os.path.exists(gpdhp_npz):
        return None, None
    pre = f"{ds}/{variant}/"
    with np.load(gpdhp_npz, allow_pickle=True) as z:
        ok, mk = pre + "observed", pre + GP_REF + "/logscores"
        if ok in z.files and mk in z.files:
            return np.asarray(z[ok], float), np.asarray(z[mk], float)
    return None, None


def recompute_dm(out, gpdhp_npz, datasets, verbose=True):
    """Fold DM-vs-GP-DHP(MAP) + Holm into an EXISTING neural npz WITHOUT re-running the grid.

    Reuses the per-obs logscores already stored in `out`; brings in the GP-DHP(MAP) reference
    from `gpdhp_npz` if it is not already present, then re-folds every non-reference model so
    bench_shared.fold_model recomputes DM + Holm and rewrites the npz.  Used both standalone
    (--recompute-dm-only, e.g. to add DM to a GPU run that stored only pLL) and as an idempotent
    finalisation pass after a fresh grid, so DM is ALWAYS present."""
    if not os.path.exists(out):
        raise SystemExit(f"[neural] recompute_dm: {out} not found")
    with np.load(out, allow_pickle=True) as z:                     # snapshot logscores (immutable) + manifest
        man = json.loads(str(z["manifest"]))
        store = {k: z[k] for k in z.files if k != "manifest"}
    for ds in datasets:
        for v in variants_for(ds):
            pre = f"{ds}/{v}/"
            if pre + "observed" not in store:
                continue
            observed = np.asarray(store[pre + "observed"], float)
            if pre + GP_REF + "/logscores" not in store:           # pull in the reference if absent
                obs2, gpref = gpdhp_ref(gpdhp_npz, ds, v)
                if gpref is None:
                    print(f"[{ds}/{v}] DM skipped (no GP-DHP(MAP) reference in {out} or {gpdhp_npz})", flush=True)
                    continue
                BS.fold_model(out, ds, v, GP_REF, gpref, obs2 if obs2 is not None else observed)
            for r in [rr for rr in man if rr["dataset"] == ds and rr["variant"] == v and rr["model"] != GP_REF]:
                k = pre + r["model"] + "/logscores"
                if k not in store:
                    continue
                meta = {kk: r[kk] for kk in ("val_pll", "sel_hypers") if kk in r} or None
                rec = BS.fold_model(out, ds, v, r["model"], np.asarray(store[k], float), observed, meta)
                if verbose:
                    t = rec.get("dm_vs_gpdhp_t"); ph = rec.get("dm_vs_gpdhp_p_holm")
                    tag = "DM n/a" if t is None else f"t={t:+.2f} p_holm={ph:.3f}"
                    print(f"[{ds}/{v}]   DM {r['model']:10} pLL {rec['pll']:.1f} | {tag}", flush=True)


def run_grid(ds, variant, tmp, a):
    """Neural grid (all 4 models) for one cell (subprocess). Per-model npz land in tmp/neural."""
    outdir = os.path.join(tmp, "neural")
    cmd = [sys.executable, os.path.join(PYFUN, "neural_grid_jax.py"),
           "--datasets", ds, "--models", ",".join(a.neural_models),
           "--max-epochs", str(a.neural_max_epochs), "--seed", str(a.seed), "--outdir", outdir]
    if a.limit_combos:
        cmd += ["--limit-combos", str(a.limit_combos)]
    if variant == "cov":
        cmd.append("--cov")
    log = os.path.join(tmp, f"neural_{ds}_{variant}.log")
    with open(log, "w") as lf:
        r = subprocess.run(cmd, env=_env(), cwd=PYFUN, stdout=lf, stderr=subprocess.STDOUT)
    if r.returncode != 0:
        raise RuntimeError(f"neural grid failed for {ds}/{variant} (see {log}):\n" + open(log).read()[-1500:])
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(ORDER))
    ap.add_argument("--out", default=os.path.join(HERE, "results_hawkes", "neural_models.npz"),
                    help="the single accumulating neural npz (separate from the non-neural one)")
    ap.add_argument("--gpdhp-npz", default=os.path.join(HERE, "results_hawkes", "nonneural_models.npz"),
                    help="non-neural npz supplying the GP-DHP(MAP) DM reference per (ds,variant)")
    ap.add_argument("--tmp", default=os.path.join(HERE, "results_hawkes", "_neural_tmp"))
    ap.add_argument("--neural-models", default=",".join(NEURAL))
    ap.add_argument("--neural-max-epochs", type=int, default=300)
    ap.add_argument("--limit-combos", type=int, default=None, help="SMOKE ONLY: truncate the grid")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--recompute-dm-only", action="store_true",
                    help="do NOT run the grid; just (re)compute DM-vs-GP-DHP(MAP)+Holm on the per-obs "
                         "logscores already stored in --out (e.g. a GPU run that saved only pLL) and rewrite it")
    ap.add_argument("--keep-tmp", action="store_true")
    a = ap.parse_args()
    a.neural_models = [m.strip() for m in a.neural_models.split(",") if m.strip()]
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    if a.recompute_dm_only:
        print(f"[neural] recompute-DM-only on {a.out} (reference <- {a.gpdhp_npz})", flush=True)
        recompute_dm(a.out, a.gpdhp_npz, datasets)
        BS.print_summary(a.out)
        return
    os.makedirs(os.path.join(a.tmp, "neural"), exist_ok=True)
    if not os.path.exists(a.gpdhp_npz):
        print(f"[neural] NOTE: {a.gpdhp_npz} not found -> DM-vs-GP-DHP omitted (run run_nonneural.py first)", flush=True)
    print(f"[neural] serial per (dataset,variant), both cov variants where available | -> {a.out}", flush=True)
    t_all = time.time()
    for ds in datasets:
        for v in variants_for(ds):
            print(f"[{ds}/{v}] neural grid ({len(a.neural_models)} models) ...", flush=True)
            outdir = run_grid(ds, v, a.tmp, a)
            observed, gpref = gpdhp_ref(a.gpdhp_npz, ds, v)
            if gpref is not None:                              # fold the GP-DHP(MAP) reference first (no DM on itself)
                BS.fold_model(a.out, ds, v, GP_REF, gpref, observed)
            for m in a.neural_models:
                f = os.path.join(outdir, f"{ds}_{v}_{m}.npz")
                if not os.path.exists(f):
                    print(f"[{ds}/{v}]   WARN neural '{m}' npz missing; skipped", flush=True)
                    continue
                z = np.load(f, allow_pickle=True)
                ls = np.asarray(z["logscores"], float)
                obs = observed if observed is not None else np.asarray(z["observed"], float)
                meta = dict(val_pll=float(z["val_pll"]), sel_hypers=str(z["sel_hypers"]))
                rec = BS.fold_model(a.out, ds, v, m, ls, obs, meta)
                dm = rec.get("dm_vs_gpdhp_t")
                print(f"[{ds}/{v}]   folded {m:10} pLL {rec['pll']:.1f}"
                      + ("" if dm is None else f" | DM vs GP-DHP t={dm:+.2f}"), flush=True)
            if not a.keep_tmp:
                import glob
                for f in glob.glob(os.path.join(outdir, f"{ds}_{v}_*")) + [os.path.join(a.tmp, f"neural_{ds}_{v}.log")]:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        print(f"=== {ds} done ({time.time()-t_all:.0f}s elapsed) ===\n", flush=True)
    # finalise: guarantee DM-vs-GP-DHP(MAP)+Holm is present on every cell (idempotent safety net)
    recompute_dm(a.out, a.gpdhp_npz, datasets, verbose=False)
    print(f"=== ALL DONE ({time.time()-t_all:.0f}s) -> {a.out} ===", flush=True)
    BS.print_summary(a.out)


if __name__ == "__main__":
    main()
