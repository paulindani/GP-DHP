"""Fit the discrete-time Gaussian-process-modulated Hawkes benchmark (squared-GP triggering kernel;
the straightforward discrete analogue of Zhou et al. 2020 and Zhang, Walder & Rizoiu 2020) on every
(dataset, variant) cell and FOLD its held-out log-scores into the non-neural benchmark npz
(results_hawkes/nonneural_models.npz) as one additional baseline row, recomputing the Diebold--Mariano
test and Holm correction against GP-DHP (MAP) per cell.

Same baseline/covariate design, rectified NB observation layer, and 40% forward-validation protocol as
the fixed-bin Histogram DHP-NB, so the two differ only in the excitation kernel (fixed bins vs squared
GP). Cells are folded SERIALLY (each fold atomically rewrites the npz).

  python run_gphawkes.py                          # all 7 cells
  python run_gphawkes.py --cells nyc/cov,gva/cov  # subset
"""
import argparse
import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.normpath(os.path.join(HERE, "..")))
os.environ.setdefault("HAWKES_ROOT", HAWKES)
sys.path.insert(0, os.path.dirname(HERE))
from library import datasets
from library import covariates as CV
from library import gphawkes_discrete as GH
from library import bench_shared as BS

MODEL = "GP-Hawkes (discrete)"
CELLS = [("dengue", "nocov"), ("crypto", "nocov"), ("rand", "nocov"),
         ("nyc", "nocov"), ("nyc", "cov"), ("gva", "nocov"), ("gva", "cov")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results_hawkes", "nonneural_models.npz"))
    ap.add_argument("--cells", default="")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cells = CELLS
    if a.cells:
        want = {tuple(c.split("/")) for c in a.cells.split(",")}
        cells = [c for c in CELLS if c in want]
    if not os.path.exists(a.out):
        sys.exit(f"npz not found: {a.out} (run run_nonneural.py first)")
    print(f"folding '{MODEL}' into {a.out}\n  cells: {cells}\n", flush=True)

    for ds, variant in cells:
        y_dev, y_test, period, daily, cal, dates = datasets.load_split_cal(ds)
        Zfull = None
        if variant == "cov":
            if not CV.has_covariates(ds):
                print(f"[{ds}/{variant}] no covariates -> skip", flush=True); continue
            Zdev, Ztest, _ = CV.build_covariates(ds, root=HAWKES)
            Zfull = np.vstack([Zdev, Ztest])
        print(f"=== {ds}/{variant}: T_fit={len(y_dev)} n_test={len(y_test)} daily={daily} "
              f"{'(+cov ' + str(Zfull.shape[1]) + ')' if Zfull is not None else ''} ===", flush=True)
        ls, info = GH.fit_score_gphawkes(y_dev, y_test, period, daily=daily, cal=cal, dates=dates,
                                         Zfull=Zfull, seed=a.seed, verbose=True)
        meta = dict(kernel="squared-GP", sigma=info["sigma"], ell=info["ell"], s=info["s"],
                    kappa=info["kappa"], peak_lag=info["peak_lag"], kernel_mass=info["kernel_mass"],
                    val_pll=info["val_pll"],
                    selection="forward-validation bilevel (20-start L-BFGS-MT, analytic hypergradient)")
        rec = BS.fold_model(a.out, ds, variant, MODEL, ls, observed=y_test, meta=meta)
        print(f"  -> pLL={float(ls.sum()):.2f}  peak_lag={info['peak_lag']} mass={info['kernel_mass']}  "
              f"DM vs GP-DHP t={rec.get('dm_vs_gpdhp_t'):+.2f} p_holm={rec.get('dm_vs_gpdhp_p_holm'):.3g}  [folded]\n",
              flush=True)

    print("=" * 70)
    BS.print_summary(a.out)
    print("done")


if __name__ == "__main__":
    main()
