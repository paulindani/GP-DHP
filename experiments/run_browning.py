"""Fit the random-histogram DHP-NB (Browning, Rousseau & Mengersen 2022) by reversible-jump MCMC
on every (dataset, variant) cell and FOLD its held-out log-scores into the existing non-neural
benchmark npz (results_hawkes/nonneural_models.npz) as one additional baseline row, recomputing the
Diebold--Mariano test and Holm correction against GP-DHP (MAP) for each cell.

The excitation kernel is Browning's random histogram (RJMCMC over the bins, Green 1995); the baseline
is matched to the fixed-bin Histogram DHP-NB (intercept + one annual harmonic, plus a weekly harmonic
for daily series, plus the external-covariate block for cov cells) so the two are a clean
fixed-vs-random / MAP-vs-Bayes pair.  Cells are folded SERIALLY (each fold atomically rewrites the npz).

  python run_browning.py                         # all 7 cells
  python run_browning.py --cells nyc/cov,gva/cov # subset
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
from library import datasets          # load_split (y_dev,y_test,period,daily,cal,dates)
from library import covariates as CV
from library import browning_rjmcmc as BR
from library import bench_shared as BS

MODEL = "Random-histogram DHP-NB"
CELLS = [("dengue", "nocov"), ("crypto", "nocov"), ("rand", "nocov"),
         ("nyc", "nocov"), ("nyc", "cov"), ("gva", "nocov"), ("gva", "cov")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results_hawkes", "nonneural_models.npz"))
    ap.add_argument("--cells", default="", help="comma list ds/variant to restrict to (default all 7)")
    ap.add_argument("--n-chains", type=int, default=4)
    ap.add_argument("--n-burn", type=int, default=8000)
    ap.add_argument("--n-samp", type=int, default=40000)
    ap.add_argument("--thin", type=int, default=20)
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
              f"{'(+cov ' + str(Zfull.shape[1]) + ' cols)' if Zfull is not None else ''} ===", flush=True)
        ls, info = BR.fit_score_browning(y_dev, y_test, period, daily=daily, baseline="matched",
                                         cal=cal, dates=dates, Zfull=Zfull,
                                         n_chains=a.n_chains, n_burn=a.n_burn, n_samp=a.n_samp,
                                         thin=a.thin, seed=a.seed, verbose=True)
        meta = dict(baseline="matched", K_bins_mean=round(info["K_mean"], 2),
                    K_bins_median=info["K_med"], birth_accept=round(info["rates"]["birth"], 4),
                    n_draws=info["n_draws"], sampler="RJMCMC")
        rec = BS.fold_model(a.out, ds, variant, MODEL, ls, observed=y_test, meta=meta)
        dm_t, dm_ph = rec.get("dm_vs_gpdhp_t"), rec.get("dm_vs_gpdhp_p_holm")
        print(f"  -> pLL={float(ls.sum()):.2f}  K~{info['K_mean']:.2f}{info['K_range']}  "
              f"DM vs GP-DHP t={dm_t:+.2f} p_holm={dm_ph:.3g}  [folded]\n", flush=True)

    print("=" * 70)
    BS.print_summary(a.out)
    print("done")


if __name__ == "__main__":
    main()
