"""Non-randomized PIT check (Czado, Gneiting & Held 2009) for the constrained GP-DHP fits.

The non-randomized PIT for count data is NOT a collection of scalar PIT values: for each held-out
observation y_t with one-step predictive CDF P_t, it contributes the piecewise-linear conditional CDF

    F_t(u) = 0                                    for u <= P_t(y_t - 1),
             (u - P_t(y_t-1)) / (P_t(y_t) - P_t(y_t-1))   in between,
             1                                    for u >= P_t(y_t),

and calibration is assessed through the AGGREGATED CDF  Fbar(u) = (1/n) sum_t F_t(u), which is the
uniform CDF (Fbar(u) = u) under probabilistic calibration.  This script computes, per cell,
  * the 10-bin non-randomized PIT histogram (bin mass 0.10 each under uniformity), and
  * the sup-distance  max_u |Fbar(u) - u|  from the uniform CDF,
for exactly the paper's stability-constrained fits: it reruns the SAME deterministic constrained
selection + ALM refit as supp_tables.py (same budgets, delta=1e-4, 20 starts, seed 0), so the
predictive distributions are those behind the calibration table (tab:realdata-calibration).
The seven cells match that table: the five headline fits plus the no-covariate variants of the two
daily shooting series.  It prints the histogram and sup-distance per cell and writes the 7-panel
supplementary figure (../paper/figureS3_pit_histograms.pdf) that the calibration paragraph references.

Run:  python pit_check.py     (~ the same runtime as supp_tables.py; selection dominates)
"""
import os, sys, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
import numpy as np, jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import nbinom
jax.config.update("jax_enable_x64", True)
HAWKES = os.environ.get("HAWKES_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HAWKES_ROOT"] = HAWKES
sys.path.insert(0, HAWKES)
from library import datasets
from library import covariates as CV
from library import gpdhp_cfv_grad as CFV
from library import gpdhp_fit
from library.lbfgs_mt import minimize_bfgs_mt
from library._gpdhp_common import softplus_link, build_design, _baseline_cols, lag_matrix_future

DELTA = 1e-4; MULT = 2.0; NST = 20; SEED = 0; ND = 10
MAXIT = int(300 * MULT); INNER = int(500 * MULT); REFIT = int(800 * MULT)
FLOOR = 1e-6
CELLS = [("Singapore dengue", "dengue", False), ("German cryptosporidiosis", "crypto", False),
         ("NYC shootings (with cov.)", "nyc", True), ("NYC shootings (no cov.)", "nyc", False),
         ("GVA shootings (with cov.)", "gva", True), ("GVA shootings (no cov.)", "gva", False),
         ("Worldwide terrorism (RAND)", "rand", False)]


def predictive(label, ds, cov):
    """Constrained selection + ALM refit (identical to supp_tables.py) -> (y_test, lam, kappa)."""
    res = gpdhp_fit.fit_constrained_cell(ds, cov, delta=DELTA, mult=MULT, n_starts=NST,
                                         seed=SEED, n_devices=ND, root=HAWKES)
    pred = gpdhp_fit.constrained_predictive(res)
    return pred["y_test"], pred["lam"], pred["kappa"]


def czado_pit(y, lam, kappa, n_grid=1000, n_bins=10):
    """Aggregated non-randomized PIT CDF Fbar on a grid; return (bin masses, sup|Fbar(u)-u|)."""
    pnb = kappa / (kappa + lam)
    Phi = nbinom.cdf(y, kappa, pnb)                    # P_t(y_t)
    Plo = np.where(y > 0, nbinom.cdf(y - 1, kappa, pnb), 0.0)   # P_t(y_t - 1), 0 for y_t = 0
    u = np.linspace(0.0, 1.0, n_grid + 1)
    # F_t(u) piecewise-linear between Plo and Phi; aggregate over t
    Ft = np.clip((u[None, :] - Plo[:, None]) / np.maximum(Phi - Plo, 1e-300)[:, None], 0.0, 1.0)
    Fbar = Ft.mean(axis=0)
    ks = float(np.max(np.abs(Fbar - u)))
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    Fedge = np.interp(edges, u, Fbar)
    bins = np.diff(Fedge)                              # bin masses; 1/n_bins each under uniformity
    return bins, ks


OUT = os.path.join(os.path.dirname(HAWKES), "paper")           # papercode/../paper


def make_figure(results, path):
    """results: list of (label, bins, ks). Save the per-cell non-randomized PIT histogram."""
    n = len(results)
    ncol = 4 if n > 6 else 3; nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.5 * ncol, 3.0 * nrow)); axes = axes.ravel()
    edges = np.linspace(0.0, 1.0, len(results[0][1]) + 1)
    for ax, (label, bins, ks) in zip(axes, results):
        ax.bar(edges[:-1], 10.0 * bins, width=np.diff(edges), align="edge",
               color="#cfe0f2", edgecolor="#1f5b9e", linewidth=0.8)
        ax.axhline(1.0, color="0.35", ls="--", lw=1.0)             # uniform reference
        ax.set_title(label, fontsize=10)
        ax.set_xlim(0, 1); ax.set_ylim(0, max(1.7, 10.0 * bins.max() * 1.08))
        ax.set_xlabel("PIT value"); ax.set_ylabel("relative frequency")
        ax.text(0.03, 0.94, rf"$\sup|\bar F_{{\rm PIT}}-u|={ks:.3f}$", transform=ax.transAxes,
                fontsize=9, va="top")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


if __name__ == "__main__":
    print(f"{'cell':28} {'sup|Fbar-u|':>12}   10-bin non-randomized PIT histogram (x10, uniform=1.00)")
    results = []
    for label, ds, cov in CELLS:
        t0 = time.time()
        y, lam, kappa = predictive(label, ds, cov)
        bins, ks = czado_pit(y, lam, kappa)
        results.append((label, bins, ks))
        hb = " ".join(f"{10*b:.2f}" for b in bins)
        print(f"{label:28} {ks:12.4f}   [{hb}]   ({time.time()-t0:.0f}s)", flush=True)
    figpath = os.path.join(OUT, "figureS3_pit_histograms.pdf")
    make_figure(results, figpath)
    print(f"wrote {figpath}")
    print("done_marker")
