"""Regenerate the real-data paper figures from the paper's CONSTRAINED GP-DHP fits.

Reads the exact stability-constrained fits (selected hyperparameters, covariate column scales, and
the fitted theta*) from the `{ds}/{variant}/constrained_fit/*` keys of
results_hawkes/nonneural_models.npz, folded in by supp_tables.py -- the same fits behind the
paper's predictive, calibration, and excitation tables -- rebuilds each design,
projects the fitted kernel / components, and writes:
  figure_kernel_small_multiples.pdf        - NB kernel / GP correction / total f(d) over lags,
                                             7 panels (the 5 series; nyc & gva shown with AND without cov.)
  figure_baseline_rates.pdf                - fitted baseline rate softplus(b(t)) over time, 7 panels
                                             (same 7 GP-DHP fits), train/test split marked
  figure_nyc_component_decomposition.pdf   - NYC held-out counts + latent component decomposition
  figureS1_predictive_intervals_final.pdf  - observed + pred mean + 80%/95% NB bands, 5 panels
Each reconstructed fit is verified in place: the recomputed held-out log-likelihood and positive
excitation mass R_+ must match the stored values (so the figures provably match the tables), and
every plotted kernel satisfies the paper's stability constraint R_+ < 1.
Run supp_tables.py first (it folds the constrained fits into the npz), then from
papercode/experiments:  python regen_paper_figures.py [outdir]   (default outdir = ../../paper).
"""
import os, sys, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))                         # papercode/experiments
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))             # papercode
os.environ["HAWKES_ROOT"] = HAWKES
sys.path.insert(0, os.path.dirname(HERE))
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
from library import datasets
from library._gpdhp_common import (build_design, _baseline_cols, lag_matrix_future,
    softplus_link, nb2_logpmf_vec)
from library import covariates as CV

OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(HAWKES), "paper")
os.makedirs(OUT, exist_ok=True)
NPZ = os.path.join(HERE, "results_hawkes/nonneural_models.npz")
if not os.path.exists(NPZ):
    sys.exit(f"{NPZ} not found: run run_nonneural.py and then supp_tables.py first.")
CF = np.load(NPZ, allow_pickle=True)
if "dengue/nocov/constrained_fit/theta" not in CF.files:
    sys.exit(f"no '<ds>/<variant>/constrained_fit/*' keys in {NPZ}: run supp_tables.py first "
             "(it computes the paper's constrained fits and folds them into this npz).")

# All seven GP-DHP fits (nyc & gva carry both covariate variants); kernel + baseline figures show all 7.
PANELS7 = [("dengue", "nocov", "Singapore dengue"),
           ("crypto", "nocov", "German cryptosporidiosis"),
           ("nyc", "cov", "NYC shootings (with cov.)"),
           ("nyc", "nocov", "NYC shootings (no cov.)"),
           ("gva", "cov", "GVA shootings (with cov.)"),
           ("gva", "nocov", "GVA shootings (no cov.)"),
           ("rand", "nocov", "Worldwide terrorism")]
# The five headline fits (paper variant: cov for nyc/gva) for the per-series predictive-interval figure.
PANELS5 = [("dengue", "nocov", "Singapore dengue"), ("crypto", "nocov", "German cryptosporidiosis"),
           ("nyc", "cov", "NYC shootings"), ("gva", "cov", "GVA shootings"),
           ("rand", "nocov", "Worldwide terrorism")]
DATADIR = {"dengue": "dengue_singapore", "crypto": "crypto_survstat_de",
           "rand": "RAND_terrorism", "nyc": "nyc_shootings", "gva": "gva_allshootings"}


def fit(ds, variant):
    """Rebuild one stored constrained fit and project its components (no re-optimization)."""
    key = f"{ds}/{variant}/constrained_fit"
    h = json.loads(str(CF[f"{key}/h"]))
    kappa = float(CF[f"{key}/kappa"]); lk = float(CF[f"{key}/link_scale"])
    th = np.asarray(CF[f"{key}/theta"], float)
    y_dev, y_test, period, daily = datasets.load_split(ds); D = int(h["D_max"]); T = len(y_dev)
    _, _, _, _, _, dates_all = datasets.load_split_cal(ds)          # dev-then-test dates, chronological
    tt = jnp.arange(T + 1, T + len(y_test) + 1, dtype=jnp.float64)
    des = build_design(y_dev, h); L = np.asarray(des["L"]); m_NB = np.asarray(des["m_NB"]); q_b = des["q_b"]
    B_sc = np.asarray(_baseline_cols(tt, h["period"], h)); X_sc = np.asarray(lag_matrix_future(y_dev, y_test, D))
    # full-span baseline design (dev then test), for the excitation-free baseline-rate figure
    t_full = jnp.arange(1, T + len(y_test) + 1, dtype=jnp.float64)
    B_full = np.asarray(_baseline_cols(t_full, h["period"], h))
    if variant == "cov":
        cs = np.asarray(CF[f"{key}/col_sigmas"], float)
        cov_dev, cov_test, _ = CV.build_covariates(ds, root=HAWKES)
        ncol = cov_test.shape[1]
        tb = th[:q_b]; tc = th[q_b:q_b + ncol]; tg = th[q_b + ncol:]
        base = B_sc @ tb + (np.asarray(cov_test) * cs) @ tc
        base_lat_full = B_full @ tb + (np.concatenate([np.asarray(cov_dev), np.asarray(cov_test)], 0) * cs) @ tc
    else:
        tb = th[:q_b]; tg = th[q_b:]
        base = B_sc @ tb
        base_lat_full = B_full @ tb
    gp = (L @ tg) if L.shape[1] else np.zeros_like(m_NB)
    fhat = m_NB + gp
    nbk = X_sc @ m_NB; gpc = X_sc @ gp; exc = X_sc @ fhat
    mu = np.asarray(softplus_link(jnp.asarray(base + exc), lk, 1e-6))
    base_rate = np.asarray(softplus_link(jnp.asarray(base_lat_full), lk, 1e-6))   # excitation-free rate
    # the reconstruction must reproduce the stored fit exactly, so the figures match the tables
    Rp = float(np.maximum(fhat, 0.0).sum())
    assert Rp < 1.0, f"{ds}/{variant}: plotted kernel violates the stability constraint (R_+={Rp})"
    assert abs(Rp - float(CF[f"{key}/R_plus"])) < 1e-8, f"{ds}/{variant}: R_+ mismatch vs stored fit"
    ls = np.asarray(nb2_logpmf_vec(jnp.asarray(np.asarray(y_test, float)), jnp.asarray(mu), kappa), float)
    d_pll = abs(float(ls.sum()) - float(CF[f"{key}/pll"]))
    assert d_pll < 1e-6, f"{ds}/{variant}: held-out pLL mismatch vs stored fit ({d_pll:.2e})"
    return dict(m_NB=m_NB, gp=gp, fhat=fhat, D=D, base=base, nbk=nbk, gpc=gpc, exc=exc,
                mu=mu, y=np.asarray(y_test, float), kappa=kappa, daily=daily,
                base_rate=base_rate, dates=pd.to_datetime(dates_all), T=T)


F = {(ds, v): fit(ds, v) for ds, v, _ in PANELS7}      # every fit needed by any figure


def _axes_322(figsize):
    """7 panels in a 3/2/2 layout on a shared 6-column base: row 1 = 3 panels (2 cols each),
    rows 2 & 3 = 2 wider panels (3 cols each). Returns (fig, [ax0..ax6]) in row-major order."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 6, left=0.05, right=0.995, top=0.95, bottom=0.075, hspace=0.42, wspace=0.55)
    return fig, [fig.add_subplot(gs[0, 0:2]), fig.add_subplot(gs[0, 2:4]), fig.add_subplot(gs[0, 4:6]),
                 fig.add_subplot(gs[1, 0:3]), fig.add_subplot(gs[1, 3:6]),
                 fig.add_subplot(gs[2, 0:3]), fig.add_subplot(gs[2, 3:6])]


def _axes_baseline(figsize):
    """Baseline-rate layout: row 1 = 2 half-width panels (the weekly series), rows 2-6 = one
    full-width panel each (the five daily series). Returns (fig, [ax0..ax6]) in that order."""
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 2, left=0.065, right=0.996, top=0.985, bottom=0.03, hspace=0.35, wspace=0.16)
    ax = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    ax += [fig.add_subplot(gs[r, 0:2]) for r in range(1, 6)]
    return fig, ax

# Figure: kernel small multiples (7 panels, 3/2/2 rows; nyc & gva with and without covariates)
fig, axes = _axes_322((15, 9))
for ax, (ds, v, title) in zip(axes, PANELS7):
    d = F[(ds, v)]; Lp = min(60, d["D"]); x = np.arange(1, Lp + 1)
    ax.axhline(0, color="0.7", lw=0.8)
    # total first (bottom layer), then the NB kernel and GP correction as dotted lines on top so
    # they remain visible where they coincide with the total
    ax.plot(x, d["fhat"][:Lp], color="black", lw=1.6, label=r"Total $\hat f(d)$", zorder=1)
    ax.plot(x, d["m_NB"][:Lp], color="C0", ls=":", lw=2.0, label="NB kernel", zorder=2)
    ax.plot(x, d["gp"][:Lp], color="C1", ls=":", lw=2.2, label="GP correction", zorder=3)
    ax.set_title(title, fontsize=10); ax.set_xlabel("lag $d$"); ax.set_ylabel("response")
axes[0].legend(frameon=False, fontsize=9)
fig.align_ylabels(axes)           # lock every "response" caption to a common x per column
fig.savefig(os.path.join(OUT, "figure_kernel_small_multiples.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Figure: fitted baseline rates softplus(b(t)) over time (row 1: 2 weekly series; rows 2-6: 5 full-width daily series)
fig, axes = _axes_baseline((13, 10.5))
for ax, (ds, v, title) in zip(axes, PANELS7):
    d = F[(ds, v)]; dates = d["dates"]; r = d["base_rate"]; T = d["T"]
    ax.plot(dates, r, color="navy", lw=0.4)
    ax.axvline(dates[T], color="0.55", ls=":", lw=1.0)                 # train | test split
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("baseline rate (counts/" + ("day" if d["daily"] else "week") + ")", fontsize=9)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
# lock all "baseline rate (...)" captions to a common x so the five full-width panels align exactly
fig.align_ylabels(axes)
fig.savefig(os.path.join(OUT, "figure_baseline_rates.pdf"), bbox_inches="tight", pad_inches=0.02); plt.close(fig)

# Figure: NYC component decomposition (headline cov fit)
df = pd.read_csv(f"{HAWKES}/data/{DATADIR['nyc']}/dataset_counts.csv")
sp = (df["final_split"] if "final_split" in df.columns else df["split"]).astype(str).str.lower()
dates = pd.to_datetime(df[sp.eq("test")]["date"]).to_numpy(); n = F[("nyc", "cov")]
fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
a1.plot(dates, n["y"], color="0.55", lw=0.7, label="observed")
a1.plot(dates, n["mu"], color="C3", lw=1.1, label="one-step predictive mean")
a1.set_ylabel("daily count"); a1.set_title("NYC shootings: held-out one-step forecast")
a1.legend(frameon=False, ncol=2, fontsize=9, loc="upper right")
a2.axhline(0, color="0.7", lw=0.8)
a2.plot(dates, n["base"], color="C2", lw=0.7, label="baseline (level+season+week+cov.)")
a2.plot(dates, n["nbk"], color="C0", lw=0.9, label="NB kernel contribution")
a2.plot(dates, n["gpc"], color="C1", ls="--", lw=0.9, label="GP correction")
a2.plot(dates, n["exc"], color="black", lw=1.0, label="total excitation")
a2.set_ylabel("latent contribution"); a2.set_xlabel("date")
_lo, _hi = a2.get_ylim(); a2.set_ylim(_lo, _hi + 0.45 * (_hi - _lo))   # headroom so legend clears the curves
a2.legend(frameon=False, ncol=2, fontsize=8, loc="upper right")
a2.xaxis.set_major_locator(mdates.MonthLocator(interval=4)); a2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure_nyc_component_decomposition.pdf")); plt.close(fig)

# Figure: predictive intervals (5 headline panels)
fig, axes = plt.subplots(2, 3, figsize=(13, 6.5)); axes = axes.ravel()
for ax, (ds, v, title) in zip(axes, PANELS5):
    d = F[(ds, v)]; mu = d["mu"]; kappa = d["kappa"]; y = d["y"]; x = np.arange(len(y))
    nb = stats.nbinom(kappa, kappa / (kappa + mu))
    ax.fill_between(x, nb.ppf(0.025), nb.ppf(0.975), color="C0", alpha=0.18, lw=0)
    ax.fill_between(x, nb.ppf(0.10), nb.ppf(0.90), color="C0", alpha=0.30, lw=0)
    ax.plot(x, y, ".", color="0.2", ms=2.0, label="observed")
    ax.plot(x, mu, color="black", lw=0.8, label="pred. mean")
    ax.set_title(title); ax.set_xlabel("test index"); ax.set_ylabel("count")
axes[0].legend(frameon=False, fontsize=9, markerscale=2); axes[5].axis("off")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figureS1_predictive_intervals_final.pdf")); plt.close(fig)
print("wrote 4 figures to", OUT)
