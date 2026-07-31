"""Regenerate the four supplementary tables from the CONSTRAINED GP-DHP fits.

The main tables now report the stability-constrained fit, so the supplements must come from the same
fits (they currently carry unconstrained-fit numbers).  Refits the 5 "final" cells -- weekly series
nocov, daily shooting series WITH covariates (the paper's final fits) -- under the settled protocol
(ALM, delta=1e-4, 2x budget, 20 starts, seed 0) and emits, per cell:
  tab:supp-selected-hyperparameters : kappa, s, K_fou, sigma_{level,season,week,lin}, K_NB, mean lag,
                                      NB size, beta, sigma_g/ell_g  (+ covariate prior scales)
  tab:supp-excitation-summaries     : R_+ and peak / 50 / 80 / 90% positive-mass accumulation lags
  tab:supp-alt-metrics              : held-out MAE / RMSE of the one-step predictive MEAN
  tab:realdata-calibration          : test n, kappa, 50/80/95% central-interval coverage + mean widths
Predictive mean/intervals use the NB2 law at the fitted one-step lambda(t): mean = lambda,
size = kappa, so p = kappa/(kappa+lambda); intervals are the central quantile intervals of that
per-observation NB.

Also folds the exact constrained fits (selected h, kappa, link scale, covariate column scales, and
the fitted theta*) into results_hawkes/nonneural_models.npz under `{ds}/{variant}/constrained_fit/*`
keys, which regen_paper_figures.py consumes so the real-data figures are drawn from the SAME fits
as these tables.
"""
import os, sys, json, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
import numpy as np, jax, jax.numpy as jnp
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
from library._gpdhp_common import (nb_kernel, warped_rbf_cholesky, softplus_link,
                                             build_design, _baseline_cols, lag_matrix_future)

DELTA = 1e-4; MULT = 2.0; NST = 20; SEED = 0; ND = 10
MAXIT = int(300 * MULT); INNER = int(500 * MULT); REFIT = int(800 * MULT)
FLOOR = 1e-6
OUT = os.path.join(HAWKES, "experiments", "results_hawkes")
# the paper's FINAL fits: weekly series without covariates, daily shooting series WITH covariates
# (label, dataset, covariates, headline).  headline fits get the full supplementary
# characterization (hyperparameters S2, excitation S5, MAE/RMSE S1) and are folded into the
# figure-pipeline npz; the two no-covariate daily variants are calibration-only extra rows
# (tab:realdata-calibration and the PIT figure), giving that table the same 7 cells as the
# predictive tables.
CELLS = [("Singapore dengue", "dengue", False, True),
         ("German cryptosporidiosis", "crypto", False, True),
         ("NYC shootings", "nyc", True, True),
         ("NYC shootings (no cov.)", "nyc", False, False),
         ("GVA shootings", "gva", True, True),
         ("GVA shootings (no cov.)", "gva", False, False),
         ("Worldwide terrorism (RAND)", "rand", False, True)]


def acc_lag(fpos, frac):
    """smallest lag d with cumulative positive mass >= frac * total positive mass"""
    c = np.cumsum(fpos); tot = c[-1]
    if tot <= 0: return 0
    return int(np.searchsorted(c, frac * tot) + 1)


rows = {}
fits = {}    # exact per-cell constrained fits -> folded into nonneural_models.npz (figure pipeline)
for label, ds, cov, headline in CELLS:
    t0 = time.time()
    res = gpdhp_fit.fit_constrained_cell(ds, cov, delta=DELTA, mult=MULT, n_starts=NST,
                                         seed=SEED, n_devices=ND, root=HAWKES)
    ls, Rp, sel = res["ls"], res["Rp"], res["sel"]
    h, kappa, lk = sel["h"], sel["kappa"], sel["link_scale"]
    pred = gpdhp_fit.constrained_predictive(res)
    lam, f_hat, theta = pred["lam"], pred["f_hat"], pred["theta"]
    y_testn = res["y_test"]; n_test = len(y_testn)

    # persist the exact fit so regen_paper_figures.py draws the figures from these same fits.
    # All seven fits are folded (incl. the no-cov nyc/gva variants), so the kernel and baseline-rate
    # figures can show both covariate variants of the two daily shooting series.
    pre = f"{ds}/{'cov' if cov else 'nocov'}/constrained_fit"
    fits[f"{pre}/theta"] = theta
    fits[f"{pre}/h"] = json.dumps(dict(h), default=float)
    fits[f"{pre}/col_sigmas"] = np.asarray(sel["col_sigmas"], float) if cov else np.zeros(0)
    fits[f"{pre}/kappa"] = np.float64(kappa)
    fits[f"{pre}/link_scale"] = np.float64(lk)
    fits[f"{pre}/R_plus"] = np.float64(np.maximum(f_hat, 0.0).sum())
    fits[f"{pre}/pll"] = np.float64(ls.sum())

    # ---- metrics ---------------------------------------------------------------------------- #
    err = y_testn - lam
    mae, rmse = float(np.mean(np.abs(err))), float(np.sqrt(np.mean(err ** 2)))
    pnb = kappa / (kappa + lam)
    cov_w = {}
    for lev in (50, 80, 95):
        a = (1 - lev / 100) / 2
        lo = nbinom.ppf(a, kappa, pnb); hi = nbinom.ppf(1 - a, kappa, pnb)
        cov_w[lev] = (100.0 * float(np.mean((y_testn >= lo) & (y_testn <= hi))),
                      float(np.mean(hi - lo)))

    fpos = np.maximum(f_hat, 0.0)
    rows[label] = dict(
        ds=ds, cov=cov, headline=headline, R_plus=Rp, n_test=n_test, kappa=kappa, s=lk, Kf=int(sel["K_fourier"]),
        sigma_level=float(h["sigma_level"]), sigma_season=float(h["sigma_fourier"]),
        sigma_week=float(h["sigma_weekly"]), sigma_lin=float(h["sigma_lin"]),
        K_NB=float(h["K_NB"]), mean_lag=float(h["mean_lag"]), nb_size=float(h["size"]),
        beta=float(h["beta"]), sigma_g=float(h["sigma_u"]), ell_g=float(h["ell_u"]),
        cov_scales=[float(s_) for s_ in sel.get("cov_scales", [])],
        peak_lag=int(np.argmax(fpos) + 1), lag50=acc_lag(fpos, .5), lag80=acc_lag(fpos, .8),
        lag90=acc_lag(fpos, .9), mae=mae, rmse=rmse,
        cov50=cov_w[50][0], cov80=cov_w[80][0], cov95=cov_w[95][0],
        w50=cov_w[50][1], w80=cov_w[80][1], w95=cov_w[95][1], pLL=float(np.asarray(ls).sum()))
    print(f"{label:27} R_+={Rp:.6f} pLL={rows[label]['pLL']:9.1f} MAE={mae:7.3f} RMSE={rmse:7.3f} "
          f"cov={cov_w[50][0]:.1f}/{cov_w[80][0]:.1f}/{cov_w[95][0]:.1f} ({time.time()-t0:.0f}s)", flush=True)

json.dump(rows, open(os.path.join(OUT, "supp_tables.json"), "w"), indent=1)
from library import bench_shared as BS
_NNPZ = os.path.join(OUT, "nonneural_models.npz")
_store, _manifest = BS._load(_NNPZ)
_store.update(fits)
BS._save(_NNPZ, _store, _manifest)
print(f"\nfolded the exact constrained fits into {_NNPZ} "
      "('<ds>/<variant>/constrained_fit/*' keys, consumed by regen_paper_figures.py)")
dash = lambda v: "--" if v <= 0 else None

# S2/S5/S1 characterize the five headline fitted models; the calibration table (Table 5) below
# additionally lists the two no-covariate daily variants, matching the 7-cell predictive tables.
headline_rows = {lab: r for lab, r in rows.items() if r["headline"]}

print("\n\n########## tab:supp-selected-hyperparameters ##########")
for lab, r in headline_rows.items():
    sw = "--   " if r["sigma_week"] <= 0 else f"{r['sigma_week']:.3f}"
    print(f"{lab:26}& {r['kappa']:.2f} & {r['s']:.3f} & {r['Kf']} & {r['sigma_level']:.3f} & "
          f"{r['sigma_season']:.3f} & {sw} & {r['sigma_lin']:.4f} & {r['K_NB']:.3f} & "
          f"{r['mean_lag']:.2f} & {r['nb_size']:.3f} & {r['beta']:.3f} & "
          f"{r['sigma_g']:.3f} / {r['ell_g']:.2f} \\\\")
print("\ncov prior scales:", {k: v["cov_scales"] for k, v in headline_rows.items() if v["cov"]})

print("\n########## tab:supp-excitation-summaries ##########")
for lab, r in headline_rows.items():
    print(f"{lab:26}& {r['R_plus']:.4f} & {r['peak_lag']} & {r['lag50']} & {r['lag80']} & {r['lag90']} \\\\")

print("\n########## tab:supp-alt-metrics ##########")
for lab, r in headline_rows.items():
    print(f"{lab:26}& {r['mae']:.3f} & {r['rmse']:.3f} \\\\")

print("\n########## tab:realdata-calibration (all 7 cells) ##########")
for lab, r in rows.items():
    print(f"{lab:26}& {r['n_test']} & {r['kappa']:.2f} & {r['cov50']:.1f} & {r['cov80']:.1f} & "
          f"{r['cov95']:.1f} & {r['w50']:.2f} & {r['w80']:.2f} & {r['w95']:.2f} \\\\")
print("\ndone_marker")
