"""Synthetic recovery experiments (Section 5.1) in JAX.

The DHP-NB2 GENERATOR implements the manuscript's Section 5.1 specification: three excitation
kernels (NB: 0.6 * NBpmf(.; 6, .6); geometric: 0.8 * Geom(.; .6); delayed NB: K_NB=0.8, mean lag 8,
size 25) and three baseline regimes (linear / seasonal / linear+seasonal).  The model FIT reuses the
production GP-DHP code unchanged -- the bilevel forward-validation hyperparameter selection
(gpdhp_select.select), the collapsed MAP refit (bench_hawkes.find_map on build_fold_model), and the
closed-form component projection (baseline via _baseline_cols, excitation via m_NB + L theta_g).

Two experiments, each 3 scenarios x n_reps replicates, T=6000, dev/test split at t=4000:
  * excitation_recovery -> figure3.pdf : recover f(d) for NB / geometric / delayed-NB kernels
  * baseline_recovery   -> figure4.pdf : recover b(t) for linear / seasonal / linear+seasonal baselines

Usage:  python synthetic_experiments.py <experiment> <n_reps> [n_starts] [maxiter]
        experiment in {excitation, baseline, both};  --smoke = 1 scenario x 2 reps, tiny budget.
"""
import os, sys, json, time
os.environ.setdefault("JAX_PLATFORMS", "cpu"); os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from library import gpdhp_select as GS
from library._gpdhp_common import (build_design, _baseline_cols, softplus_link, build_fold_model)
from library.bench_hawkes import find_map

P = 365.0
OUT = os.path.join(os.path.dirname(os.path.dirname(HERE)), "paper")   # papercode/../paper
RES = os.path.join(HERE, "results_hawkes")
os.makedirs(RES, exist_ok=True)

# ------------------------------------------------------------------ generative (Section 5.1 spec)
def synth_baseline(kind, t):
    if kind == "linear":          return 0.5 + 0.001 * t
    if kind == "seasonal":        return 0.8 + 0.05 * np.cos(2*np.pi*t/P) + 0.25 * np.sin(2*np.pi*t/P)
    if kind == "linear_seasonal": return 0.5 + 0.001*t + 0.05*np.cos(2*np.pi*t/P) + 0.25*np.sin(2*np.pi*t/P)
    raise ValueError(kind)

def synth_excitation(kind, D):
    """f indexed 0..D-1 == lags 1..D (f[0] multiplies y(t-1))."""
    k = np.arange(D)
    if kind == "nb":        return 0.6 * stats.nbinom.pmf(k, 6, 0.6)         # K_NB=0.6, NBpmf(size=6, prob=.6)
    if kind == "geometric": return 0.8 * 0.6 * (0.4 ** k)                    # K_NB=0.8, Geom(prob=.6)
    if kind == "nb_base":   return 0.5 * stats.nbinom.pmf(k, 6, 0.6)         # baseline-recovery excitation
    if kind == "delayed":                                                    # K_NB*q_NB, mean lag 8, size 25
        q = stats.nbinom.pmf(k, 25, 25.0/(25.0+8.0)); return 0.8 * q / q.sum()
    raise ValueError(kind)

def simulate(baseline_kind, excitation_kind, T=6000, D_gen=200, kappa=20.0, seed=0):
    rng = np.random.RandomState(seed)
    t = np.arange(1, T+1, dtype=float)
    b = synth_baseline(baseline_kind, t); f = synth_excitation(excitation_kind, D_gen)
    y = np.zeros(T)
    for i in range(T):
        md = min(D_gen, i)
        exc = float(np.dot(y[i-md:i][::-1], f[:md])) if md > 0 else 0.0
        lam = max(b[i] + exc, 1e-6)
        y[i] = rng.negative_binomial(kappa, kappa/(kappa+lam))
    return y, b, f

# ------------------------------------------------------------------ fit (reuses production GP-DHP)
def fit_project(y_dev, y_test, D_fit, t_full, daily=False, n_starts=20, maxiter=300, seed=0):
    """FV-select hyperparameters, refit collapsed MAP, project b_hat(t_full) and f_hat(d)."""
    yd = np.asarray(y_dev, float); yt = np.asarray(y_test, float)
    sel = GS.select(yd, P, daily, D_max=D_fit, n_starts=n_starts, maxiter=maxiter, seed=seed)
    h, kappa, lk = sel["h"], sel["kappa"], sel["link_scale"]
    des = build_design(yd, h); L = np.asarray(des["L"]); m_NB = np.asarray(des["m_NB"]); q_b = des["q_b"]
    fm = build_fold_model(yd, yt, h, kappa, lk); q = fm["q"]
    th = np.asarray(find_map(fm["nlp"], q, jnp.zeros(q))[0]); tb = th[:q_b]; tg = th[q_b:]
    fhat = m_NB + (L @ tg if L.shape[1] else np.zeros_like(m_NB))
    B_full = np.asarray(_baseline_cols(jnp.asarray(t_full, jnp.float64), h["period"], h))
    bhat = B_full @ tb
    return dict(fhat=fhat, bhat=bhat, val_pll=sel["val_pll"], K_NB=float(sel["x"]["K_NB"]),
                sigma_u=float(sel["x"]["sigma_u"]), kappa=kappa)

# ------------------------------------------------------------------ scenarios (Section 5.1)
EXC_SCEN = [("nb",        "linear",          "NB excitation",       20.0),
            ("geometric", "seasonal",        "Geometric excitation", 20.0),
            ("delayed",   "linear_seasonal", "Delayed NB excitation", 20.0)]
BASE_SCEN = [("linear",          "Linear baseline"),
             ("seasonal",        "Seasonal baseline"),
             ("linear_seasonal", "Linear + seasonal baseline")]

BLUE = "#1f5b9e"; BAND = "#cfe0f2"


def run_experiment(which, n_reps, T, D_gen, D_fit, n_starts, maxiter, seed0):
    t_full = np.arange(1, T+1, dtype=float); split = T * 2 // 3   # dev = first 2/3 (t<=4000 for T=6000)
    store = {}
    scenarios = EXC_SCEN if which == "excitation" else [(s[0],) for s in BASE_SCEN]
    for si, scen in enumerate(scenarios):
        if which == "excitation":
            exc_kind, base_kind, title, kappa = scen
        else:
            base_kind = scen[0]; exc_kind = "nb_base"; kappa = 100.0
            title = dict(BASE_SCEN)[base_kind]
        fh_list, bh_list = [], []
        for r in range(n_reps):
            sd = seed0 + 1000*si + r
            y, b_tru, f_tru = simulate(base_kind, exc_kind, T=T, D_gen=D_gen, kappa=kappa, seed=sd)
            res = fit_project(y[:split], y[split:], D_fit, t_full, n_starts=n_starts, maxiter=maxiter, seed=sd)
            fh_list.append(res["fhat"]); bh_list.append(res["bhat"])
            print(f"[{which}] {title:28s} rep {r+1}/{n_reps}  valpLL={res['val_pll']:.1f} "
                  f"K_NB={res['K_NB']:.3f} sig_u={res['sigma_u']:.3g}", flush=True)
        store[base_kind if which == "baseline" else exc_kind] = dict(
            title=title, fhat=np.array(fh_list), bhat=np.array(bh_list),
            f_true=synth_excitation(exc_kind, D_fit), b_true=synth_baseline(base_kind, t_full))
    np.savez(os.path.join(RES, f"synthetic_{which}.npz"), t_full=t_full, split=split, D_fit=D_fit,
             store=np.array(store, dtype=object))
    return store, t_full, split


def band(a):  # mean + central 90% across replicates (axis 0)
    return a.mean(0), np.percentile(a, 5, 0), np.percentile(a, 95, 0)


def plot_excitation(store, Dshow=60):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    for ax, key in zip(axes, ["nb", "geometric", "delayed"]):
        s = store[key]; d = np.arange(1, s["fhat"].shape[1]+1)
        m, lo, hi = band(s["fhat"])
        ax.fill_between(d, lo, hi, color=BAND, lw=0, label="Central 90% fitted")
        ax.plot(d, m, color=BLUE, lw=1.6, label="Mean fitted")
        ax.plot(d, s["f_true"], "k--", lw=1.4, label="Truth")
        ax.set_xlim(1, Dshow); ax.set_title(s["title"], fontsize=10); ax.set_xlabel("Lag")
    axes[0].set_ylabel("Excitation kernel")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure3.pdf")); plt.close(fig)


def plot_baseline(store, t_full, split, Dshow=105):
    fig = plt.figure(figsize=(11, 6)); gs = fig.add_gridspec(2, 3, height_ratios=[1.05, 0.9])
    for j, key in enumerate(["linear", "seasonal", "linear_seasonal"]):
        s = store[key]; ax = fig.add_subplot(gs[0, j])
        m, lo, hi = band(s["bhat"])
        ax.fill_between(t_full, lo, hi, color=BAND, lw=0, label="Central 90% fitted")
        ax.plot(t_full, m, color=BLUE, lw=1.1, label="Mean fitted")
        ax.plot(t_full, s["b_true"], "k--", lw=1.1, label="Truth")
        ax.axvline(split, color="0.5", ls=":", lw=0.8); ax.set_title(s["title"], fontsize=10)
        ax.set_xlabel("Time index")
        if j == 0: ax.set_ylabel("Baseline"); ax.legend(frameon=False, fontsize=8, loc="upper left")
    axf = fig.add_subplot(gs[1, :]); s0 = store["linear"]; d = np.arange(1, s0["fhat"].shape[1]+1)
    allf = np.vstack([store[k]["fhat"] for k in ["linear", "seasonal", "linear_seasonal"]])
    m, lo, hi = band(allf)
    axf.fill_between(d, lo, hi, color=BAND, lw=0, label="Central 90% fitted")
    axf.plot(d, m, color=BLUE, lw=1.4, label="Mean fitted")
    axf.plot(d, synth_excitation("nb_base", len(d)), "k--", lw=1.4, label="Truth")
    axf.set_xlim(1, Dshow); axf.set_title("Fixed excitation", fontsize=10)
    axf.set_xlabel("Lag"); axf.set_ylabel("Excitation kernel")
    axf.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "figure4.pdf")); plt.close(fig)


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv; argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    which = argv[0] if argv else "both"
    n_reps = int(argv[1]) if len(argv) > 1 else (2 if smoke else 10)
    n_starts = int(argv[2]) if len(argv) > 2 else (3 if smoke else 20)
    maxiter = int(argv[3]) if len(argv) > 3 else (60 if smoke else 300)
    T = 900 if smoke else 6000; D_gen = 200; D_fit = 100
    t0 = time.time()
    exps = ["excitation", "baseline"] if which == "both" else [which]
    for e in exps:
        st, tf, sp = run_experiment(e, n_reps, T, D_gen, D_fit, n_starts, maxiter, seed0=1)
        if not smoke:
            (plot_excitation if e == "excitation" else plot_baseline)(st, *( () if e=="excitation" else (tf, sp)))
    print(f"done in {time.time()-t0:.0f}s")
