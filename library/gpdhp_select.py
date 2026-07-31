"""Forward-validation hyperparameter selection for the softplus GP-DHP by MULTI-START
L-BFGS-MT on the 40% held-out validation pLL.

Protocol: for a hyperparameter vector the whitened latent is MAP-fitted on the 60% train
window and scored by one-step-ahead NB predictive log-likelihood on the last-40% validation
window (gpdhp_fv_grad.make_fv_fns builds this FV bilevel + its analytic hypergradient).  select()
samples n_starts random hyperparameter initializations -- each coordinate log-uniform for the
sd/variance/rate params, linear for the bounded ones, via _unit_to_nat of a uniform unit-cube
draw; K_fourier round-robin -- and refines EACH by box-constrained L-BFGS-MT on the ANALYTIC
bilevel hypergradient (the implicit-function-theorem adjoint of the inner train-MAP argmin, never
autodiffing the solver), all in parallel across a thread pool of CPU cores
(jax_baselines.multistart_lbfgs), keeping the start with the best validation pLL.  The selected
hyperparameters are returned as a clean h-dict (+ kappa, link_scale) for the final full-period
refit; kappa is one of the sampled hyperparameters (never fitted with the latent), and the
subsequent MAP fits only the latent at those fixed hyperparameters.
"""
import os
import sys
import numpy as np
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
_HERE = os.path.dirname(os.path.abspath(__file__))                    # papercode/library
_HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(_HERE))       # papercode
FLOOR = 1e-6


def _warped_rbf_chol(D, beta, sigma_u, ell_u, eps_f=1e-4):
    """jit-safe warped-RBF lower-Cholesky (beta>0 branch of the common version;
    the search keeps beta in [0.05, 0.5], so the beta->0 limit is never hit)."""
    d = jnp.arange(1, D + 1, dtype=jnp.float64)
    a = sigma_u * jnp.exp(-beta * d / 2)
    g = (1.0 - jnp.exp(-beta * d)) / (beta * ell_u)
    K = jnp.outer(a, a) * jnp.exp(-0.5 * (g[:, None] - g[None, :]) ** 2) + (eps_f ** 2) * jnp.eye(D)
    K = 0.5 * (K + K.T)
    return jnp.linalg.cholesky(K + 1e-12 * jnp.eye(D))


def _lag_matrix(y, D):
    """X[t,d] = y[t-d] for d=1..D (0 before the start). Vectorized indexing."""
    y = np.asarray(y, float); T = len(y)
    idx = np.arange(T)[:, None] - np.arange(1, D + 1)[None, :]
    return jnp.asarray(np.where(idx >= 0, y[np.clip(idx, 0, None)], 0.0))


# (lo, hi) bounds and log-scale flags for each hyperparameter.
BOUNDS = dict(sigma_level=(0.5, 20.), sigma_fourier=(0.01, 20.), sigma_lin=(1e-8, 1e-2),
              K_NB=(0., 1.25), mean_lag=(0.25, 64.), size=(0.25, 50.), kappa=(0.25, 1e6),
              sigma_u=(1e-4, 10.), ell_u=(1., 30.), beta=(0.05, 0.5), link_scale=(0.02, 2.),
              sigma_weekly=(1e-3, 20.))
LOG = dict(sigma_level=1, sigma_fourier=1, sigma_lin=1, K_NB=0, mean_lag=1, size=1,
           kappa=1, sigma_u=1, ell_u=1, beta=1, link_scale=1, sigma_weekly=1)

NAMES_BASE = ["sigma_level", "sigma_fourier", "sigma_lin", "K_NB", "mean_lag", "size",
              "kappa", "sigma_u", "ell_u", "beta", "link_scale"]


def names_for(daily):
    return NAMES_BASE + (["sigma_weekly"] if daily else [])


def build_h(x, K_fourier, daily, period, D_max=100, threshold=False):
    """Assemble the h-dict. With threshold=True, apply the zeroing rules that turn
    off nearly-inactive blocks in the final selected specification."""
    sw = x.get("sigma_weekly", 0.0)
    sl = x["sigma_lin"]
    su = x["sigma_u"]
    if threshold:
        sl = 0.0 if sl < 1e-6 else sl
        su = 0.0 if su < 1e-3 else su
        sw = 0.0 if sw < 2e-3 else sw
    return dict(D_max=int(D_max), period=float(period), K_fourier=int(K_fourier),
                sigma_level=float(x["sigma_level"]), sigma_fourier=float(x["sigma_fourier"]),
                sigma_lin=float(sl), sigma_weekly=float(sw if daily else 0.0),
                K_weekly=3, period_weekly=7.0, K_NB=float(x["K_NB"]),
                mean_lag=float(x["mean_lag"]), size=float(x["size"]),
                beta=float(x["beta"]), sigma_u=float(su), ell_u=float(x["ell_u"]))


def _unit_to_nat(u, names):
    """Map a unit-hypercube point u in [0,1]^d to natural hyperparameters (in the
    names_for(daily) order make_fv_fns expects), applying each coordinate's log/linear bound map."""
    nat = np.empty(len(names))
    for i, nm in enumerate(names):
        lo, hi = BOUNDS[nm]
        ui = float(min(max(u[i], 0.0), 1.0))
        nat[i] = np.exp(np.log(lo) + ui * (np.log(hi) - np.log(lo))) if LOG[nm] else lo + ui * (hi - lo)
    return nat


def select(y_fit, period, daily, D_max=100, min_train=0.6, n_starts=20, maxiter=300,
           seed=0, n_devices=10, verbose=False, use_fft=True):
    """Select GP-DHP hyperparameters by MULTI-START L-BFGS-MT on the 40% held-out validation pLL:
    sample n_starts random hyperparameter initializations (each coordinate log-uniform for
    sd/variance/rate params, linear for the bounded ones, via _unit_to_nat of a uniform unit-cube
    draw; K_fourier round-robin) and refine EACH by box-constrained L-BFGS-MT (<=maxiter) on the
    ANALYTIC FV bilevel hypergradient (gpdhp_fv_grad.make_fv_fns -- the implicit-function-theorem
    adjoint of the inner train-MAP argmin, never autodiffing the solver), all in parallel across a
    thread pool of n_devices CPU cores (jax_baselines.multistart_lbfgs -- jitted L-BFGS trajectories
    dispatched across threads, NOT vmap); keep the best validation pLL.  `seed` fixes the sampler.
    kappa is one of the sampled hyperparameters (never fitted with the latent); the subsequent MAP
    fits ONLY the latent at the selected hyperparameters.

    y_fit : counts of the full fitting period (all data before the test window).
    Returns dict(h, kappa, link_scale, K_fourier, val_pll, x, n_evals, refine=None)."""
    from . import gpdhp_fv_grad as FG
    from .jax_baselines import multistart_lbfgs
    y_fit = np.asarray(y_fit, float)
    names = names_for(daily); dims = len(names)
    fvsets = {Kf: FG.make_fv_fns(y_fit, period, daily, Kf, D_max, min_train=min_train, use_fft=use_fft)
              for Kf in (1, 2, 3)}
    neg_fvs = {Kf: fvsets[Kf]["neg_fv"] for Kf in (1, 2, 3)}           # the FV validation-NLL custom_vjp per Kf
    bounds = [BOUNDS[nm] for nm in names]
    sample_fn = lambda rng: _unit_to_nat(rng.random(dims), names)      # log/linear per-coord random start
    nat, pll, Kf = multistart_lbfgs(neg_fvs, [1, 2, 3], bounds, sample_fn,
                                    n_starts=n_starts, maxiter=maxiter, seed=seed, n_devices=n_devices)
    if verbose:
        print(f"  multi-start L-BFGS-MT: {n_starts} starts, best Kf={int(Kf)} val_pLL={float(pll):.3f}", flush=True)
    x = dict(zip(names, [float(v) for v in nat]))
    h = build_h(x, int(Kf), daily, period, D_max, threshold=True)
    return dict(h=h, kappa=float(x["kappa"]), link_scale=float(x["link_scale"]),
                K_fourier=int(Kf), val_pll=float(pll), score=float(pll),
                objective="fv_multistart", x=x, n_evals=int(n_starts), refine=None)


if __name__ == "__main__":
    import argparse, pandas as pd, time
    HAWKES = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA = {"dengue": ("dengue_singapore", range(2018, 2023)),
            "crypto": ("crypto_survstat_de", range(2015, 2020)),
            "nyc": ("nyc_shootings", None), "gva": ("gva_allshootings", None)}
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", required=True)
    a = ap.parse_args()
    data_name, years = DATA[a.dataset]
    df = pd.read_csv(f"{HAWKES}/data/{data_name}/dataset_counts.csv")
    df["year"] = df["date"].str[:4].astype(int)
    period = int(df["period"].iloc[0]); daily = period == 365
    if years is not None:
        y_fit = df[df.year < min(years)]["count"].to_numpy(float)
    else:
        y_fit = df[df.final_split == "development"]["count"].to_numpy(float)
    t0 = time.time()
    r = select(y_fit, period, daily, verbose=True)
    print(f"\n[{a.dataset}] GP-DHP MULTI-START SELECTED Kf={r['K_fourier']} val_pLL={r['val_pll']:.3f} "
          f"({r['n_evals']} starts, {time.time()-t0:.0f}s)")
    print("  kappa=%.4g link_scale=%.4g" % (r["kappa"], r["link_scale"]))
    print("  h=", {k: (round(v, 5) if isinstance(v, float) else v) for k, v in r["h"].items()})
