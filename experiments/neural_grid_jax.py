"""Grid-search hyperparameter selection for the four neural NB models (MLP-NB, GRU-NB,
LSTM-NB, DeepAR-NB) defined in ``neural_nb_jax`` -- the sole neural selector used by the
paper (``run_neural.py`` subprocesses this module).  It enumerates a small grid of sensible
hyperparameters -- 2*3*4*2*3 = 144 fits per (dataset, model) -- and keeps the combination
with the highest 40% forward-validation NB predictive log-likelihood (train Lion on the
first 60% of the fitting period, score one-step-ahead on the last 40%).

Training is entirely in JAX: optax **Lion**, each fit a single jitted ``lax.scan`` over
epochs that carries and returns the best-validation parameters (early stopping with
unbounded patience); the learning rate is a *traced* argument via
``optax.inject_hyperparams`` so the step compiles once per architecture shape.  Dropout is a
2-value axis {0.0, 0.1}; the grid excludes the longest context windows, which dominate RNN
runtime.  Grid search is the only neural selection path.

Runs ONE (dataset, model) at a time in a single process (GPU-memory-safe).  For each run it
saves the best hyperparameters + test pLL to a consolidated CSV, and the full 144-point grid
(sorted by validation pLL) to a per-run JSON, plus a per-observation-log-score npz that is
Diebold--Mariano-comparable to GP-DHP.

  python neural_grid_jax.py                                   # all 4 models x 5 datasets
  python neural_grid_jax.py --datasets nyc,gva,rand --models GRU-NB,LSTM-NB,DeepAR-NB
  python neural_grid_jax.py --max-epochs 300 --verbose        # heavier training, print every grid point

Grid values are the module-level GRID_* constants -- edit them to widen/narrow the search.
Set the env var HAWKES_ROOT to override the data root.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from functools import partial

import numpy as np
import pandas as pd
import jax
jax.config.update("jax_enable_x64", False)      # train in float32 (the x64 equivalence test is a separate process)
import jax.numpy as jnp
import optax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from library import neural_nb_jax as J
from library import covariates as CV
from library import datasets as DS

HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # papercode/
MODELS = ["MLP-NB", "GRU-NB", "LSTM-NB", "DeepAR-NB"]


# --------------------------------------------------------------------------- #
# Training + forward-validation helpers for the neural NB models
# (numerics validated to ~1e-14 against the original torch reference implementation).
# --------------------------------------------------------------------------- #
# --- torch-free helpers (mirror neural_nb.py) ---
def inv_softplus(x):
    x = max(float(x), 1e-5)
    return x if x > 30 else math.log(math.expm1(x))


def scale_from_train(y):
    y = np.asarray(y, float); pos = y[y > 0]
    return max(1.0, float(np.mean(pos))) if len(pos) else max(1.0, float(np.mean(np.abs(y))) + 1.0)


def mom_kappa(y):
    y = np.asarray(y, float)
    if len(y) < 2:
        return 20.0
    m, v = float(np.mean(y)), float(np.var(y, ddof=1))
    return float(np.clip(m * m / (v - m), 0.25, 1000.0)) if (v > m + 1e-6 and m > 1e-6) else 200.0


def supervised(counts, target_positions, context_length, count_scale, cov):
    """Lag-window design matching neural_nb.make_supervised (require_full_context=False),
    vectorized (no Python loop): X_mlp = [newest..oldest scaled lags, target covariates];
    X_seq = chronological [scaled lag, covariates] steps.  Returns (X_mlp, X_seq,
    target_cov, y) as jnp arrays."""
    counts = np.asarray(counts, float); cov = np.asarray(cov, float); cd = cov.shape[1]
    tp = np.asarray(target_positions, int)
    src = tp[:, None] - np.arange(context_length, 0, -1)[None, :]     # (n, ctx), oldest..newest
    valid = src >= 0
    csrc = np.clip(src, 0, None)
    lag = np.where(valid, counts[csrc] / count_scale, 0.0)            # scaled lagged counts
    if cd:
        cov_src = np.where(valid[..., None], cov[csrc], 0.0)
        Xseq = np.concatenate([lag[..., None], cov_src], axis=-1)     # (n, ctx, 1+cd), chronological
        tcov = cov[tp]
    else:
        Xseq = lag[..., None]                                         # (n, ctx, 1)
        tcov = np.zeros((len(tp), 0))
    Xmlp = np.concatenate([lag[:, ::-1], tcov], axis=1)              # newest..oldest lags + target cov
    return (jnp.asarray(Xmlp), jnp.asarray(Xseq), jnp.asarray(tcov.reshape(len(tp), cd)), jnp.asarray(counts[tp]))


# --- parameter initialization (PyTorch-style; output head zero-weight + data-derived bias) ---
def _unif(key, shape, k):
    return jax.random.uniform(key, shape, jnp.float32, -k, k)


def init_params(model, key, cov_dim, context, hidden, layers, mu_init, kappa_init, nu):
    ks = list(jax.random.split(key, max(24, 4 * layers + 4)))   # enough keys for deep RNNs (4/layer + head); >=24 keeps layers<=5 unchanged
    b_mu = inv_softplus(max(mu_init / nu, 1e-5))         # training scales mu by nu (mu = nu*softplus)
    sd = {}
    if model == "MLP-NB":
        dims = [context + cov_dim] + [hidden] * layers + [1]        # layers hidden Linears + final
        for i in range(layers + 1):
            fin = dims[i]
            if i < layers:
                sd[f"net.{i}.weight"] = _unif(ks[2 * i], (dims[i + 1], fin), 1 / math.sqrt(fin))
                sd[f"net.{i}.bias"] = _unif(ks[2 * i + 1], (dims[i + 1],), 1 / math.sqrt(fin))
            else:
                sd[f"net.{i}.weight"] = jnp.zeros((1, fin))          # final: zero weight, bias=inv_softplus(mu)
                sd[f"net.{i}.bias"] = jnp.full((1,), b_mu)
        sd["raw_kappa"] = jnp.asarray(inv_softplus(kappa_init))
        return sd
    gates = 3 if model == "GRU-NB" else 4                            # GRU has 3 gates, LSTM/DeepAR 4
    ki = 0
    for l in range(layers):
        ind = (1 + cov_dim) if l == 0 else hidden
        kk = 1 / math.sqrt(hidden)
        sd[f"rnn.weight_ih_l{l}"] = _unif(ks[ki], (gates * hidden, ind), kk); ki += 1
        sd[f"rnn.weight_hh_l{l}"] = _unif(ks[ki], (gates * hidden, hidden), kk); ki += 1
        sd[f"rnn.bias_ih_l{l}"] = _unif(ks[ki], (gates * hidden,), kk); ki += 1
        sd[f"rnn.bias_hh_l{l}"] = _unif(ks[ki], (gates * hidden,), kk); ki += 1
    if model == "DeepAR-NB":
        sd["mu_head.weight"] = jnp.zeros((1, hidden + cov_dim))
        sd["mu_head.bias"] = jnp.full((1,), inv_softplus(max(mu_init / nu, 1e-5)))
        sd["kappa_head.weight"] = jnp.zeros((1, hidden + cov_dim))
        sd["kappa_head.bias"] = jnp.full((1,), inv_softplus(max(kappa_init / math.sqrt(nu), 1e-5)))
    else:
        fin = hidden + cov_dim
        sd["head.0.weight"] = _unif(ks[ki], (hidden, fin), 1 / math.sqrt(fin)); ki += 1
        sd["head.0.bias"] = _unif(ks[ki], (hidden,), 1 / math.sqrt(fin)); ki += 1
        sd["head.1.weight"] = jnp.zeros((1, hidden))                 # final: zero weight, bias=inv_softplus(mu)
        sd["head.1.bias"] = jnp.full((1,), b_mu)
        sd["raw_kappa"] = jnp.asarray(inv_softplus(kappa_init))
    return sd


# --- jitted training run (Lion; best-validation params over epochs) ---
@partial(jax.jit, static_argnums=(0, 1, 2))
def train_run(model, num_layers, max_epochs, params0, Xtr, Xval, lr, dropout, nu, rng):
    Xmlp_t, Xseq_t, tc_t, y_t = Xtr
    Xmlp_v, Xseq_v, tc_v, y_v = Xval
    opt = optax.inject_hyperparams(optax.lion)(learning_rate=lr, weight_decay=1e-3)
    state0 = opt.init(params0)

    def val_pll(p):
        mu, k = J.apply_model(model, p, Xmlp_v, Xseq_v, tc_v, num_layers, nu)
        return jnp.sum(J.nb_logpmf(y_v, mu, k))

    def epoch(carry, ek):
        params, state, best_p, best_v = carry

        def loss_fn(p):
            mu, k = J.apply_train(model, p, Xmlp_t, Xseq_t, tc_t, num_layers, ek, dropout, nu)
            return -jnp.mean(J.nb_logpmf(y_t, mu, k))

        _, g = jax.value_and_grad(loss_fn)(params)
        upd, state = opt.update(g, state, params)
        params = optax.apply_updates(params, upd)
        v = val_pll(params)
        better = v > best_v
        best_p = jax.tree_util.tree_map(lambda a, b: jnp.where(better, a, b), params, best_p)
        best_v = jnp.where(better, v, best_v)
        return (params, state, best_p, best_v), v

    (_, _, best_p, best_v), _ = jax.lax.scan(
        epoch, (params0, state0, params0, -jnp.inf), jax.random.split(rng, max_epochs))
    return best_v, best_p


_DESIGN_CACHE = {}


def _design(counts, tr_pos, va_pos, context, cov):
    """Lag-window design + data-derived scalars for a (data, context, split). These do
    NOT depend on hidden/layers/dropout/lr, so they are built (and host->device
    transferred) once per context and reused across every candidate in a grid run,
    instead of being rebuilt for each of the ~144 evaluations."""
    key = (int(context), int(cov.shape[1]), len(counts), len(tr_pos), len(va_pos),
           round(float(counts.sum()), 3), round(float(cov.sum()) if cov.size else 0.0, 3))
    d = _DESIGN_CACHE.get(key)
    if d is None:
        scale = scale_from_train(counts[tr_pos])
        d = dict(scale=scale, mu_init=max(float(np.mean(counts[tr_pos])), 1e-6),
                 kappa_init=mom_kappa(counts[tr_pos]),
                 Xtr=supervised(counts, tr_pos, context, scale, cov),
                 Xva=supervised(counts, va_pos, context, scale, cov))
        _DESIGN_CACHE[key] = d
    return d


def _prep(model, counts, tr_pos, va_pos, hyper, cov, seed):
    d = _design(counts, tr_pos, va_pos, hyper["context"], cov)
    ki, kj = jax.random.split(jax.random.PRNGKey(seed))
    p0 = init_params(model, ki, cov.shape[1], hyper["context"], hyper["hidden"], hyper["layers"],
                     d["mu_init"], d["kappa_init"], d["scale"])
    return p0, d["Xtr"], d["Xva"], d["scale"], kj


def fv_score(model, y_dev, hyper, is_daily, cov_dev=None, max_epochs=300, seed=0, min_train=0.6):
    """Forward-validation objective: train on the first 60% of development, return the
    best one-step validation NB pLL on the last 40%."""
    counts = np.asarray(y_dev, float); n = len(counts); a = int(min_train * n)
    cov = np.asarray(cov_dev, float) if cov_dev is not None else np.zeros((n, 0))
    p0, Xtr, Xva, scale, kj = _prep(model, counts, np.arange(a), np.arange(a, n), hyper, cov, seed)
    best_v, _ = train_run(model, hyper["layers"], max_epochs, p0, Xtr, Xva,
                          jnp.float32(hyper["lr"]), jnp.float32(hyper["dropout"]), jnp.float32(scale), kj)
    v = float(best_v)
    return v if np.isfinite(v) else -1e10


def refit_and_test(model, y_dev, y_test, hyper, is_daily, cov_dev=None, cov_test=None, max_epochs=300, seed=0):
    """Refit with the 60/40 split (best-validation params) and score the held-out test set."""
    dev = np.asarray(y_dev, float); test = np.asarray(y_test, float); n = len(dev); a = int(0.6 * n)
    cov = np.asarray(cov_dev, float) if cov_dev is not None else np.zeros((n, 0))
    p0, Xtr, Xva, scale, kj = _prep(model, dev, np.arange(a), np.arange(a, n), hyper, cov, seed)
    best_v, best_p = train_run(model, hyper["layers"], max_epochs, p0, Xtr, Xva,
                               jnp.float32(hyper["lr"]), jnp.float32(hyper["dropout"]), jnp.float32(scale), kj)
    counts_all = np.concatenate([dev, test])
    cov_all = np.zeros((len(counts_all), cov.shape[1])) if cov_dev is None else \
        np.vstack([cov, np.asarray(cov_test, float)])
    Xte = supervised(counts_all, np.arange(n, n + len(test)), hyper["context"], scale, cov_all)
    mu, k = J.apply_model(model, best_p, Xte[0], Xte[1], Xte[2], hyper["layers"], scale)
    logscores = np.asarray(J.nb_logpmf(Xte[3], mu, k), dtype=float)         # per-observation test NB log-scores
    return float(best_v), float(logscores.sum()), logscores, np.asarray(Xte[3], dtype=float)


def dm_test(d):
    """Diebold-Mariano on a per-observation score differential d = A - B (higher-is-
    better log-scores). Newey-West HAC variance (bandwidth floor(n^(1/3))) + the
    Harvey-Leybourne-Newbold small-sample correction (one-step, h=1); two-sided p from
    t(n-1). Mirrors bench_shared.dm_test used for the GP-DHP comparisons."""
    from scipy import stats
    d = np.asarray(d, float); n = len(d)
    if n < 3:
        return (float(d.mean()) if n else 0.0), 0.0, 1.0
    dbar = float(d.mean()); dd = d - dbar
    L = int(n ** (1.0 / 3.0))
    lrv = float(np.mean(dd * dd))
    for lag in range(1, L + 1):
        lrv += 2.0 * (1.0 - lag / (L + 1.0)) * float(np.mean(dd[lag:] * dd[:-lag]))
    var = lrv / n
    if var <= 0:
        return dbar, 0.0, 1.0
    stat = (dbar / np.sqrt(var)) * np.sqrt((n - 1.0) / n)
    return dbar, float(stat), float(2.0 * stats.t.sf(abs(stat), df=n - 1))


def save_result_npz(path, dataset, model, hyper, val_pll, test_pll, logscores, observed,
                    n_evals, seconds, gpdhp_dir=None, covariates="none"):
    """Save a neural result in the GP-DHP ``benchmarks/*.npz`` style -- per-observation
    ``logscores`` + ``observed`` so it is Diebold-Mariano-comparable to GP-DHP. If a
    GP-DHP npz for this dataset is found under ``gpdhp_dir`` (or $GPDHP_NPZ_DIR), also
    store the DM test of the neural model vs GP-DHP (MAP). Returns (gpdhp_pll, t, p)/None."""
    logscores = np.asarray(logscores, float); observed = np.asarray(observed, float)
    mode = "fixed_allpast" if DS.DATA[dataset][2] is not None else "single_split"
    out = dict(dataset=dataset, model=model, mode=mode, covariates=covariates,
               logscores=logscores, observed=observed, test_pll=float(test_pll),
               val_pll=float(val_pll), sel_hypers=json.dumps(hyper),
               n_evals=int(n_evals), seconds=float(seconds))
    gpdhp_dir = gpdhp_dir or os.environ.get("GPDHP_NPZ_DIR")
    dm = None
    if gpdhp_dir:
        gp = os.path.join(gpdhp_dir, f"{dataset}.npz")
        if os.path.exists(gp):
            z = np.load(gp, allow_pickle=True)
            gp_ls = np.asarray(z["map_logscores"], float)          # GP-DHP (MAP) per-obs log-scores
            if len(gp_ls) == len(logscores):
                dbar, t, p = dm_test(logscores - gp_ls)            # neural - GP-DHP(MAP); >0 => neural better
                out.update(gpdhp_pll=float(gp_ls.sum()), dm_vs_gpdhp_meandiff=dbar,
                           dm_vs_gpdhp_t=t, dm_vs_gpdhp_p=p)
                dm = (float(gp_ls.sum()), t, p)
            else:
                out["dm_note"] = f"n mismatch neural={len(logscores)} gpdhp={len(gp_ls)}"
    np.savez_compressed(path, **out)
    return dm


def load_dengue_like(dataset):
    # Shared split table/logic (library.datasets); returns (dev, test, is_daily).
    dev, test, _period, daily = DS.load_split(dataset)
    return dev, test, daily


# --------------------------------------------------------------------------- #
# The grid (2*3*4*2*3 = 144 combos per dataset x model)
# --------------------------------------------------------------------------- #
GRID_CTX = {True: [30, 90], False: [26, 52]}            # daily / weekly context windows (skips the slow 100/180)
GRID_HID = [20, 40, 80]                                 # hidden size (width)
GRID_LAY = [1, 2, 3, 4]                                 # number of layers (depth)
GRID_DROP = [0.0, 0.1]                                  # dropout rate (2-value axis matching the completed npz grid)
GRID_LR = [1e-4, 3e-4, 1e-3]                            # Lion learning rate


def grid_combos(is_daily):
    return [dict(context=c, hidden=h, layers=l, dropout=dr, lr=lr)
            for c in GRID_CTX[is_daily] for h in GRID_HID for l in GRID_LAY
            for dr in GRID_DROP for lr in GRID_LR]


def grid_select(model, y_dev, is_daily, max_epochs=100, seed=0, verbose=False, cov_dev=None, limit=None):
    """Evaluate every grid point by 40% forward-validation pLL; return the best.
    `limit` (smoke tests only) truncates the grid to the first N combos."""
    combos = grid_combos(is_daily)
    if limit:
        combos = combos[:int(limit)]
    grid = []
    for hp in combos:
        v = fv_score(model, y_dev, hp, is_daily, cov_dev=cov_dev, max_epochs=max_epochs, seed=seed)
        grid.append({**hp, "val_pll": round(v, 3)})
        if verbose:
            print(f"      ctx={hp['context']:>3} hid={hp['hidden']:>2} L={hp['layers']} "
                  f"drop={hp['dropout']:.2f} lr={hp['lr']:.0e}  val_pLL={v:.2f}", flush=True)
    grid.sort(key=lambda g: -g["val_pll"])
    best = grid[0]
    hyper = {k: best[k] for k in ("context", "hidden", "layers", "dropout", "lr")}
    return dict(hyper=hyper, val_pll=best["val_pll"], n_evals=len(combos), grid=grid)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="dengue,crypto,nyc,gva,rand")
    ap.add_argument("--models", default="MLP-NB,GRU-NB,LSTM-NB,DeepAR-NB")
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--verbose", action="store_true", help="print every grid point's validation pLL")
    ap.add_argument("--no-clear-caches", action="store_true")
    ap.add_argument("--gpdhp-dir", default=None,
                    help="dir holding GP-DHP benchmarks/<dataset>.npz; if given (or $GPDHP_NPZ_DIR), "
                         "also DM-test each neural model vs GP-DHP (MAP) on the shared test points")
    ap.add_argument("--cov", action="store_true",
                    help="feed the temp+holiday covariates (nyc/gva only); outputs are tagged '_cov'")
    ap.add_argument("--outdir", default=None,
                    help="dir for per-run npz/json + consolidated CSV (default experiments/results_hawkes/neural_grid); "
                         "the run_neural orchestrator points this at a scratch dir it later cleans up")
    ap.add_argument("--limit-combos", type=int, default=None,
                    help="SMOKE TESTS ONLY: truncate the grid to the first N combos")
    a = ap.parse_args()
    variant = "cov" if a.cov else "nocov"
    datasets = [d.strip() for d in a.datasets.split(",") if d.strip()]
    models = [m.strip() for m in a.models.split(",") if m.strip()]

    outdir = a.outdir or os.path.join(HAWKES, "experiments", "results_hawkes", "neural_grid")
    os.makedirs(outdir, exist_ok=True)
    out = a.out or os.path.join(outdir, "neural_grid_results.csv")

    n_weekly, n_daily = len(grid_combos(False)), len(grid_combos(True))
    print(f"JAX backend: {jax.default_backend()} | devices: {jax.devices()}", flush=True)
    print(f"grid: {n_weekly} combos (weekly) / {n_daily} (daily) per run | "
          f"{len(datasets)} datasets x {len(models)} models | max_epochs={a.max_epochs}\n", flush=True)

    rows = []
    t_all = time.time()
    for ds in datasets:
        try:
            dev, test, is_daily = load_dengue_like(ds)
        except Exception as e:
            print(f"!! could not load {ds}: {e}", flush=True)
            continue
        cov_dev = cov_test = None
        if a.cov:
            if not CV.has_covariates(ds):
                print(f"!! --cov set but {ds} has no covariates; skipping\n", flush=True)
                continue
            cov_dev, cov_test, _ = CV.build_covariates(ds, root=HAWKES)
        for m in models:
            tag = f"{ds}/{variant}/{m}"
            print(f">>> {tag}  (dev={len(dev)} test={len(test)} daily={is_daily}, {len(grid_combos(is_daily))} grid points) ...", flush=True)
            t0 = time.time()
            try:
                r = grid_select(m, dev, is_daily, max_epochs=a.max_epochs, seed=a.seed, verbose=a.verbose,
                                cov_dev=cov_dev, limit=a.limit_combos)
                _, test_pll, logscores, observed = refit_and_test(m, dev, test, r["hyper"], is_daily,
                                                                  cov_dev=cov_dev, cov_test=cov_test,
                                                                  max_epochs=a.max_epochs, seed=a.seed)
                h = r["hyper"]
                row = dict(dataset=ds, model=m, variant=variant, status="ok",
                           val_pll=r["val_pll"], test_pll=round(test_pll, 3),
                           n_dev=len(dev), n_test=len(test), context=h["context"], hidden=h["hidden"],
                           layers=h["layers"], dropout=h["dropout"], lr=float(h["lr"]),
                           n_evals=r["n_evals"], seconds=round(time.time() - t0, 1))
                dm = save_result_npz(os.path.join(outdir, f"{ds}_{variant}_{m}.npz"), ds, m, h, r["val_pll"],
                                     test_pll, logscores, observed, r["n_evals"], row["seconds"],
                                     gpdhp_dir=a.gpdhp_dir, covariates=("temp+holidays" if a.cov else "none"))
                if dm:
                    row.update(gpdhp_pll=round(dm[0], 2), dm_vs_gpdhp_t=round(dm[1], 3), dm_vs_gpdhp_p=round(dm[2], 4))
                with open(os.path.join(outdir, f"{ds}_{variant}_{m}.json"), "w") as fh:
                    json.dump({**row, "grid": r["grid"]}, fh, indent=2)   # full grid, sorted best-first
                dm_str = f" | vs GP-DHP t={dm[1]:+.2f} p={dm[2]:.3f}" if dm else ""
                print(f"    {tag}: BEST val_pLL={row['val_pll']} test_pLL={row['test_pll']}  "
                      f"ctx={h['context']} hid={h['hidden']} L={h['layers']} drop={h['dropout']} "
                      f"lr={row['lr']:.0e}{dm_str}  ({row['seconds']}s)\n", flush=True)
            except Exception as e:
                row = dict(dataset=ds, model=m, variant=variant, status="failed", error=str(e), seconds=round(time.time() - t0, 1))
                print(f"    !! {tag} FAILED: {e}\n{traceback.format_exc()}\n", flush=True)
            rows.append(row)
            pd.DataFrame(rows).to_csv(out, index=False)                  # incremental save
            if not a.no_clear_caches:
                jax.clear_caches()

    df = pd.DataFrame(rows)
    print(f"=== done: {len(rows)} runs in {time.time()-t_all:.0f}s -> {out} ===")
    cols = [c for c in ["dataset", "model", "val_pll", "test_pll", "gpdhp_pll", "dm_vs_gpdhp_t", "dm_vs_gpdhp_p",
                        "context", "hidden", "layers", "dropout", "lr", "status"] if c in df]
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
