"""Shared benchmark folding for the two serial orchestrators (run_nonneural.py, run_neural.py).

Both accumulate per-observation NB log-scores into a SINGLE npz with the exact schema the old
run_master.py produced, so any pair of models is Diebold-Mariano-comparable:

  flat keys : "<ds>/<variant>/observed"            -> shared observed test counts (per cell)
              "<ds>/<variant>/<model>/logscores"   -> per-obs NB log-scores for each model
  "manifest": JSON list of records, one per (ds, variant, model):
              {dataset, variant, model, pll, n_test, [val_pll, sel_hypers, ...],
               [dm_vs_gpdhp_meandiff, dm_vs_gpdhp_t, dm_vs_gpdhp_p, dm_vs_gpdhp_p_holm]}

`fold_model` adds ONE model to the npz and atomically rewrites it, recomputing the DM test vs
GP-DHP(MAP) and the Holm correction over the cell's non-reference models present so far -- so the
file is valid and up to date after every single model fit (incremental).  `dm_test`/`holm` are the
exact JAX-free copies used by run_master (matching gpdhp_runner and the manuscript)."""
import json
import os
import numpy as np

GP_REF = "GP-DHP (MAP)"                                 # the DM reference model, present in every cell


def dm_test(d):
    """Diebold-Mariano on a per-obs log-score differential d (Newey-West HAC + HLN small-sample
    correction).  Positive mean favours the FIRST model.  Returns (dbar, dm_stat, two_sided_p)."""
    from scipy import stats
    d = np.asarray(d, float); n = len(d); dbar = d.mean()
    L = max(1, int(np.floor(n ** (1 / 3))))
    s = np.mean((d - dbar) ** 2)
    for k in range(1, L + 1):
        s += 2 * (1 - k / (L + 1)) * np.mean((d[k:] - dbar) * (d[:-k] - dbar))
    var = s / n
    if var <= 0:
        return float(dbar), float("nan"), float("nan")
    dm = dbar / np.sqrt(var) * np.sqrt((n + 1 - 2 * L + L * (L - 1) / n) / n)
    return float(dbar), float(dm), float(2 * stats.t.sf(abs(dm), df=n - 1))


def holm(pvals):
    """Holm-Bonferroni step-down adjusted p-values (input order preserved; non-finite stay nan and
    are excluded from the family size)."""
    p = np.asarray(pvals, float); m = len(p)
    adj = np.full(m, np.nan)
    idx = np.where(np.isfinite(p))[0]
    if not len(idx):
        return adj
    order = idx[np.argsort(p[idx])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(idx) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def _load(path):
    store, manifest = {}, []
    if os.path.exists(path):
        with np.load(path, allow_pickle=True) as z:            # materialise then close before rewrite
            store = {k: z[k] for k in z.files if k != "manifest"}
            if "manifest" in z.files:
                manifest = json.loads(str(z["manifest"]))
    return store, manifest


def _save(path, store, manifest):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp.npz"                                     # atomic: a crash mid-write cannot corrupt the npz
    np.savez_compressed(tmp, manifest=json.dumps(manifest), **store)
    os.replace(tmp, path)


_DM_KEYS = ("dm_vs_gpdhp_meandiff", "dm_vs_gpdhp_t", "dm_vs_gpdhp_p", "dm_vs_gpdhp_p_holm")


def fold_model(path, ds, variant, model, logscores, observed, meta=None, ref=GP_REF):
    """Add ONE model's per-obs logscores for (ds, variant) to the single npz at `path`, recompute
    DM-vs-`ref` + Holm over the cell's non-reference models present so far, and atomically rewrite.
    `ref` (GP-DHP(MAP)) must be folded FIRST so it is available as the DM reference; the reference
    model's own record carries no DM fields.  Idempotent per (ds, variant, model).  Returns the record."""
    logscores = np.asarray(logscores, float); observed = np.asarray(observed, float)
    store, manifest = _load(path)
    pre = f"{ds}/{variant}/"
    store[pre + "observed"] = observed
    store[pre + model + "/logscores"] = logscores
    manifest = [r for r in manifest
                if not (r["dataset"] == ds and r["variant"] == variant and r["model"] == model)]
    rec = dict(dataset=ds, variant=variant, model=model,
               pll=float(logscores.sum()), n_test=int(len(logscores)))
    if meta:
        rec.update(meta)
    manifest.append(rec)

    # recompute DM vs ref + Holm over ALL non-reference models in this cell that are present so far
    ref_key = pre + ref + "/logscores"
    cell = [r for r in manifest if r["dataset"] == ds and r["variant"] == variant and r["model"] != ref]
    for r in cell:                                             # clear stale DM before recompute
        for k in _DM_KEYS:
            r.pop(k, None)
    if ref_key in store:
        refls = np.asarray(store[ref_key], float)
        order, raw = [], {}
        for r in cell:
            ls = store.get(pre + r["model"] + "/logscores")
            if ls is not None and len(ls) == len(refls):
                raw[r["model"]] = dm_test(np.asarray(ls, float) - refls)   # >0 => model beats the reference
                order.append(r["model"])
        ph = holm([raw[m][2] for m in order])
        for j, m in enumerate(order):
            r = next(rr for rr in cell if rr["model"] == m)
            r["dm_vs_gpdhp_meandiff"], r["dm_vs_gpdhp_t"], r["dm_vs_gpdhp_p"] = raw[m]
            r["dm_vs_gpdhp_p_holm"] = float(ph[j])
    _save(path, store, manifest)
    return rec


def print_summary(path):
    """Print the manifest of a folded npz as a table (dataset/variant/model/pLL/DM t/p_holm)."""
    with np.load(path, allow_pickle=True) as z:
        man = json.loads(str(z["manifest"]))
    print(f"{'dataset':8} {'variant':7} {'model':24} {'pLL':>10} {'DM t':>7} {'p_holm':>8}")
    for r in man:
        dt = r.get("dm_vs_gpdhp_t"); ph = r.get("dm_vs_gpdhp_p_holm")
        print(f"{r['dataset']:8} {r['variant']:7} {r['model']:24} {r['pll']:10.1f} "
              f"{('' if dt is None else f'{dt:+.2f}'):>7} {('' if ph is None else f'{ph:.3f}'):>8}")
