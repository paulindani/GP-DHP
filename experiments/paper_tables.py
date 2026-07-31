"""Source of truth for the REVISED paper tables: constrained GP-DHP as the reference model.

Re-fits the 7 stability-constrained cells (ALM, delta=1e-4, 2x budget, 20 starts, seed 0 -- the
settled protocol), asserts achieved R_+ < 1 on each, then recomputes Diebold-Mariano + Holm with
the paper's EXACT family structure, which differs from the diagnostic runs:
  * main tables (Tables 5/6): family = the 9 non-neural benchmarks.  GP-DHP (Bayes) is REMOVED from
    the paper, so it leaves the Holm family too (m: 10 -> 9), which changes every adjusted p.
  * neural supplement (Table S6): family = the 4 neural predictors, Holm-adjusted separately (m=4).
Prints table-ready pLL (1 dp) and significance superscripts, and saves per-obs logscores + a JSON
summary so the numbers written into the .tex are reproducible and auditable.
"""
import os, sys, json, time
os.environ["JAX_PLATFORMS"] = "cpu"; os.environ["OMP_NUM_THREADS"] = "1"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false"
import numpy as np, jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
HAWKES = os.environ.get("HAWKES_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ["HAWKES_ROOT"] = HAWKES
sys.path.insert(0, HAWKES)
from library import datasets
from library import covariates as CV
from library import gpdhp_cfv_grad as CFV
from library import gpdhp_fit
from library import bench_shared as BS

DELTA = 1e-4; MULT = 2.0; NST = 20; SEED = 0; ND = 10
MAXIT = int(300 * MULT); INNER = int(500 * MULT); REFIT = int(800 * MULT)
RES = os.path.join(HAWKES, "experiments", "results_hawkes")
OUT = os.path.join(HAWKES, "experiments", "results_hawkes")

NONNEURAL = ["Baseline only", "Discrete DHP", "Linear DHP", "Sinusoidal DHP",
             "Linear + Sinusoidal DHP", "Histogram DHP-NB", "Random-histogram DHP-NB",
             "GP-Hawkes (discrete)", "NB-INGARCH"]                     # 9 -> main-table Holm family
NEURAL = ["MLP-NB", "GRU-NB", "LSTM-NB", "DeepAR-NB"]                  # 4 -> supplement Holm family
CELLS = ["dengue", "crypto", "rand", "nyc", "nyc/cov", "gva", "gva/cov"]

znn = np.load(os.path.join(RES, "nonneural_models.npz"), allow_pickle=True)
znr = np.load(os.path.join(RES, "neural_models.npz"), allow_pickle=True)


def stars(p):
    return "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""


def fit(cell):
    ds = cell.split("/")[0]; cov = cell.endswith("/cov")
    res = gpdhp_fit.fit_constrained_cell(ds, cov, delta=DELTA, mult=MULT, n_starts=NST,
                                         seed=SEED, n_devices=ND, root=HAWKES)
    return res["ls"], res["Rp"], res["sel"]


summary = {}
print(f"CONSTRAINED GP-DHP tables | delta={DELTA:g} (R_+<={1-DELTA:.4f}) budget x{MULT:g} "
      f"{NST} starts seed {SEED}\n")
for cell in CELLS:
    t0 = time.time()
    ds = cell.split("/")[0]; tag = f"{ds}/{'cov' if cell.endswith('/cov') else 'nocov'}"
    ls, Rp, sel = fit(cell)
    assert Rp < 1.0, f"{tag}: NOT SUBCRITICAL (R_+={Rp!r})"
    obs = np.asarray(znn[f"{tag}/observed"], float)
    assert np.allclose(np.asarray(datasets.load_split(ds)[1], float), obs), f"{tag}: obs misaligned"
    stored = float(np.asarray(znn[f"{tag}/GP-DHP (MAP)/logscores"], float).sum())

    rec = {"tag": tag, "R_plus": Rp, "margin": 1 - Rp, "pLL": float(ls.sum()),
           "stored_unconstrained_pLL": stored, "cost_vs_unconstrained": float(ls.sum()) - stored,
           "K_NB": float(sel["x"]["K_NB"]), "sigma_u": float(sel["h"]["sigma_u"]),
           "kappa": float(sel["kappa"]), "link_scale": float(sel["link_scale"]),
           "K_fourier": int(sel["K_fourier"]), "val_pll": float(sel["val_pll"]),
           "gamma": float(sel["gamma"]), "pen_viol": float(sel["pen_viol"]), "families": {}}

    for fam_name, names, z in (("nonneural", NONNEURAL, znn), ("neural", NEURAL, znr)):
        comp, pv = {}, []
        for nm in names:
            key = f"{tag}/{nm}/logscores"
            if key not in z.files:
                continue
            c = np.asarray(z[key], float)
            dbar, dm, p = BS.dm_test(ls - c)
            comp[nm] = [float(c.sum()), float(dbar), float(dm), float(p)]; pv.append(p)
        padj = BS.holm(pv)
        for (nm, v), pa in zip(comp.items(), padj):
            v.append(float(pa)); v.append(stars(pa) if v[1] > 0 else f"WORSE:{stars(pa)}")
        rec["families"][fam_name] = {"m": len(comp), "rows": comp}
    summary[cell] = rec

    print(f"=== {tag} | R_+={Rp:.7f} (<1, margin {1-Rp:.2e}) | pLL={ls.sum():.1f} | "
          f"cost vs uncon {ls.sum()-stored:+.2f} | K_NB={rec['K_NB']:.3f} sigma_u={rec['sigma_u']:.2e} "
          f"Kf={rec['K_fourier']} | {time.time()-t0:.0f}s")
    for fam_name in ("nonneural", "neural"):
        f = rec["families"][fam_name]
        print(f"   [{fam_name}, Holm m={f['m']}]")
        for nm, v in f["rows"].items():
            print(f"      {nm:26} pLL={v[0]:9.1f}  meanD={v[1]:+.4f}  t={v[2]:6.2f}  "
                  f"p_holm={v[4]:.3e}  {v[5] or 'ns'}")
    print(flush=True)

with open(os.path.join(OUT, "paper_tables.json"), "w") as fh:
    json.dump(summary, fh, indent=1)

# ---- table-ready blocks ----------------------------------------------------------------------- #
def cellval(cell, nm, fam="nonneural"):
    r = summary[cell]["families"][fam]["rows"]
    if nm not in r: return "  ---   "
    pll, _, _, _, pa, st = r[nm]
    return f"${pll:.1f}^{{{st}}}$" if st else f"${pll:.1f}$"

print("\n\n########## TABLE 5 (nocov: dengue, crypto, rand) ##########")
print(f"\\textbf{{GP-DHP}} & \\textbf{{{summary['dengue']['pLL']:.1f}}} & "
      f"\\textbf{{{summary['crypto']['pLL']:.1f}}} & \\textbf{{{summary['rand']['pLL']:.1f}}} \\\\")
for nm in NONNEURAL:
    print(f"{nm:24}& {cellval('dengue',nm)} & {cellval('crypto',nm)} & {cellval('rand',nm)} \\\\")

print("\n########## TABLE 6 (cov: nyc no/with, gva no/with) ##########")
print(f"\\textbf{{GP-DHP}} & \\textbf{{{summary['nyc']['pLL']:.1f}}} & \\textbf{{{summary['nyc/cov']['pLL']:.1f}}} & "
      f"\\textbf{{{summary['gva']['pLL']:.1f}}} & \\textbf{{{summary['gva/cov']['pLL']:.1f}}} \\\\")
for nm in NONNEURAL:
    print(f"{nm:24}& {cellval('nyc',nm)} & {cellval('nyc/cov',nm)} & {cellval('gva',nm)} & {cellval('gva/cov',nm)} \\\\")

print("\n########## NEURAL SUPPLEMENT (Holm m=4 per cell) ##########")
for nm in NEURAL:
    row = " & ".join(cellval(c, nm, "neural") for c in ["dengue", "crypto", "rand", "nyc/cov", "gva/cov"])
    print(f"{nm:12}& {row} \\\\")

print("\n########## R_+ / margin per cell ##########")
for c in CELLS:
    r = summary[c]
    print(f"{r['tag']:12} R_+={r['R_plus']:.7f}  margin={r['margin']:.2e}  cost={r['cost_vs_unconstrained']:+.2f}")
print("\ndone_marker")
