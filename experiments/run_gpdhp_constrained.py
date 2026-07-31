"""ALM stability-constrained GP-DHP + DM, for any cell(s).   usage: alm_cells.py <delta> <mult> <cells>

Settled protocol (from the gva/cov tuning thread): delta=1e-4, 2x budget (outer=600, inner=1000,
refit=1600), 20 starts.  4x is byte-identical to 2x at delta=1e-4 => 2x is converged.
For each cell: constrained FV selection -> full-period constrained MAP refit -> per-obs held-out NB
log-scores -> DM (bench_shared, Newey-West HAC + HLN) + Holm vs the 13 competitors.
Cost is quoted against the STORED production GP-DHP (MAP) -- the paper's own unconstrained number.
NOTE the ALM satisfies the cap only ASYMPTOTICALLY in rho, so the achieved R_+ can sit a hair above
the requested budget; what matters for the guarantee is R_+ < 1, which is ASSERTED per cell.
"""
import os, sys, time
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

DELTA = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
MULT = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
CELLS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["dengue", "crypto"])
NST = 20; SEED = 0; ND = 10
MAXIT = int(300 * MULT); INNER = int(500 * MULT); REFIT = int(800 * MULT)
RES = os.path.join(HAWKES, "experiments", "results_hawkes")
znn = np.load(os.path.join(RES, "nonneural_models.npz"), allow_pickle=True)
znr = np.load(os.path.join(RES, "neural_models.npz"), allow_pickle=True)

print(f"ALM constrained GP-DHP + DM | delta={DELTA:g} (R_+<={1-DELTA:.7f}) budget x{MULT:g} "
      f"(outer={MAXIT} inner={INNER} refit={REFIT}) {NST} starts\n", flush=True)

for cell in CELLS:
    t0 = time.time()
    ds = cell.split("/")[0]; cov = cell.endswith("/cov")
    tag = f"{ds}/{'cov' if cov else 'nocov'}"
    res = gpdhp_fit.fit_constrained_cell(ds, cov, delta=DELTA, mult=MULT, n_starts=NST,
                                         seed=SEED, n_devices=ND, root=HAWKES, verbose=True)
    ls, Rp, sel = res["ls"], res["Rp"], res["sel"]
    y_test = res["y_test"]

    # ---- the guarantee that actually matters ------------------------------------------------- #
    assert Rp < 1.0, f"{tag}: FITTED KERNEL IS NOT SUBCRITICAL (R_+={Rp!r})"
    obs = np.asarray(znn[f"{tag}/observed"], float)
    assert np.allclose(np.asarray(y_test, float), obs), f"{tag}: test counts misaligned vs stored"
    stored = np.asarray(znn[f"{tag}/GP-DHP (MAP)/logscores"], float)   # paper's unconstrained MAP

    print(f"\n=== {tag} | R_+={Rp:.7f} (<1 OK, margin {1-Rp:.2e}; budget {1-DELTA:.7f}, "
          f"within_budget={Rp <= 1-DELTA+1e-9}) ===")
    kb = float(sel["x"]["K_NB"]); cap_k = 1.0 - DELTA
    print(f"    ALM pLL={ls.sum():.2f}   stored GP-DHP(MAP) pLL={stored.sum():.2f}   "
          f"cost={ls.sum()-stored.sum():+.2f}   (val_pLL={sel['val_pll']:.3f}, "
          f"K_NB={kb:.4f}{' [AT CAP]' if kb > cap_k - 1e-3 else ''}, "
          f"sigma_u={sel['h']['sigma_u']:.2e}, Kf={sel['K_fourier']}, {time.time()-t0:.0f}s)", flush=True)

    comp = {}
    for z in (znn, znr):
        for k in z.files:
            if k.startswith(f"{tag}/") and k.endswith("/logscores"):
                nm = k[len(f"{tag}/"):-len("/logscores")]
                if not nm.startswith("GP-DHP") and nm not in comp:
                    comp[nm] = np.asarray(z[k], float)
    fam = list(comp); rows, pv = [], []
    for nm in fam:
        dbar, dm, p = BS.dm_test(ls - comp[nm])
        rows.append([nm, comp[nm].sum(), dbar, dm, p]); pv.append(p)
    padj = BS.holm(pv); order = np.argsort([r[1] for r in rows])[::-1]
    print(f"    {'competitor':26} {'pLL':>10} {'meanΔ':>8} {'DM t':>7} {'p_holm':>9}  sig")
    nsig = 0
    for i in order:
        nm, pll, dbar, dm, p = rows[i]; pa = padj[i]
        sig = ("***" if pa < 1e-3 else "**" if pa < 1e-2 else "*" if pa < 5e-2 else "ns")
        if dbar > 0 and pa < 5e-2: nsig += 1
        mark = sig if dbar > 0 else f"({sig}, GP-DHP WORSE)"
        print(f"    {nm:26} {pll:10.2f} {dbar:+8.4f} {dm:7.2f} {pa:9.2e}  {mark}")
    print(f"    --> constrained GP-DHP significantly better than {nsig}/{len(fam)} competitors", flush=True)
print("\ndone_marker")
