# GP-DHP — Gaussian Process Discrete Hawkes Process

Reproduction code for the manuscript "A Semiparametric Discrete Hawkes Model with a Collapsed
Gaussian-Process Prior" by T. Brisley, G. Ross and D. Paulin available on <https://arxiv.org/abs/2509.21996>. GP-DHP is a semiparametric collapsed-latent
Gaussian-process discrete-time Hawkes model for self-exciting count data, with MAP estimation,
forward-validation hyperparameter selection by an analytic bilevel (implicit-function-theorem)
hypergradient, and a closed-form baseline/excitation projection. **Every GP-DHP fit reported in the
paper is selected under the stability constraint `R_+ <= 1 - 1e-4`** on the positive excitation mass
(augmented-Lagrangian forward validation, Appendix C of the paper). The whole pipeline is
**Python / JAX**.

---

## 1. Directory layout

```
papercode/
├── library/                    importable package -- the high-level GP-DHP code (import as `library.X`)
│   ├── gpdhp_cfv_grad.py            the production constrained (ALM) selector
│   ├── gpdhp_fit.py                 one-call constrained fit + predictive helpers (used by the runners)
│   ├── datasets.py                  the dataset split table (DATA) + load_split / load_split_cal
│   ├── _gpdhp_common.py, _gpdhp_fft.py   collapsed-latent model builder (design, link, NB2, FFT excitation)
│   └── gpdhp_select.py, gpdhp_fv_grad.py, jax_baselines.py, covariates.py, ...   (model + benchmark modules)
├── experiments/                runnable scripts reproducing the paper (thin wrappers over `library`)
│   ├── results_hawkes/         output npz/json (created/updated by the scripts below)
│   │   ├── nonneural_models.npz    all count-process benchmarks + the unconstrained GP-DHP anchor
│   │   │                            + the paper's exact constrained fits (folded in by supp_tables.py)
│   │   ├── neural_models.npz        the four neural NB benchmarks
│   │   ├── synthetic_{baseline,excitation}.npz   the Section 5.1 recovery-experiment outputs
│   │   ├── paper_tables.json        constrained GP-DHP + the paper's DM/Holm numbers (main predictive tables + neural supplement)
│   │   └── supp_tables.json         hypers/excitation/MAE-RMSE supplementary tables + the main-text calibration table
│   ├── run_nonneural.py, run_neural.py, paper_tables.py, supp_tables.py, pit_check.py, gpdhp_runner.py, ...
│   └── prepare_*.py            reproduce each data/<ds>/dataset_counts.csv from source (see §3)
├── checks/                     verification + micro-benchmarks (verify_*, fft_benchmark, timing_*, delta_sensitivity)
└── data/                  the five processed count series + external covariate files
```

The experiment/check scripts import the `library` package: each prepends `papercode/` to `sys.path`,
so run them from anywhere, e.g. `python experiments/paper_tables.py`. `HAWKES_ROOT` is auto-detected as
the `papercode/` directory from each script's location, so the code is location-independent as long as
`library/`, `experiments/`, `checks/`, and `data/` stay siblings under `papercode/`. To be
explicit, `export HAWKES_ROOT=/path/to/papercode`.

---

## 2. Environment

Python 3.14 with (versions this was run under):

| package | version | used for |
|---|---|---|
| jax (x64 enabled) | 0.10.0 | all models + selection |
| numpy | 2.4.4 | everywhere |
| scipy | — | NB pmf, optimisation, DM test |
| pandas | — | data loading / CSV I/O |
| optax | 0.2.8 | neural NB training |
| matplotlib | 3.10.8 | figures |

```bash
python -m venv venv && source venv/bin/activate
pip install "jax[cpu]" numpy scipy pandas optax matplotlib      # add a CUDA jax build for the neural grid
```
JAX x64 is enabled inside the code. The CPU scripts pin `JAX_PLATFORMS=cpu`; the neural grid
benefits from a GPU. All selections run under a fixed random seed (deterministic reproduction).

---

## 3. Reproduce all paper results

Run from anywhere (paths are self-locating). Steps (a)-(b) build the benchmark npz files; steps
(c)-(e) produce the paper's GP-DHP numbers and verification from them; steps (f)-(h) regenerate the
figures. The synthetic experiments (g) and the illustrative figures (h) are self-contained and need
none of the npz files or real datasets.

```bash
# (a) Count-process benchmarks -> experiments/results_hawkes/nonneural_models.npz
#     The 4 parametric DHPs, Baseline-only, Histogram DHP-NB, NB-INGARCH, Random-histogram DHP-NB
#     (Browning, reversible-jump MCMC), GP-Hawkes (discrete, squared-GP), and an UNCONSTRAINED
#     GP-DHP anchor. ~30-45 min (the Browning RJMCMC on the long daily series dominates).
python experiments/run_nonneural.py
#     faster variants:
python experiments/run_nonneural.py --no-browning          # skip the slow RJMCMC competitor
python experiments/run_nonneural.py --datasets dengue,nyc  # a subset of the five datasets

# (b) Neural benchmarks        -> experiments/results_hawkes/neural_models.npz   (GPU recommended)
python experiments/run_neural.py

# (c) THE PAPER'S GP-DHP: stability-constrained fits + DM/Holm tables
#     Re-fits all 7 cells under R_+ <= 1-1e-4 (augmented-Lagrangian forward validation, 2x budget,
#     20 starts, seed 0), asserts achieved R_+ < 1 per cell, and recomputes Diebold-Mariano + Holm
#     with the paper's family structure (9 count-process comparisons per column; 4 neural
#     comparisons as a separate family). Writes results_hawkes/paper_tables.json — the source of
#     truth for the predictive Tables 3-4 and Supplementary Table S3. Needs (a)+(b) present. ~40 min.
python experiments/paper_tables.py

# (d) Supplementary tables (selected hyperparameters, excitation summaries R_+/lags, MAE/RMSE)
#     plus the main-text calibration Table 5, all from the SAME constrained fits
#     -> results_hawkes/supp_tables.json; also folds the exact constrained fits (selected h, kappa,
#     link scale, covariate column scales, fitted theta*) into nonneural_models.npz under
#     '<ds>/<variant>/constrained_fit/*' keys for step (f). ~35 min.
python experiments/supp_tables.py

# (e) Hessian positive-definiteness + drift-rate verification (paper Appendix C):
#     smallest eigenvalue of the exact inner Hessian at the selected optimum AND the full-period
#     refit of every cell (all >= 1.0), plus the implied drift rate rho0 per fitted kernel. ~40 min.
python checks/verify_hessian_pd.py

# (f) Real-data figures        -> ../paper/{figure_kernel_small_multiples,
#                                  figure_nyc_component_decomposition,
#                                  figureS1_predictive_intervals_final}.pdf
#     Drawn from the CONSTRAINED fits folded into nonneural_models.npz by step (d) -- the same
#     fits as the tables; each reconstruction is verified in place against the stored pLL and R_+.
#     Needs (d). Seconds (no re-optimization).
python experiments/regen_paper_figures.py

# (g) Synthetic recovery experiments (Section 5.1) -> ../paper/{figure3,figure4}.pdf
#     Self-contained: simulate DHP-NB2 count data (generator per the paper's Section 5.1 spec) for
#     3 excitation kernels x 3 baseline regimes, T=6000, 10 replicates each, then fit GP-DHP by the
#     SAME bilevel forward-validation selection (gpdhp_select.select) and project the components.
#     figure3 = excitation-kernel recovery (NB / geometric / delayed-NB); figure4 = baseline recovery
#     (linear / seasonal / linear+seasonal) + fixed excitation. Aggregated npz ->
#     results_hawkes/synthetic_{excitation,baseline}.npz. ~30-45 min on a multicore CPU.
python experiments/synthetic_experiments.py both 10
#     quick pipeline check (1/6 the size, tiny budget, no figure written):
python experiments/synthetic_experiments.py excitation --smoke

# (h) Illustrative (conceptual) figures -> ../paper/{discrete_hawkes_intensity_mu05,
#                                            gp_realisations_beta{0.00,0.03,0.10},
#                                            Kf_heatmap_beta{0.00,0.03,0.10}}.png
#     Analytic diagrams with NO fitted-data dependence (parameters fixed in the manuscript captions):
#     a toy discrete-Hawkes intensity path, and a beta-sweep of prior draws + covariance heatmaps for
#     the warped-RBF GP lag kernel (the SAME kernel the model uses, mirrored in numpy). <5 s.
python experiments/illustration_figures.py
```

One-off diagnostics: `python experiments/run_gpdhp_constrained.py <delta> <budget-mult> <cells>` fits
the constrained GP-DHP for chosen cells (e.g. `1e-4 2 dengue,gva/cov`) and prints achieved `R_+`,
held-out pLL, and DM against the stored benchmarks — the driver behind the paper's protocol
(`delta=1e-4`, `2` = production budget: 600 outer / 1000 inner / 1600 refit iterations).
`python experiments/pit_check.py` recomputes the non-randomized PIT summary quoted in the paper's
calibration paragraph (§5.2): it reruns the 5 constrained selections deterministically and prints,
per cell, the 10-bin aggregated-PIT histogram and the sup-distance from the uniform CDF, and writes the 5-panel `../paper/figureS3_pit_histograms.pdf` (~10 min).
Further one-off checks behind specific paper claims: `timing_collapsed_vs_direct.py` (Table 1's
collapsed-vs-full-latent timing comparison; ~2 min, wall times machine-specific),
`verify_constrained_grad.py` (the Appendix C constrained-hypergradient verification vs
AD-through-unrolled-Newton, three regimes; ~2 min), `delta_sensitivity.py` (the §5.2 stability-margin
sweep, GVA+cov at delta=1e-5/1e-4/1e-3; ~6 min), and `fft_benchmark.py` (the Appendix C FFT-vs-dense
per-evaluation benchmark on the terrorism series at D=100/400; ~5 min).

Inspect a results file at any time:
```bash
cd papercode && python -c "from library import bench_shared as BS; BS.print_summary('experiments/results_hawkes/nonneural_models.npz')"
```

Datasets (keys → `data/` folder): `dengue`→`dengue_singapore`, `crypto`→`crypto_survstat_de`,
`rand`→`RAND_terrorism`, `nyc`→`nyc_shootings`, `gva`→`gva_allshootings`. NYC and GVA are additionally fit
`+cov` with the temperature-anomaly and holiday covariate block.

### Data sources (open data)

All five series are built from public sources; each `data/<dataset>/` ships the processed
`dataset_counts.csv`. Per-dataset access details, the exact aggregation, and the covariate construction
are in [`data/README_DATA_PROVENANCE.md`](data/README_DATA_PROVENANCE.md) (and
`data/crypto_survstat_de/PROVENANCE.md` for the SurvStat query).

| Series (`data/`) | Source | URL |
|---|---|---|
| `dengue_singapore` | MOH Singapore Weekly Infectious Disease Bulletin, via data.gov.sg (`d_ca168b2cb763640d72c4600a68f9909e`) | https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view |
| `crypto_survstat_de` | Robert Koch Institute — SurvStat@RKI 2.0 | https://survstat.rki.de |
| `nyc_shootings` | NYPD *Shootings (2006–Present)*, NYC Open Data | https://data.cityofnewyork.us/Public-Safety/Shootings-2006-Present-/5ucz-vwe8 |
| `gva_allshootings` | Gun Violence Archive *All Shootings*, figshare `10.6084/m9.figshare.25517224.v2` | https://figshare.com/articles/dataset/Gun_Violence_-_All_Shootings/25517224 |
| `RAND_terrorism` | RAND Database of Worldwide Terrorism Incidents (RDWTI) | https://www.rand.org/nsrd/projects/terrorism-incidents.html |
| covariate: temperature | NOAA GHCN-Daily TMAX (NYC = Central Park `USW00094728`; GVA = metro composite) | https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily |

**Processing.** All series use the same recipe: count events per period on a gap-free grid (weekly for
dengue/crypto, daily for nyc/gva/rand). Each `data/<dataset>/` ships only the **derived counts**;
no raw event/notification records are redistributed (each source's terms govern its raw data — the RAND
RDWTI agreement in particular prohibits redistributing it, and it must be cited as *"RAND Database of
Worldwide Terrorism Incidents", https://smapp.rand.org/rwtid*).

Every series has a runnable `experiments/prepare_*.py` that re-derives the counts from source and
**verifies** them against the shipped `dataset_counts.csv`:

| Script | Source access | Verification |
|---|---|---|
| `prepare_dengue_counts.py` | fetches the data.gov.sg API | exact (574 weeks, 164,447 cases) |
| `prepare_nyc_counts.py` | fetches NYC Open Data (Socrata) | exact (7,305 days, 23,988 incidents) |
| `prepare_gva_counts.py` | downloads the figshare CSV (~72 MB) | exact (3,652 days, 390,012 incidents) |
| `prepare_terrorism_counts.py --raw …` | user-downloaded RDWTI file (not redistributable) | exact (40,129 incidents → 15,302 days) |
| `prepare_crypto_counts.py` | SurvStat@RKI export ZIP (bundled `survstat.zip`) | exact from the original export (990 weeks, 26,024 cases) |
| `prepare_nyc_temperature.py` | fetches NOAA GHCN Central Park | `tmax_c` exact; anomaly ≈0.14 °C RMSE |

crypto differs from the event datasets. Its SurvStat@RKI series comes from an interactive query builder
(no public API), so the export ZIP is downloaded by hand — the original 2026-07-13 export is bundled as
`data/crypto_survstat_de/survstat.zip`, and `prepare_crypto_counts.py` melts it (each week-53 at its
own ISO date) to the weekly series and **verifies a byte-identical match** to the shipped
`dataset_counts.csv`. A *fresh* SurvStat export will not be byte-identical because RKI revises historical
counts (~0.996 week-to-week correlation), so the shipped counts remain the canonical modelling input. The
exact query — and a note on a week-53 melt bug corrected on 2026-07-23 — are in the script docstring and
`data/crypto_survstat_de/PROVENANCE.md`.

The holiday covariates are computed deterministically in code (`library/calendar_utils.holiday_indicators`);
the temperature anomaly (daily TMAX minus a smooth annual harmonic climatology, z-scored on the
development period) is assembled by `library/covariates.py` from the shipped `temp_anomaly.csv` (the
GVA anomaly is a population-weighted metro composite, documented in the provenance file).

---

## 4. Module map

The map is grouped by the three directories: `library/` (importable modules), `experiments/` (runnable
scripts), and `checks/` (verification + micro-benchmarks).

### `library/` — importable package (`import library.X`)

*GP-DHP core*
- `datasets.py` — the dataset split table (`DATA`) + `load_split` / `load_split_cal`; the single data-loading source shared by every experiment.
- `gpdhp_fit.py` — high-level `fit_constrained_cell` (ALM selection + constrained refit) and `constrained_predictive` (θ\* / λ derivation); the experiment scripts `paper_tables` / `supp_tables` / `pit_check` / `run_gpdhp_constrained` are thin wrappers over these.
- `_gpdhp_common.py` — collapsed-latent design, softplus link, NB2 likelihood, component projection.
- `_gpdhp_fft.py` — FFT-accelerated excitation, O(T log T + D²), the **default** everywhere (`use_fft=True`); matches the dense path to machine precision; break-even at D=100, ≈6× faster per bilevel solve at D=400.
- `gpdhp_select.py` — unconstrained forward-validation selection (multi-start L-BFGS-MT); supplies the shared hyper-box/names.
- `gpdhp_fv_grad.py` — the unconstrained analytic FV bilevel hypergradient (implicit-function-theorem, θ-coordinates).
- `gpdhp_cfv_grad.py` — **the production selector**: augmented-Lagrangian inner MAP under `R_+ <= 1-δ` (ρ-ramp 30→3·10⁵, multiplier updates), finite-ρ Sherman–Morrison constrained hypergradient (two H-solves), penalized outer objective (`pen_mu=1e5`) that steers the multi-start out of the infeasible region; `select_constrained[_cov]`, `constrained_refit_score[_cov]`.
- `bench_hawkes.py` — MAP/scoring utilities used by the synthetic experiments.

*Count-process benchmark models*
- `jax_baselines.py` — parametric DHP family, Histogram DHP-NB, Baseline-only, NB-INGARCH (all FV-selected).
- `browning_rjmcmc.py` — random-histogram DHP-NB (Browning 2022), reversible-jump MCMC.
- `gphawkes_discrete.py` — discrete GP-modulated Hawkes (squared-GP kernel; Zhou 2020 / Zhang 2020), softplus link; hyperparameters selected by the **same** bilevel forward-validation protocol as GP-DHP (20-start L-BFGS-MT on the analytic hypergradient), GP coefficients by MAP.
- `lbfgs_mt.py` — Moré–Thuente bound-constrained L-BFGS.

*Neural benchmark models + shared utilities*
- `neural_nb_jax.py` — MLP/GRU/LSTM/DeepAR NB model definitions.
- `bench_shared.py` — the single-npz folding + `dm_test` (Diebold–Mariano, Newey–West + Harvey–Leybourne–Newbold) + Holm.
- `covariates.py`, `calendar_utils.py` — covariate block construction (temperature anomaly + in-code holiday indicators).

### `experiments/` — runnable scripts (thin wrappers over `library`)

*Orchestrators / paper drivers*
- `run_nonneural.py` — every count-process benchmark + the unconstrained GP-DHP anchor, per dataset/variant (folds each model incrementally; invokes `run_gphawkes.py` + `run_browning.py` for the two Hawkes competitors).
- `run_neural.py` — the four neural NB models (subprocesses `neural_grid_jax.py`).
- `gpdhp_runner.py` — one-cell unconstrained GP-DHP MAP runner (the npz anchor).
- `jax_baselines_runner.py` — drives the `library/jax_baselines.py` count-process baselines per (dataset, variant).
- `neural_grid_jax.py` — self-contained neural selector: Lion training + 40% forward-validation + the 144-point grid (the sole neural selector).
- `run_gpdhp_constrained.py` — constrained fit + DM for chosen cells (`<delta> <mult> <cells>` CLI).
- `run_browning.py`, `run_gphawkes.py` — fit + fold one competitor for chosen `--cells ds/variant` (also invoked by `run_nonneural.py`).
- `paper_tables.py` — **the paper's GP-DHP**: constrained refits of all 7 cells + DM/Holm with the paper's exact family structure → `paper_tables.json`.
- `supp_tables.py` — the supplementary tables + the main-text calibration table from the same constrained fits → `supp_tables.json`; also folds the exact fits into `nonneural_models.npz` (`constrained_fit/*` keys, consumed by `regen_paper_figures.py`).
- `pit_check.py` — non-randomized PIT (Czado–Gneiting–Held 2009) for the 7 constrained paper fits: aggregated conditional-CDF construction, 10-bin histogram + sup-distance from the uniform CDF; backs the calibration PIT sentence and Supplementary PIT figure (§5.2).
- `regen_paper_figures.py` — rebuild the three real-data figures from the constrained fits stored in `nonneural_models.npz` (run `supp_tables.py` first); asserts each reconstructed fit reproduces the stored held-out pLL and `R_+`, so the figures provably match the tables.
- `synthetic_experiments.py` — Section 5.1 recovery experiments (JAX). The DHP-NB2 *generator* implements the manuscript's Section 5.1 specification (three excitation kernels — NB, geometric, and the delayed negative-binomial with K_NB=0.8, mean lag 8, size 25 — and three baseline regimes); the fit reuses the production GP-DHP code unchanged. Regenerates `figure3.pdf` (excitation recovery) and `figure4.pdf` (baseline recovery).
- `illustration_figures.py` — the seven conceptual/illustrative figures (Section 2 toy discrete-Hawkes intensity path + the Appendix prior-illustration β-sweep of GP draws and `K_f` covariance heatmaps). Pure analytic diagrams with **no** data or npz dependence; the GP kernel is the warped-RBF `warped_rbf_cholesky` re-expressed in numpy, so the illustration and the fitted model share one kernel definition.

*Data preparation — reproduce each `data/<dataset>/dataset_counts.csv` from source (see §3)*
- `prepare_terrorism_counts.py` — reproduces the RAND worldwide-terrorism series from a *user-downloaded* RDWTI incident file (`--raw`; the raw file is not redistributed, per RDWTI terms) and verifies an exact match.
- `prepare_dengue_counts.py`, `prepare_nyc_counts.py`, `prepare_gva_counts.py` — fetch each openly-redistributable series from source (data.gov.sg API / NYC Open Data Socrata / figshare), re-derive the counts, and verify an exact match against the shipped file.
- `prepare_crypto_counts.py` — melts the SurvStat@RKI export ZIP (interactive query, no API; the original is bundled as `data/crypto_survstat_de/survstat.zip`) into the weekly series and **verifies a byte-identical match** to the shipped file (990 weeks); a fresh export drifts by RKI revisions (~0.996 corr).
- `prepare_nyc_temperature.py` — fetches NOAA GHCN Central Park, verifies the shipped `tmax_c` exactly, and recomputes the day-of-year harmonic anomaly (≈0.14 °C RMSE; the shipped `temp_anomaly.csv` is the canonical covariate input).

### `checks/` — verification + micro-benchmarks
- `verify_hessian_pd.py` — smallest eigenvalues of the exact inner Hessians (selection optimum + refit, all cells) and infimal drift rates ϱ₀; backs the local-regularity verification quoted in paper Appendix C.
- `verify_constrained_grad.py` — verifies the constrained (ALM) bilevel hypergradient against AD through an unrolled Newton solve of the final augmented subproblem at a fixed active set, in slack/binding/infeasible regimes; backs the Appendix C verification.
- `timing_collapsed_vs_direct.py` — the Table 1 experiment: collapsed vs direct full-latent optimization at fixed hyperparameters on the §5.1.3 synthetic DGP; reports median times, speedup, and machine-precision agreement.
- `fft_benchmark.py` — the Appendix C FFT-vs-dense benchmark: one bilevel evaluation on the terrorism series at D_max∈{100,400}, dense vs FFT, with an exactness check.
- `delta_sensitivity.py` — the §5.2 stability-margin sweep: full constrained selection+refit on GVA+cov at δ∈{1e-5,1e-4,1e-3}, reporting the held-out-pLL spread.

---

## 5. Results schema

`nonneural_models.npz` / `neural_models.npz`: keys are `"{dataset}/{variant}/{model}/logscores"`
(per-observation held-out NB log-scores) plus `"{dataset}/{variant}/observed"`; a JSON `"manifest"`
lists every record with its total `pll` and DM statistics against the npz's internal GP-DHP anchor.
Note the npz anchor is the **unconstrained** GP-DHP: **the paper's GP-DHP numbers and adjusted
p-values are those in `paper_tables.json`** (constrained fits, m=9 count-process comparisons per
column), with per-cell achieved `R_+` and margins; `supp_tables.json` carries the supplementary-table
numbers from the same fits.

`nonneural_models.npz` additionally carries, per paper cell, `"{dataset}/{variant}/constrained_fit/..."`
keys (from step (d)): the exact constrained fit — selected hyperparameters `h` (JSON), `kappa`,
`link_scale`, covariate `col_sigmas`, the fitted coefficient `theta`, and the achieved `R_plus`/`pll`
that `regen_paper_figures.py` re-verifies before drawing the figures.

`synthetic_{excitation,baseline}.npz` (from step (g)) each hold a `store` object keyed by scenario,
with the per-replicate fitted excitation kernels (`fhat`, reps × D) and baselines (`bhat`, reps × T)
alongside the ground-truth `f_true`/`b_true`; the figures plot the replicate mean and central-90%
band against the truth.
