# data provenance and processing

Every series here is built from a **public, open** source by the same recipe: aggregate events to
counts on a fixed grid (weekly for the disease series, daily for the event series), then place them
on a gap-free time grid (missing periods = 0). The entries below give the exact source, a precise URL,
and the processing for each series and covariate. Column meaning in every `dataset_counts.csv`:
`date`, `count` (the modelled count), `period` (52 weekly / 365 daily), and the chronological
`split`/`final_split`/`neural_split` labels applied downstream by `library/datasets.py::load_split`.

Each `<dataset>/dataset_counts.csv` holds only the **derived counts**; no raw event/notification
records are redistributed here (each source's terms govern its raw data, and the RAND RDWTI agreement
in particular prohibits redistributing its data). To rebuild a series, download the raw records from
the source portal linked below and apply the aggregation described. For RAND this recipe is runnable,
verified code: `experiments/prepare_terrorism_counts.py` reconstructs the shipped counts from a
user-downloaded RDWTI file and checks the match exactly.

---

## Count series

### `dengue_singapore` — Singapore dengue (weekly)
- **Source:** Ministry of Health, Singapore — Weekly Infectious Disease Bulletin (dengue case counts),
  distributed through **data.gov.sg**, dataset `d_ca168b2cb763640d72c4600a68f9909e`.
- **URL:** https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view
  (API: `https://data.gov.sg/api/action/datastore_search?resource_id=d_ca168b2cb763640d72c4600a68f9909e`)
- **Processing:** the portal already reports weekly (epidemiological-week) dengue case counts; we take
  the consecutive weekly series **2012-01-01 … 2022-12-25** (574 weeks, `period=52`).
- **Reproduce:** `python experiments/prepare_dengue_counts.py` fetches the data.gov.sg API, rebuilds these counts, and verifies an exact match to the shipped file.
- **Split:** development 2012–2017 (314 wks), test 2018–2022 (260 wks).

### `crypto_survstat_de` — German cryptosporidiosis (weekly)
- **Source:** Robert Koch Institute — **SurvStat@RKI 2.0** (notified national cryptosporidiosis cases).
- **URL:** https://survstat.rki.de  (interactive query builder; the exact query filters — reference
  definition = Yes, disease = Cryptosporidiosis, rows = week of notification, columns = year,
  zero values on — are recorded in `crypto_survstat_de/PROVENANCE.md`).
- **Processing:** the ZIP export is a week × year matrix (UTF-16 TSV); it is melted to a continuous
  chronological weekly series (each week dated at its ISO-week Monday) and restricted to the pre-pandemic
  window (through 2019) — **990 weeks, 26,024 cases**. See `crypto_survstat_de/PROVENANCE.md` for the full
  extraction record, a same-source consistency check, and a note on a week-53 melt bug corrected 2026-07-23.
- **Reproduce:** `python experiments/prepare_crypto_counts.py` melts the bundled `survstat.zip` and
  verifies a byte-identical match to the shipped file.
- **Split:** development 2001–2014 (730 wks), test 2015–2019 (260 wks).

### `nyc_shootings` — New York City shootings (daily)
- **Source:** New York City Police Department — *Shootings (2006–Present)*, via **NYC Open Data**.
- **URL:** https://data.cityofnewyork.us/Public-Safety/Shootings-2006-Present-/5ucz-vwe8
- **Processing:** one row per shooting incident → daily counts of **distinct `INCIDENT_KEY` per
  occurrence date** (`count_method` column: "Distinct incident_key per occurrence date"), placed on a
  gap-free daily grid (`period=365`). Paired with the covariate block below.
- **Reproduce:** `python experiments/prepare_nyc_counts.py` fetches NYC Open Data (Socrata), rebuilds these counts, and verifies an exact match to the shipped file.
- **Split:** chronological `final_split` (development, then a held-out final test block).

### `gva_allshootings` — Gun Violence Archive shootings (daily)
- **Source:** Gun Violence Archive, via the figshare dataset *Gun Violence — All Shootings*.
- **URL:** https://figshare.com/articles/dataset/Gun_Violence_-_All_Shootings/25517224
  (DOI `10.6084/m9.figshare.25517224.v2`; GVA methodology: https://www.gunviolencearchive.org/methodology
  — archived at https://web.archive.org/web/20260331020805/https://www.gunviolencearchive.org/methodology)
- **Processing:** incident rows (2014–2023) → daily counts on a gap-free daily grid (`period=365`).
  Paired with the covariate block below.
- **Reproduce:** `python experiments/prepare_gva_counts.py` downloads the figshare CSV, rebuilds these counts, and verifies an exact match to the shipped file.
- **Split:** chronological `final_split` (development, then a held-out final test block).

### `RAND_terrorism` — Worldwide terrorism incidents (daily)
- **Source:** RAND Corporation — **RAND Database of Worldwide Terrorism Incidents (RDWTI)**,
  worldwide incidents 1968–2009.
- **URL:** https://www.rand.org/nsrd/projects/terrorism-incidents.html
- **Required citation:** "RAND Database of Worldwide Terrorism Incidents", https://smapp.rand.org/rwtid
- **Raw file NOT redistributed:** the RDWTI user agreement prohibits redistributing its data, so the
  raw incident file (`Date, City, Country, Perpetrator, Weapon, Injuries, Fatalities, Description`;
  40,129 rows) is **not** included. Download it yourself from the URL above for your own use.
- **Processing (runnable in code):** with the raw file downloaded,
  `python experiments/prepare_terrorism_counts.py --raw /path/to/incidents.csv` parses the incident
  dates, counts incidents per occurrence date, and reindexes onto a gap-free daily grid
  (1968-02-09 … 2009-12-31, 15,302 days), then **verifies** it equals the shipped derived
  `dataset_counts.csv` exactly (0 differing days, 40,129 incidents). Only the derived counts are shipped.
- **Split:** development 1968–2004, test 2005–2009 (final 5 years).

---

## External covariates (daily shooting series only)

Built by `library/covariates.py` into a 10-column block: one temperature anomaly + nine holiday
indicators. Both are identical across every model (GP-DHP, the count-process baselines, and the
neural models), so the comparison is on the same information.

- **Temperature anomaly** — `temp_anomaly.csv` (`date, [tmax_c,] anomaly, anomaly_lag1`).
  - *Source:* NOAA GHCN-Daily maximum temperature (TMAX). NYC uses the Central Park station
    (GHCN id `USW00094728`); GVA uses a population-weighted composite of U.S. metro stations.
    GHCN-Daily: https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
  - *Processing:* `anomaly` = daily TMAX minus a smooth annual harmonic (day-of-year) climatology,
    isolating unusually warm/cool days from the ordinary seasonal cycle; `covariates.py` then
    z-scores it on the **development** period only. The NYC file also ships the raw `tmax_c`, from
    which the anomaly is recomputable to ≈0.14 °C RMSE with a 3-harmonic climatology.
  - *Reproduce:* `python experiments/prepare_nyc_temperature.py` fetches GHCN Central Park, verifies the shipped `tmax_c` exactly, and reports the anomaly-recompute RMSE. GVA's population-weighted composite is not reproduced by this single-station script.
- **Holiday indicators** — computed **deterministically in code** from the dates by
  `library/calendar_utils.py::holiday_indicators` (no data file): nine U.S. holidays — New Year,
  Juneteenth, Independence Day, Halloween, Christmas, New Year's Eve, Memorial Day, Labor Day,
  Thanksgiving.

The weekly disease series use no external covariates.
