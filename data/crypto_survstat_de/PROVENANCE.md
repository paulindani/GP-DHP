# crypto_survstat_de provenance

German **national weekly cryptosporidiosis** (Kryptosporidiose) case counts, from the Robert Koch
Institute surveillance portal **SurvStat@RKI 2.0** (https://survstat.rki.de). Downloaded via the
interactive query builder on **2026-07-13** (Data Status: Current, 2026-07-12); the export ZIP and its
query record are bundled here as `survstat.zip` (Data.csv + Info.pdf).

Query (recorded in Info.pdf):
- Notification = via local and state health department
- Filters: Reference definition = Yes; Disease/Pathogen = Cryptosporidiosis
- Rows = Week of notification; Columns = Year of notification; Zero values = Yes; Totals = No

`Data.csv` is a UTF-16, tab-separated week (01..53) x year (2001..2026) matrix; a blank cell = that
ISO week is absent in that year. This is the same national reporting source as the shorter series used
in earlier work — a genuine same-source extension, not a substitute.

## Shipped series
`dataset_counts.csv`: the **pre-COVID** window **2001-01-01 .. 2019-12-23**, melted to a chronological
weekly series (one row per non-blank cell, dated at the Monday of its ISO week). **990 weeks, 26,024
cases**, `period = 52`, `D_max = 100`. The pre-pandemic cut avoids the COVID-era reporting disruption.

Split (year-based, applied in code by `library/datasets.py::load_split`, which reads the year, not the
`split` columns): **development 2001-2014 (730 weeks)**, **test 2015-2019 (260 weeks)**. The
`split`/`final_split`/`neural_split` columns in the CSV are set to this same year-based split.

## Reproduce
`python experiments/prepare_crypto_counts.py` melts the bundled `survstat.zip` and **verifies a
byte-identical match** to `dataset_counts.csv` (990 weeks, 26,024 cases). Each week-53 is dated at its
own ISO Monday. A *fresh* SurvStat export is NOT byte-identical: RKI revises historical counts, so a
later pull differs from the 2026-07-13 original (~0.996 week-to-week correlation) — the shipped file is
the canonical modelling input. Analyses using SurvStat@RKI 2.0 may be published with the citation in
Info.pdf.

## Week-53 correction (2026-07-23)
An earlier build of this file had a week-53 melt bug: it collected the non-blank week-53 cells — which
occur only in the ISO-53-week years (2004, 2015, 2020, with 2, 8, 5 cases) — and mis-appended them, in
file order, to the *first* year columns (2001, 2002, 2003) as duplicate-dated rows, dropping the true
2004/2015 week-53 dates and leaking in a 2020 (post-window) count. That misattributed 15 cases across
years (991 weeks / 26,029 cases). It has been corrected — each week-53 now sits at its own ISO date —
and the dataset, results npz, and tables regenerated. The pre-correction files are archived under
`archive/crypto_week53fix_2026-07-23/`.
