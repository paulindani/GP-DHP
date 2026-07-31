"""Rebuild the German cryptosporidiosis weekly count series from a SurvStat@RKI export.

Source : Robert Koch Institute -- SurvStat@RKI 2.0 (notified national cryptosporidiosis cases).
URL    : https://survstat.rki.de   (interactive query builder -- there is no public API, so the
         export ZIP must be downloaded by hand; see the numbered steps below).

Download the export ZIP (SurvStat -> "Create an individual query"):
  1. Notification            : "Via local and state health department" (default).
  2. Filter "Reference definition" = Yes  (already present by default -- keep it).
  3. Filter "Disease/ Pathogen" = Cryptosporidiosis  (add this second filter; it is the step most
     easily missed -- without it the export is ALL notifiable diseases).
  4. Row attribute           : Week of notification.
  5. Column attribute        : Year of notification.
  6. Display options         : Zero values = Yes ; Totals = No.
  7. "Download ZIP"  ->  survstat.zip (contains Data.csv, a UTF-16, tab-separated
     week(01..53) x year(2001..) matrix; blank cell = week not present that year).
  Confirm in the bundled Info.pdf that "Filter Settings" lists BOTH Reference definition: Yes and
  Disease/ Pathogen: Cryptosporidiosis.

Processing (this script) -- a straightforward ISO-week melt:
  * parse the UTF-16 TSV matrix; melt year-major, weeks 01..53, one row per non-blank cell, dated at
    the Monday of ISO week (year, week);
  * restrict to the year columns [from_year, to_year] (default 2001-2019, pre-COVID) and write the
    weekly (date, count, period) series.

    python prepare_crypto_counts.py                       # read survstat.zip in the crypto folder + compare
    python prepare_crypto_counts.py --zip /path/to.zip    # or point at a ZIP / Data.csv
    python prepare_crypto_counts.py --out rebuilt.csv     # also write the rebuilt date,count series

The shipped `data/crypto_survstat_de/dataset_counts.csv` (990 weeks, 26,024 cases) is built by this
recipe from the bundled original 2026-07-13 export, so the script verifies a byte-identical match to its
(date, count).

HISTORICAL NOTE. An earlier build of the shipped file had a week-53 bug: it collected the non-blank
week-53 cells -- which occur only in the ISO-53-week years (2004, 2015, 2020, with 2, 8, 5 cases) -- and
mis-appended them, in file order, to the FIRST year columns (2001, 2002, 2003) as duplicate-dated rows,
and correspondingly dropped the true 2004/2015 week-53 dates and leaked in a 2020 (post-window) count.
That misattributed 15 cases across years; it has been corrected -- each week-53 now sits at its own ISO
date -- and the dataset regenerated (see archive/crypto_week53fix_2026-07-23/).

NOTE ON A FRESH EXPORT. SurvStat serves the *current* data status and RKI revises historical case counts,
so a later pull will not be byte-identical to the bundled original (~0.99 week-to-week correlation, a few
cases per week); the shipped file remains the canonical modelling input. The development/test split
(dev 2001-2014, test 2015-2019) is applied downstream by `library/datasets.py::load_split`.
"""
import argparse
import csv
import io
import os
import zipfile
from datetime import date

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))            # papercode
DATADIR = os.path.join(HAWKES, "data", "crypto_survstat_de")
SHIPPED = os.path.join(DATADIR, "dataset_counts.csv")
DEFAULT_ZIP = os.path.join(DATADIR, "survstat.zip")


def _read_data_csv(path):
    """Return the raw UTF-16 TSV text of Data.csv from a ZIP, a Data.csv, or a folder."""
    if os.path.isdir(path):
        path = os.path.join(path, "survstat.zip")
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as z:
            name = next(n for n in z.namelist() if n.lower().endswith("data.csv"))
            return z.read(name).decode("utf-16")
    return open(path, "rb").read().decode("utf-16")


def _iso_monday(y, w):
    try:
        return date.fromisocalendar(y, w, 1)
    except ValueError:                                   # invalid week 53 in a 52-week ISO year -> W52
        return date.fromisocalendar(y, 52, 1)


def build_weekly_counts(src, from_year=2001, to_year=2019):
    """SurvStat week x year matrix -> chronological weekly (date, count, period) series over the year
    columns [from_year, to_year]. Each (year, week) cell is dated at the Monday of its ISO week, so a
    week 53 sits at its own year/date (the corrected melt; see the module's HISTORICAL NOTE)."""
    rows = list(csv.reader(io.StringIO(_read_data_csv(src)), delimiter="\t"))
    # rows[0] = title; rows[1] = ['', '2001', '2002', ...]; rows[2:] = ['01', c2001, ...]; rows[54] = W53
    years = [int(y) for y in rows[1][1:] if y.strip()]
    week_rows = [r for r in rows[2:] if r and r[0].strip()]           # 53 rows, labels '01'..'53'

    rec = []
    for yi, y in enumerate(years):                       # year-major
        if not (from_year <= y <= to_year):
            continue
        for wi in range(53):                             # weeks 01..53; blank cells skipped
            cell = week_rows[wi][1 + yi]
            if cell.strip() == "":
                continue
            rec.append((pd.Timestamp(_iso_monday(y, wi + 1)), int(cell)))
    df = pd.DataFrame(rec, columns=["date", "count"]).sort_values("date").reset_index(drop=True)
    df["period"] = 52
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--zip", dest="src", default=DEFAULT_ZIP,
                    help="SurvStat export: a ZIP, a Data.csv, or the crypto folder (default: survstat.zip there)")
    ap.add_argument("--from-year", type=int, default=2001, help="first year column (default 2001)")
    ap.add_argument("--to-year", type=int, default=2019, help="last year column (default 2019, pre-COVID)")
    ap.add_argument("--out", default=None, help="write the rebuilt date,count,period series to this CSV")
    a = ap.parse_args()

    if not os.path.exists(a.src):
        raise SystemExit(f"SurvStat export not found: {a.src}\nDownload it from https://survstat.rki.de "
                         "(see the module docstring for the exact query) and place survstat.zip in "
                         f"{DATADIR}.")
    built = build_weekly_counts(a.src, a.from_year, a.to_year)
    print(f"built {len(built)} weeks, {built['date'].dt.date.min()}..{built['date'].dt.date.max()}, "
          f"total cases {int(built['count'].sum())}")

    if os.path.exists(SHIPPED):
        ship = pd.read_csv(SHIPPED, parse_dates=["date"])[["date", "count"]]
        exact = (len(built) == len(ship)
                 and (built["date"].values == ship["date"].values).all()
                 and (built["count"].values == ship["count"].values).all())
        if exact:
            print(f"VERIFIED: rebuilt (date, count) is byte-identical to "
                  f"data/crypto_survstat_de/dataset_counts.csv ({len(ship)} weeks, "
                  f"{int(ship['count'].sum())} cases).")
        else:
            both = ship.merge(built[["date", "count"]], on="date", how="inner", suffixes=("_shipped", "_rebuilt"))
            corr = float(np.corrcoef(both["count_shipped"], both["count_rebuilt"])[0, 1]) if len(both) > 1 else float("nan")
            mad = float((both["count_shipped"] - both["count_rebuilt"]).abs().mean())
            print(f"vs shipped dataset_counts.csv: {len(both)} common-date weeks, corr={corr:.4f}, "
                  f"mean|Δ|={mad:.2f} cases, Σ shipped={int(ship['count'].sum())} vs "
                  f"Σ rebuilt={int(built['count'].sum())}.")
            print("  (not byte-identical -- expected for a non-original export: RKI revises historical "
                  "counts, and/or the 'Reference definition = Yes' filter was omitted. The shipped file "
                  "is the canonical input -- see the module docstring.)")
    else:
        print("(shipped dataset_counts.csv not found; skipping comparison)")

    if a.out:
        built.to_csv(a.out, index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
