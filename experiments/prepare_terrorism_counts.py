"""Rebuild the worldwide-terrorism daily count series from the raw RAND incident file.

The raw RAND file is **not redistributed** in this repository: the RAND Database of Worldwide
Terrorism Incidents (RDWTI) user agreement prohibits redistributing RDWTI data. Only the derived
daily-count series (`data/RAND_terrorism/dataset_counts.csv`, aggregate counts) is included, on
the same footing as the other four series (processed counts in the repo, raw obtained from the source).

This script keeps the raw -> counts recipe in the repo as runnable, verifiable code. To use it,
download the database yourself and point the script at the file:

    Source : RAND Database of Worldwide Terrorism Incidents (RDWTI), worldwide incidents 1968-2009.
    URL    : https://www.rand.org/nsrd/projects/terrorism-incidents.html
    Cite   : "RAND Database of Worldwide Terrorism Incidents", https://smapp.rand.org/rwtid

    # after downloading the incident CSV (columns: Date, City, Country, Perpetrator, Weapon,
    # Injuries, Fatalities, Description):
    python prepare_terrorism_counts.py --raw /path/to/RAND_..._Incidents.csv

The script parses the incident dates, counts incidents per occurrence date, reindexes onto a gap-free
daily grid, and verifies the result equals the shipped `dataset_counts.csv` exactly (0 differing days,
40,129 incidents -> 15,302 daily rows over 1968-02-09 .. 2009-12-31). The chronological
development/test split (final five years, 2005-2009, held out) is applied downstream by
`library/datasets.py::load_split`.
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))            # papercode
DATADIR = os.path.join(HAWKES, "data", "RAND_terrorism")
SHIPPED = os.path.join(DATADIR, "dataset_counts.csv")
# default location the user may populate with their own download (not redistributed here)
DEFAULT_RAW = os.path.join(DATADIR, "RAND_Database_of_Worldwide_Terrorism_Incidents.csv")

SOURCE_URL = "https://www.rand.org/nsrd/projects/terrorism-incidents.html"
CITATION = '"RAND Database of Worldwide Terrorism Incidents", https://smapp.rand.org/rwtid'


def build_daily_counts(raw_path):
    """Raw RAND incident rows -> gap-free daily count series (DataFrame with date, count)."""
    raw = pd.read_csv(raw_path, encoding="latin-1")                     # RAND file is latin-1
    # dates are like "9-Feb-68"; %y sends 68->2068, so pull two-digit years >2025 back one century
    d = pd.to_datetime(raw["Date"], format="%d-%b-%y", errors="coerce")
    if d.isna().any():
        raise ValueError(f"{int(d.isna().sum())} unparseable dates in the raw RAND file")
    d = d.mask(d.dt.year > 2025, d - pd.offsets.DateOffset(years=100)).dt.normalize()
    daily = d.value_counts().sort_index()
    grid = pd.date_range(daily.index.min(), daily.index.max(), freq="D")  # continuous daily span
    counts = daily.reindex(grid, fill_value=0).astype(int)
    return pd.DataFrame({"date": grid, "count": counts.to_numpy()})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", default=DEFAULT_RAW,
                    help="path to the RAND incident CSV you downloaded (default: the "
                         "RAND_terrorism/ folder, where it is NOT shipped — see the module docstring)")
    ap.add_argument("--write", metavar="OUT", default=None,
                    help="write the rebuilt date,count series to this CSV path")
    a = ap.parse_args()

    if not os.path.exists(a.raw):
        sys.exit(
            "Raw RAND incident file not found (it is not redistributed here per the RDWTI terms).\n"
            f"  Download it from: {SOURCE_URL}\n"
            f"  Cite as:         {CITATION}\n"
            f"  Then re-run:     python prepare_terrorism_counts.py --raw /path/to/incidents.csv\n"
            "The derived data/RAND_terrorism/dataset_counts.csv is already included; this script "
            "only reproduces it from the raw file for verification.")

    built = build_daily_counts(a.raw)
    print(f"raw -> {len(built)} daily rows, {built['date'].dt.date.min()}..{built['date'].dt.date.max()}, "
          f"total incidents {int(built['count'].sum())}")

    if os.path.exists(SHIPPED):
        ship = pd.read_csv(SHIPPED, parse_dates=["date"])[["date", "count"]]
        m = built.merge(ship, on="date", how="outer", suffixes=("_built", "_shipped")).fillna(0)
        ndiff = int((m["count_built"].astype(int) != m["count_shipped"].astype(int)).sum())
        assert len(built) == len(ship), f"row count differs: built {len(built)} vs shipped {len(ship)}"
        assert ndiff == 0, f"{ndiff} day(s) differ from the shipped counts"
        print(f"VERIFIED: rebuilt counts match data/RAND_terrorism/dataset_counts.csv exactly "
              f"({len(ship)} days, {int(ship['count'].sum())} incidents).")
    else:
        print("(shipped dataset_counts.csv not found; skipping verification)")

    if a.write:
        built.to_csv(a.write, index=False)
        print(f"wrote {a.write}")


if __name__ == "__main__":
    main()
