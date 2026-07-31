"""Rebuild the New York City shootings daily count series from NYC Open Data.

Source : New York City Police Department -- *Shootings (2006-Present)*, via NYC Open Data
         (Socrata dataset `5ucz-vwe8`).
URL    : https://data.cityofnewyork.us/Public-Safety/Shootings-2006-Present-/5ucz-vwe8
API    : https://data.cityofnewyork.us/resource/5ucz-vwe8.json

Recipe (reproduces the shipped `data/nyc_shootings/dataset_counts.csv` exactly):
  * each raw row is one shooting *victim*; multiple victims of one incident share an `incident_key`;
  * count **distinct `incident_key` per `occur_date`** (the shipped `count_method`), then place on a
    gap-free daily grid over the shipped span (2006-01-01 .. 2025-12-31, `period=365`).

The source is openly redistributable, so the script fetches it directly:

    python prepare_nyc_counts.py                      # fetch from the Socrata API + verify
    python prepare_nyc_counts.py --raw records.json   # use a previously saved API JSON
    python prepare_nyc_counts.py --write out.csv      # also write the rebuilt date,count series

It restricts to the shipped date span and verifies the rebuilt counts equal the shipped
`dataset_counts.csv` exactly. NYC Open Data keeps growing (later occurrence dates appear over time);
past incidents in the shipped window are stable, so the check reports any differing days rather than
drifting. The chronological development/test split is applied downstream by
`library/datasets.py::load_split`. The paired temperature covariate is handled by
`prepare_covariates.py`.
"""
import argparse
import json
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))            # papercode
SHIPPED = os.path.join(HAWKES, "data", "nyc_shootings", "dataset_counts.csv")

RESOURCE = "5ucz-vwe8"
API = f"https://data.cityofnewyork.us/resource/{RESOURCE}.json"


def fetch_records():
    """Page through the Socrata API for (incident_key, occur_date) and return a DataFrame."""
    rows, offset, page = [], 0, 50000
    while True:
        url = f"{API}?$select=incident_key,occur_date&$order=:id&$limit={page}&$offset={offset}"
        with urllib.request.urlopen(url, timeout=90) as r:
            chunk = json.load(r)
        if not chunk:
            break
        rows.extend(chunk)
        offset += len(chunk)
        if len(chunk) < page:
            break
    return pd.DataFrame(rows)


def build_daily_counts(recs, lo, hi):
    """Victim rows -> distinct-incident-key-per-day counts on a gap-free grid over [lo, hi]."""
    recs = recs.copy()
    recs["date"] = pd.to_datetime(recs["occur_date"]).dt.normalize()
    sub = recs[(recs["date"] >= lo) & (recs["date"] <= hi)]
    daily = sub.groupby("date")["incident_key"].nunique()
    grid = pd.date_range(lo, hi, freq="D")
    return pd.DataFrame({"date": grid, "count": daily.reindex(grid, fill_value=0).to_numpy(int)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", metavar="JSON", default=None,
                    help="use a previously saved API response JSON instead of fetching")
    ap.add_argument("--write", metavar="OUT", default=None,
                    help="write the rebuilt date,count series to this CSV path")
    a = ap.parse_args()

    if a.raw:
        with open(a.raw) as fh:
            recs = pd.DataFrame(json.load(fh))
    else:
        print(f"fetching from {API} ...")
        recs = fetch_records()
    print(f"got {len(recs)} victim rows")

    if not os.path.exists(SHIPPED):
        raise SystemExit(f"shipped file not found: {SHIPPED}")
    ship = pd.read_csv(SHIPPED, parse_dates=["date"])[["date", "count"]]
    lo, hi = ship["date"].min(), ship["date"].max()

    built = build_daily_counts(recs, lo, hi)
    print(f"built {len(built)} days, {lo.date()}..{hi.date()}, total incidents {int(built['count'].sum())}")

    m = built.merge(ship, on="date", how="outer", suffixes=("_built", "_shipped")).fillna(-1)
    ndiff = int((m["count_built"].astype(int) != m["count_shipped"].astype(int)).sum())
    if len(built) == len(ship) and ndiff == 0:
        print(f"VERIFIED: rebuilt counts match data/nyc_shootings/dataset_counts.csv exactly "
              f"({len(ship)} days, {int(ship['count'].sum())} incidents).")
    else:
        print(f"WARNING: {ndiff} day(s) differ, built {len(built)} vs shipped {len(ship)} rows "
              f"(possible NYPD data revision; the shipped file is the modelling input).")

    if a.write:
        built.to_csv(a.write, index=False)
        print(f"wrote {a.write}")


if __name__ == "__main__":
    main()
