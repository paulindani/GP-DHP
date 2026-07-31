"""Rebuild the Singapore dengue weekly count series from the public data.gov.sg source.

Source : Ministry of Health, Singapore -- Weekly Infectious Disease Bulletin, distributed by
         data.gov.sg, dataset resource `d_ca168b2cb763640d72c4600a68f9909e`.
URL    : https://data.gov.sg/datasets/d_ca168b2cb763640d72c4600a68f9909e/view
API    : https://data.gov.sg/api/action/datastore_search?resource_id=d_ca168b2cb763640d72c4600a68f9909e

Recipe (reproduces the shipped `data/dengue_singapore/dataset_counts.csv` exactly):
  * keep the two dengue rows -- `Dengue Fever` + `Dengue Haemorrhagic Fever`;
  * sum the weekly case counts per epidemiological week (`epi_week`, e.g. "2012-W01");
  * restrict to 2012-W01 .. 2022-W52 (574 weeks) and place them, in chronological week order,
    on a gap-free 7-day grid starting 2012-01-01 (`period=52`).

Unlike RAND, this source is openly redistributable, so the script can fetch it directly:

    python prepare_dengue_counts.py                       # fetch from the data.gov.sg API + verify
    python prepare_dengue_counts.py --raw records.json    # use a previously saved API JSON
    python prepare_dengue_counts.py --write out.csv       # also write the rebuilt date,count series

It then verifies the rebuilt counts equal the shipped `dataset_counts.csv` exactly. Note the portal
serves the *current* bulletin, which can be revised retrospectively; the shipped counts are the
canonical modelling input (a later pull may differ if MOH revises past weeks -- the check reports any
differing weeks rather than silently drifting). The development/test split (2012-2017 dev, 2018-2022
test) is applied downstream by `library/datasets.py::load_split`.
"""
import argparse
import json
import os
import sys
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))            # papercode
SHIPPED = os.path.join(HAWKES, "data", "dengue_singapore", "dataset_counts.csv")

RESOURCE_ID = "d_ca168b2cb763640d72c4600a68f9909e"
API = f"https://data.gov.sg/api/action/datastore_search?resource_id={RESOURCE_ID}"
DISEASES = ["Dengue Fever", "Dengue Haemorrhagic Fever"]
START_DATE = "2012-01-01"       # first epi-week (2012-W01) placed here; 7-day grid thereafter
FIRST_WEEK, LAST_WEEK = "2012-W01", "2022-W52"


def fetch_records():
    """Page through the datastore_search API and return all rows as a DataFrame."""
    rows, offset, total = [], 0, None
    while total is None or offset < total:
        with urllib.request.urlopen(f"{API}&limit=5000&offset={offset}", timeout=60) as r:
            j = json.load(r)["result"]
        total = j["total"]
        rows.extend(j["records"])
        got = len(j["records"])
        if not got:
            break
        offset += got
    return pd.DataFrame(rows)


def build_weekly_counts(recs):
    """data.gov.sg bulletin rows -> gap-free weekly dengue count series (date, count)."""
    recs = recs.copy()
    recs["cases"] = recs["no._of_cases"].astype(int)
    sub = recs[recs["disease"].isin(DISEASES)]
    g = sub.groupby("epi_week")["cases"].sum()
    weeks = sorted(w for w in g.index if FIRST_WEEK <= w <= LAST_WEEK)   # lexical == chronological
    g = g.loc[weeks]
    grid = pd.date_range(START_DATE, periods=len(g), freq="7D")
    return pd.DataFrame({"date": grid, "count": g.to_numpy(int)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", metavar="JSON", default=None,
                    help="use a previously saved API response JSON instead of fetching")
    ap.add_argument("--write", metavar="OUT", default=None,
                    help="write the rebuilt date,count series to this CSV path")
    a = ap.parse_args()

    if a.raw:
        with open(a.raw) as fh:
            j = json.load(fh)
        recs = pd.DataFrame(j["result"]["records"] if "result" in j else j)
    else:
        print(f"fetching from {API} ...")
        recs = fetch_records()
    print(f"got {len(recs)} bulletin rows")

    built = build_weekly_counts(recs)
    print(f"built {len(built)} weeks, {built['date'].dt.date.min()}..{built['date'].dt.date.max()}, "
          f"total dengue cases {int(built['count'].sum())}")

    if os.path.exists(SHIPPED):
        ship = pd.read_csv(SHIPPED, parse_dates=["date"])[["date", "count"]]
        m = built.merge(ship, on="date", how="outer", suffixes=("_built", "_shipped")).fillna(-1)
        ndiff = int((m["count_built"].astype(int) != m["count_shipped"].astype(int)).sum())
        if len(built) == len(ship) and ndiff == 0:
            print(f"VERIFIED: rebuilt counts match data/dengue_singapore/dataset_counts.csv "
                  f"exactly ({len(ship)} weeks, {int(ship['count'].sum())} cases).")
        else:
            print(f"WARNING: {ndiff} week(s) differ, built {len(built)} vs shipped {len(ship)} rows "
                  f"(likely a retrospective MOH revision; the shipped file is the modelling input).")
    else:
        print("(shipped dataset_counts.csv not found; skipping verification)")

    if a.write:
        built.to_csv(a.write, index=False)
        print(f"wrote {a.write}")


if __name__ == "__main__":
    main()
