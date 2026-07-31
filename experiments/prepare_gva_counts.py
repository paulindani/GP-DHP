"""Rebuild the Gun Violence Archive shootings daily count series from the figshare dataset.

Source : Gun Violence Archive -- *Gun Violence: All Shootings*, via figshare.
URL    : https://figshare.com/articles/dataset/Gun_Violence_-_All_Shootings/25517224
DOI    : 10.6084/m9.figshare.25517224.v2
File   : `all-shootings-2014-2023.csv` (the combined 2014-2023 incident table, ~72 MB;
         figshare file id 45398374).
Methodology: https://www.gunviolencearchive.org/methodology
         (archived: https://web.archive.org/web/20260331020805/https://www.gunviolencearchive.org/methodology)

Recipe (reproduces the shipped `data/gva_allshootings/dataset_counts.csv` exactly):
  * each raw row is one shooting incident (unique `Incident_ID`);
  * count **distinct `Incident_ID` per `Incident_Date`**, then place on a gap-free daily grid over
    the shipped span (2014-01-01 .. 2023-12-31, `period=365`).

    python prepare_gva_counts.py                       # download the figshare CSV + verify
    python prepare_gva_counts.py --raw all-shootings-2014-2023.csv   # use a local copy
    python prepare_gva_counts.py --write out.csv       # also write the rebuilt date,count series

It restricts to the shipped span and verifies the rebuilt counts equal the shipped
`dataset_counts.csv` exactly. The chronological development/test split is applied downstream by
`library/datasets.py::load_split`; the paired temperature covariate is handled by
`prepare_covariates.py`.
"""
import argparse
import os
import urllib.request

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))            # papercode
SHIPPED = os.path.join(HAWKES, "data", "gva_allshootings", "dataset_counts.csv")

FIGSHARE_FILE = "https://ndownloader.figshare.com/files/45398374"        # all-shootings-2014-2023.csv


def download(dest):
    print(f"downloading {FIGSHARE_FILE} (~72 MB) -> {dest} ...")
    urllib.request.urlretrieve(FIGSHARE_FILE, dest)
    return dest


def build_daily_counts(raw_path, lo, hi):
    """GVA incident rows -> distinct-incident-per-day counts on a gap-free grid over [lo, hi]."""
    raw = pd.read_csv(raw_path, usecols=["Incident_ID", "Incident_Date"])
    raw["date"] = pd.to_datetime(raw["Incident_Date"]).dt.normalize()
    sub = raw[(raw["date"] >= lo) & (raw["date"] <= hi)]
    daily = sub.groupby("date")["Incident_ID"].nunique()
    grid = pd.date_range(lo, hi, freq="D")
    return pd.DataFrame({"date": grid, "count": daily.reindex(grid, fill_value=0).to_numpy(int)})


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", metavar="CSV", default=None,
                    help="path to a local all-shootings-2014-2023.csv (else download from figshare)")
    ap.add_argument("--write", metavar="OUT", default=None,
                    help="write the rebuilt date,count series to this CSV path")
    a = ap.parse_args()

    if not os.path.exists(SHIPPED):
        raise SystemExit(f"shipped file not found: {SHIPPED}")
    ship = pd.read_csv(SHIPPED, parse_dates=["date"])[["date", "count"]]
    lo, hi = ship["date"].min(), ship["date"].max()

    raw_path = a.raw
    tmp = None
    if raw_path is None:
        tmp = os.path.join(os.path.dirname(SHIPPED), "_gva_download.csv")
        raw_path = download(tmp)
    built = build_daily_counts(raw_path, lo, hi)
    if tmp and os.path.exists(tmp):
        os.remove(tmp)
    print(f"built {len(built)} days, {lo.date()}..{hi.date()}, total incidents {int(built['count'].sum())}")

    m = built.merge(ship, on="date", how="outer", suffixes=("_built", "_shipped")).fillna(-1)
    ndiff = int((m["count_built"].astype(int) != m["count_shipped"].astype(int)).sum())
    if len(built) == len(ship) and ndiff == 0:
        print(f"VERIFIED: rebuilt counts match data/gva_allshootings/dataset_counts.csv exactly "
              f"({len(ship)} days, {int(ship['count'].sum())} incidents).")
    else:
        print(f"WARNING: {ndiff} day(s) differ, built {len(built)} vs shipped {len(ship)} rows "
              f"(possible GVA revision / figshare version change; the shipped file is the modelling input).")

    if a.write:
        built.to_csv(a.write, index=False)
        print(f"wrote {a.write}")


if __name__ == "__main__":
    main()
