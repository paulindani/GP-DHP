"""Rebuild / verify the NYC temperature-anomaly covariate from NOAA GHCN-Daily.

Source : NOAA GHCN-Daily, station Central Park NY US (`USW00094728`), variable TMAX.
URL    : https://www.ncei.noaa.gov/products/land-based-station/global-historical-climatology-network-daily
CSV    : https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/USW00094728.csv

What is exactly reproducible vs. not
------------------------------------
* `tmax_c` (daily maximum temperature, deg C = GHCN `TMAX` / 10) is the **raw** source value and is
  reproduced here **exactly** against the shipped `data/nyc_shootings/temp_anomaly.csv`.
* `anomaly` = `tmax_c` minus a smooth day-of-year harmonic climatology. The climatology fit is not
  bit-pinned (harmonic count / fit window can differ), so re-deriving it reproduces the shipped
  `anomaly` only to within a small RMSE (order 0.1 deg C). The **shipped `anomaly` column is therefore
  the canonical covariate input** consumed by `library/covariates.py`; this script recomputes it with a
  3-harmonic climatology and reports the RMSE as a provenance check, not an exact match.

    python prepare_nyc_temperature.py                    # fetch GHCN + verify tmax_c, report anomaly RMSE
    python prepare_nyc_temperature.py --raw ghcn.csv     # use a local GHCN station CSV
    python prepare_nyc_temperature.py --harmonics 3      # climatology harmonic count for the recompute

The GVA temperature anomaly is a population-weighted composite of several U.S. metro stations (weights
not shipped in code); it is documented in `data/README_DATA_PROVENANCE.md` and is not reproduced
by this single-station script -- its shipped `temp_anomaly.csv` is the canonical input.
"""
import argparse
import os
import urllib.request

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
HAWKES = os.environ.get("HAWKES_ROOT", os.path.dirname(HERE))            # papercode
SHIPPED = os.path.join(HAWKES, "data", "nyc_shootings", "temp_anomaly.csv")

STATION = "USW00094728"                                                  # Central Park NY US
GHCN_CSV = f"https://www.ncei.noaa.gov/data/global-historical-climatology-network-daily/access/{STATION}.csv"


def harmonic_anomaly(dates, tmax_c, n_harm, fit_mask):
    """tmax_c minus a K-harmonic day-of-year climatology (least-squares fit on fit_mask rows)."""
    doy = dates.dt.dayofyear.to_numpy()
    ang = 2 * np.pi * doy / 365.25
    cols = [np.ones_like(ang)]
    for k in range(1, n_harm + 1):
        cols += [np.cos(k * ang), np.sin(k * ang)]
    X = np.column_stack(cols)
    beta, *_ = np.linalg.lstsq(X[fit_mask], tmax_c[fit_mask], rcond=None)
    return tmax_c - X @ beta


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", metavar="CSV", default=None,
                    help="local GHCN station CSV (else fetch Central Park USW00094728)")
    ap.add_argument("--harmonics", type=int, default=3, help="day-of-year harmonics for the climatology")
    a = ap.parse_args()

    if not os.path.exists(SHIPPED):
        raise SystemExit(f"shipped file not found: {SHIPPED}")
    ship = pd.read_csv(SHIPPED, parse_dates=["date"])
    lo, hi = ship["date"].min(), ship["date"].max()

    if a.raw:
        src = a.raw
    else:
        src = os.path.join(os.path.dirname(SHIPPED), "_ghcn_download.csv")
        print(f"fetching {GHCN_CSV} ...")
        urllib.request.urlretrieve(GHCN_CSV, src)
    g = pd.read_csv(src, usecols=["DATE", "TMAX"]).dropna(subset=["TMAX"])
    if a.raw is None and os.path.exists(src):
        os.remove(src)
    g["date"] = pd.to_datetime(g["DATE"]).dt.normalize()
    g["tmax_c"] = g["TMAX"] / 10.0
    g = g[(g["date"] >= lo) & (g["date"] <= hi)][["date", "tmax_c"]]

    m = ship.merge(g, on="date", how="left", suffixes=("_shipped", "_ghcn"))
    nmiss = int(m["tmax_c_ghcn"].isna().sum())
    dtmax = (m["tmax_c_shipped"] - m["tmax_c_ghcn"]).abs()
    ndiff = int((dtmax > 1e-6).sum())
    if nmiss == 0 and ndiff == 0:
        print(f"VERIFIED tmax_c: rebuilt matches shipped exactly ({len(ship)} days, station {STATION}).")
    else:
        print(f"tmax_c: {ndiff} differing / {nmiss} missing of {len(ship)} days "
              f"(GHCN revision or station gap; max|Δ|={np.nanmax(dtmax):.3f} degC).")

    # dev-period fit mask for the climatology (matches covariates.py z-scoring on development rows)
    split = ship["split"] if "split" in ship.columns else None
    if "final_split" in ship.columns:
        fit_mask = (~ship["final_split"].astype(str).str.lower().eq("test")).to_numpy()
    else:
        fit_mask = np.ones(len(ship), bool)
    recomputed = harmonic_anomaly(ship["date"], m["tmax_c_ghcn"].to_numpy(float), a.harmonics, fit_mask)
    rmse = float(np.sqrt(np.nanmean((recomputed - ship["anomaly"].to_numpy(float)) ** 2)))
    print(f"anomaly recompute ({a.harmonics}-harmonic climatology): RMSE vs shipped = {rmse:.4f} degC "
          f"(approximate by construction; the shipped `anomaly` column is the canonical covariate input).")


if __name__ == "__main__":
    main()
