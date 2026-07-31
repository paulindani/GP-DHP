"""Shared external-covariate builder for the covariate-matched benchmark (nyc/gva).

Builds the covariate block used by every model on nyc/gva:
  * column 0  = daily temperature anomaly, z-scored on the DEVELOPMENT period only
                (mean/std from the dev rows, applied to the whole dev+test series);
  * columns 1..9 = the 9 holiday 0/1 indicators from ``calendar_utils.holiday_indicators``.
K = 10 covariates.  This is the single source of truth used by the neural models and
GP-DHP, so all models see identical covariates on nyc/gva.
"""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
from .calendar_utils import holiday_indicators

# 9 holidays, fixed order/names
HOL = ["new_year", "juneteenth", "independence", "halloween", "christmas",
       "new_years_eve", "memorial", "labor", "thanksgiving"]
COV_DATADIR = {"nyc": "nyc_shootings", "gva": "gva_allshootings"}   # datasets that carry covariates
N_COV = 1 + len(HOL)                                                # 10


def has_covariates(dataset):
    return dataset in COV_DATADIR


def _resolve_root(root):
    return root or os.environ.get("HAWKES_ROOT") or \
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # papercode/


def build_covariates(dataset, root=None):
    """Build the (n,10) covariate matrix for a covariate dataset and split it by the
    dataset's development/test mask.

    Returns (cov_dev, cov_test, dates):
      cov_dev  : (n_dev, 10) float array, row-aligned to the development counts
      cov_test : (n_test, 10) float array, row-aligned to the test counts
      dates    : pandas Series of all dates (dev then test, chronological)
    """
    root = _resolve_root(root)
    if dataset not in COV_DATADIR:
        raise ValueError(f"{dataset} has no covariates (only {list(COV_DATADIR)})")
    datadir = COV_DATADIR[dataset]
    df = pd.read_csv(f"{root}/data/{datadir}/dataset_counts.csv", parse_dates=["date"])
    split = df["final_split"] if "final_split" in df.columns else df["split"]
    is_test = split.astype(str).str.lower().eq("test").to_numpy()
    T = int((~is_test).sum())                                      # number of development rows
    dates = df["date"]

    # temperature anomaly aligned by date, z-scored on the development period
    ta = pd.read_csv(f"{root}/data/{datadir}/temp_anomaly.csv", parse_dates=["date"])
    lut = dict(zip(ta["date"].dt.strftime("%Y-%m-%d"), ta["anomaly"].astype(float)))
    anom = dates.dt.strftime("%Y-%m-%d").map(lut).to_numpy(float)
    if not np.isfinite(anom).all():
        raise ValueError(f"{dataset}: {int((~np.isfinite(anom)).sum())} dates missing a temperature anomaly")
    anom = (anom - anom[:T].mean()) / anom[:T].std()

    hol = holiday_indicators(dates)                                # DataFrame with the 9 HOL columns
    C = np.column_stack([anom] + [hol[c].to_numpy(float) for c in HOL])   # (n, 10)
    cov_dev, cov_test = C[:T], C[T:]
    return cov_dev, cov_test, dates
