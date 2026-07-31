"""Dataset split loading for the GP-DHP experiments.

Single source of the DATA split table and the two load-split variants (unified from the former
duplicated copies in gpdhp_runner.py and jax_baselines_runner.py):
  * load_split      -> (y_dev, y_test, period, daily)                    -- GP-DHP / constrained fits
  * load_split_cal  -> (y_dev, y_test, period, daily, cal, dates)        -- baseline / GP-Hawkes / Browning
Both share DATA and the same fit/test masking (year-based for dengue/crypto/rand, final_split for nyc/gva).
"""
import os
import numpy as np
import pandas as pd

HAWKES = os.environ.get("HAWKES_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# dataset -> (data_dir, period, test_years or None).
DATA = {"dengue": ("dengue_singapore", 52, range(2018, 2023)),
        "crypto": ("crypto_survstat_de", 52, range(2015, 2020)),
        "rand":   ("RAND_terrorism", 365, range(2005, 2010)),
        "nyc":    ("nyc_shootings", 365, None),
        "gva":    ("gva_allshootings", 365, None)}


def _split_masks(dataset):
    data_dir, period, years = DATA[dataset]
    df = pd.read_csv(f"{HAWKES}/data/{data_dir}/dataset_counts.csv")
    if years is not None:                                              # year-based (dengue/crypto/rand)
        yr = df["date"].str[:4].astype(int)
        dev_mask = (yr < min(years)).to_numpy(); test_mask = yr.isin(list(years)).to_numpy()
    else:                                                              # final_split (nyc/gva)
        split = (df["final_split"] if "final_split" in df.columns else df["split"]).astype(str).str.lower()
        test_mask = split.eq("test").to_numpy(); dev_mask = ~test_mask
    return df, period, dev_mask, test_mask


def load_split(dataset):
    """(y_dev, y_test, period, daily) -- 4-tuple used by GP-DHP MAP + the constrained fits."""
    df, period, dev_mask, test_mask = _split_masks(dataset)
    y = df["count"].to_numpy(float)
    return y[dev_mask], y[test_mask], period, (period == 365)


def load_split_cal(dataset):
    """(y_dev, y_test, period, daily, cal, dates) -- adds the 1-based calendar index and the
    date labels, dev-then-test; used by the baseline / GP-Hawkes / Browning runners."""
    df, period, dev_mask, test_mask = _split_masks(dataset)
    y = df["count"].to_numpy(float)
    cal = np.concatenate([np.where(dev_mask)[0] + 1, np.where(test_mask)[0] + 1])
    dates = np.concatenate([df["date"].to_numpy()[dev_mask], df["date"].to_numpy()[test_mask]])
    return y[dev_mask], y[test_mask], float(period), (period == 365), cal, dates
