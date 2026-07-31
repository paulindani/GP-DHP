"""Deterministic US calendar covariates (prediction-safe: known arbitrarily far
ahead). Two builders:

  holiday_indicators(dates)  -> individual 0/1 columns for the major holidays
                                that drive gun violence (NYC/GVA).
  gtd_calendar(dates)        -> (symbolic_dates set as (month,day), election-day
                                boolean series) for the GTD history-weighted design.
"""
import numpy as np, pandas as pd


def _nth_weekday(year, month, weekday, n):
    """Date of the n-th `weekday` (Mon=0) of (year, month); n<0 counts from end."""
    if n > 0:
        d = pd.Timestamp(year, month, 1)
        off = (weekday - d.weekday()) % 7
        return d + pd.Timedelta(days=off + 7 * (n - 1))
    last = pd.Timestamp(year, month, 1) + pd.offsets.MonthEnd(0)
    off = (last.weekday() - weekday) % 7
    return last - pd.Timedelta(days=off)


def holiday_indicators(dates):
    """Individual 0/1 indicator columns for gun-violence-relevant US holidays,
    on the true calendar date (not the observed weekday)."""
    d = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    yrs = range(d.dt.year.min(), d.dt.year.max() + 1)
    fixed = {"new_year": (1, 1), "juneteenth": (6, 19), "independence": (7, 4),
             "halloween": (10, 31), "christmas": (12, 25), "new_years_eve": (12, 31)}
    movable = {  # name -> (month, weekday Mon=0, nth)
        "memorial": (5, 0, -1), "labor": (9, 0, 1), "thanksgiving": (11, 3, 4)}
    cols = {}
    for name, (mo, day) in fixed.items():
        s = {pd.Timestamp(y, mo, day) for y in yrs}
        cols[name] = d.isin(s).astype(float).to_numpy()
    for name, (mo, wd, n) in movable.items():
        s = {_nth_weekday(y, mo, wd, n) for y in yrs}
        cols[name] = d.isin(s).astype(float).to_numpy()
    out = pd.DataFrame({"date": d.dt.strftime("%Y-%m-%d"), **cols})
    return out

