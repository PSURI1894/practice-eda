"""Load + clean the PJM hourly electricity-demand panel (11 regional zones, 2002–2018).

PJM is a large US grid operator; each column is a zone's hourly demand in **megawatts (MW)**. The
series are in **local time**, which creates two daylight-saving artifacts that must be handled:
  * **duplicate timestamps** — the autumn "fall back" repeats the 01:00–02:00 hour,
  * **missing timestamps** — the spring "spring forward" skips an hour.
The headline structure is **triple seasonality** (daily × weekly × annual) and a **bimodal** annual
cycle: demand peaks in *both* winter (heating) and summer (air-conditioning).
"""
from __future__ import annotations

import holidays
import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW

REGIONS = ["PJME", "PJMW", "AEP", "COMED", "DAYTON", "DOM", "DUQ", "FE", "DEOK", "NI", "EKPC"]
PRIMARY = "PJME"                                   # PJM East — the longest, gap-free zone
PANEL_LONG = ["PJME", "PJMW", "AEP", "DAYTON", "DOM", "DUQ"]   # zones with full 2005–2018 overlap
_US_HOL = holidays.US(years=range(2002, 2019))


def load_panel() -> pd.DataFrame:
    """The wide region×time panel as assembled (one column per zone, MW)."""
    df = pd.read_csv(DATA_RAW / "pjm_energy.csv", parse_dates=["Datetime"]).set_index("Datetime")
    return df.sort_index()


def primary() -> pd.Series:
    """The PJME demand series on a *complete* hourly index — duplicate (DST) timestamps dropped and
    the handful of missing hours filled by time interpolation. This is the clean series we model."""
    s = load_panel()[PRIMARY]
    s = s[~s.index.duplicated(keep="first")]
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    return s.reindex(full).interpolate("time").rename("load_mw")


def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    """Attach the calendar features that drive electricity demand."""
    idx = df.index
    df = df.copy()
    df["hour"] = idx.hour
    df["dow"] = idx.dayofweek                      # 0 = Monday
    df["month"] = idx.month
    df["year"] = idx.year
    df["is_weekend"] = (idx.dayofweek >= 5).astype(int)
    df["is_holiday"] = pd.Series(idx.date, index=idx).isin(_US_HOL).astype(int).values
    df["season"] = (idx.month % 12 // 3).map({0: "winter", 1: "spring", 2: "summer", 3: "fall"})
    return df


def clean_primary() -> pd.DataFrame:
    """Primary load series + calendar features — the modelling-ready table."""
    return add_calendar(primary().to_frame())


def panel_overlap() -> pd.DataFrame:
    """The six long zones on their common gap-free window — for the multivariate analysis."""
    p = load_panel()[PANEL_LONG]
    p = p[~p.index.duplicated(keep="first")]
    return p.dropna()


def build_processed() -> pd.DataFrame:
    d = clean_primary()
    d.to_csv(DATA_PROC / "pjme_clean.csv")
    return d


if __name__ == "__main__":
    d = build_processed()
    print(d.shape, "| span", d.index.min(), "->", d.index.max(),
          "| mean %.0f MW" % d.load_mw.mean(), "| holidays flagged:", int(d.is_holiday.sum()))
