"""Load + clean the two real-world datasets.

Cross-sectional : Telco Customer Churn  (categoricals, MAR missingness, imbalanced target)
Time series     : Shiller S&P 500 monthly (+ VIX daily companion)

The two data-quality traps these datasets teach:
  * Telco  `TotalCharges` is stored as text; 11 blank/whitespace values -> NaN, and every
    one of them has tenure == 0 (a brand-new customer with no completed billing cycle).
    That makes the missingness MAR (mechanically tied to tenure), not random.
  * Shiller uses 0 as a "not available" placeholder. `isna()` reports nothing, yet a CAPE
    ratio or CPI can never be 0. PE10 has leading zeros (needs 10y trailing earnings, so it
    only starts in 1881); the fundamental columns have trailing zeros where the recent months
    are not yet reported. Always check `(df == 0).sum()`, not just `isna()`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import DATA_PROC, DATA_RAW

# Columns where a literal 0 is impossible -> it is a disguised missing value.
SHILLER_ZERO_IS_MISSING = [
    "Dividend", "Earnings", "Consumer Price Index", "Long Interest Rate",
    "Real Price", "Real Dividend", "Real Earnings", "PE10",
]

# Telco service columns whose third level ("No internet service" / "No phone service")
# is redundant with InternetService=="No" / PhoneService=="No".
TELCO_NO_SERVICE = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "MultipleLines",
]


# --------------------------------------------------------------------------- Telco
def load_telco_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "telco_churn.csv")


def clean_telco(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a model-ready Telco frame. Keeps the cleaning decisions explicit so the
    missing-data *mechanism* can still be demonstrated on the raw file in Part 1."""
    df = load_telco_raw() if df is None else df.copy()

    # 1. TotalCharges: text -> numeric. The 11 coercion failures are blank strings.
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # They are all tenure==0 (never billed a full cycle) -> a total of 0 is the honest fix.
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)

    # 2. SeniorCitizen is 0/1 but genuinely categorical; make it read as one.
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"}).astype("category")

    # 3. customerID is a pure identifier -> drop (carries no signal, would just memorize rows).
    df = df.drop(columns=["customerID"])

    # 4. A numeric churn flag alongside the Yes/No label, for correlations / MI.
    df["churn_flag"] = (df["Churn"] == "Yes").astype(int)

    # 5. Make the remaining object columns proper categoricals.
    for c in df.select_dtypes("object").columns:
        df[c] = df[c].astype("category")

    return df


# --------------------------------------------------------------------------- Shiller
def load_shiller_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "sp500_shiller.csv")


def clean_shiller(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a monthly-indexed Shiller frame with disguised zeros fixed and the core
    return / valuation features derived."""
    df = load_shiller_raw() if df is None else df.copy()

    df["Date"] = pd.to_datetime(df["Date"])
    df = df.set_index("Date").sort_index()
    df.index.freq = "MS"  # month-start frequency (regular monthly series)

    # Disguised missingness: 0 -> NaN only where 0 is impossible.
    df[SHILLER_ZERO_IS_MISSING] = df[SHILLER_ZERO_IS_MISSING].replace(0.0, np.nan)

    # Derived features (the things we actually model / analyse).
    df["return"] = df["SP500"].pct_change(fill_method=None)        # simple monthly return
    df["log_return"] = np.log(df["SP500"]).diff()                 # log return (additive)
    df["real_return"] = df["Real Price"].pct_change(fill_method=None)  # inflation-adjusted
    df["cape"] = df["PE10"]                                  # Shiller CAPE (alias)
    df["cape_z"] = (df["cape"] - df["cape"].mean()) / df["cape"].std()

    return df


# --------------------------------------------------------------------------- VIX (companion)
def load_vix_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_RAW / "vix_daily.csv")


def clean_vix(df: pd.DataFrame | None = None) -> pd.DataFrame:
    df = load_vix_raw() if df is None else df.copy()
    df.columns = [c.strip().lower() for c in df.columns]      # DATE/OPEN/... -> date/open/...
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date").sort_index()


# --------------------------------------------------------------------------- stock panel (Part 4)
# 12 S&P 500 large caps in 6 same-sector pairs (2013-2018 daily close), curated from the
# plotly all_stocks_5yr mirror. The pairs give natural cointegration candidates.
SECTORS = {
    "AAPL": "Tech", "MSFT": "Tech", "JPM": "Bank", "BAC": "Bank",
    "XOM": "Energy", "CVX": "Energy", "KO": "Staples", "PEP": "Staples",
    "JNJ": "Health", "PFE": "Health", "WMT": "Retail", "TGT": "Retail",
}


def load_stock_panel() -> pd.DataFrame:
    """Wide panel of daily close prices: rows = trading days, columns = tickers."""
    df = pd.read_csv(DATA_RAW / "stock_panel.csv", parse_dates=["date"]).set_index("date")
    return df.sort_index()


def stock_log_returns(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    """Daily log returns of the panel (stationary; the input to correlation/PCA/VAR/Granger)."""
    px = load_stock_panel() if prices is None else prices
    return np.log(px).diff().dropna()


# --------------------------------------------------------------------------- build all
def build_processed() -> dict[str, pd.DataFrame]:
    """Clean every dataset and persist to data/processed/. Returns the frames."""
    telco = clean_telco()
    shiller = clean_shiller()
    vix = clean_vix()

    telco.to_csv(DATA_PROC / "telco_clean.csv", index=False)
    shiller.to_csv(DATA_PROC / "sp500_monthly.csv")  # keep the DatetimeIndex
    vix.to_csv(DATA_PROC / "vix_clean.csv")
    return {"telco": telco, "shiller": shiller, "vix": vix}


if __name__ == "__main__":
    frames = build_processed()
    for name, f in frames.items():
        print(f"{name:8s} {f.shape}")
