"""Propagate *weather-forecast* uncertainty into the demand forecast.

Parts 3-5 fed the model the **realized** weather — perfect foresight. In production you forecast
demand from a weather *forecast*, which has error. We don't have historical forecasts, so we
**simulate** a plausible ~24h-ahead error (calibrated magnitudes, temporally correlated), draw a
Monte-Carlo ensemble of weather scenarios, and push each through the fixed model to get the
input-driven spread of demand forecasts.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Illustrative ~24h-ahead weather-forecast error (stationary std). Real numbers are in this range:
# day-ahead temperature RMSE ~1-2 C, humidity ~8-10 pts, wind ~3 km/h.
FORECAST_SD = {"temp_C": 1.5, "hum_pct": 8.0, "wind_kmh": 3.0}


def ar1_noise(n: int, sd: float, phi: float = 0.8, rng=None) -> np.ndarray:
    """AR(1) noise with stationary std `sd` — forecast errors are autocorrelated, not white."""
    rng = rng or np.random.default_rng()
    e = rng.normal(0, sd * np.sqrt(1 - phi ** 2), n)
    x = np.zeros(n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x


def perturb_weather(df: pd.DataFrame, rng, sd=FORECAST_SD, phi: float = 0.8) -> pd.DataFrame:
    """Return a copy of `df` with the weather columns nudged by a simulated forecast error."""
    d = df.copy()
    for c, s in sd.items():
        if c in d:
            d[c] = np.clip(d[c].to_numpy() + ar1_noise(len(d), s, phi, rng), 0, None)
    return d


def monte_carlo(model, X: pd.DataFrame, cols: list[str], n_scenarios: int = 200,
                seed: int = 0, sd=FORECAST_SD) -> np.ndarray:
    """Ensemble of demand forecasts under `n_scenarios` weather-forecast draws (model fixed)."""
    rng = np.random.default_rng(seed)
    P = np.empty((n_scenarios, len(X)))
    for s in range(n_scenarios):
        P[s] = np.clip(model.predict(perturb_weather(X, rng, sd)[cols]), 0, None)
    return P
