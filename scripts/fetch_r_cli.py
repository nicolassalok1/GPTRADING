#!/usr/bin/env python
"""
CLI helper to fetch/interpolate risk-free rate r(T) (decimal) using yfinance.
Falls back to DEFAULT_RF_RATE env var (default 0.02) on failure.
"""
import sys
import os
import time
import math
from typing import Dict, List, Tuple

import numpy as np
import yfinance as yf
import requests

RATE_SYMBOLS: Dict[str, float] = {
    "^IRX": 0.25,  # 13-week
    "^FVX": 5.0,   # 5-year
    "^TNX": 10.0,  # 10-year
}
FRED_SERIES: Dict[str, str] = {
    "^IRX": "DGS3MO",
    "^FVX": "DGS5",
    "^TNX": "DGS10",
}
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0
DEFAULT_RF = float(os.getenv("DEFAULT_RF_RATE", "0.02"))


def _fetch_from_fred(series_id: str) -> float:
    """Fetch the latest FRED value for a series id and return as float."""
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("Empty FRED response")
    last_val = lines[-1].split(",")[-1]
    if last_val.strip() == ".":
        raise RuntimeError("No usable FRED value")
    return float(last_val)


def _fetch_last_close(symbol: str) -> float:
    """Fetch latest daily close via yfinance with retries."""
    last_err = None
    for _ in range(MAX_RETRIES):
        try:
            hist = yf.Ticker(symbol).history(period="5d", interval="1d")
            if not hist.empty and "Close" in hist.columns:
                return float(hist["Close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(SLEEP_BETWEEN)
    if last_err:
        raise RuntimeError(f"Failed to fetch {symbol}: {last_err}") from last_err
    raise RuntimeError(f"Failed to fetch {symbol}: empty history")


def compute_r(T: float) -> float:
    """
    Interpolate risk-free rate r(T) from available proxies or fall back to default.

    Args:
        T: Target maturity in years.
    Returns:
        float: Risk-free rate in decimal.
    """
    points: List[Tuple[float, float]] = []
    for sym, mat in RATE_SYMBOLS.items():
        try:
            pct = _fetch_last_close(sym)
            val = pct / 100.0
            if math.isfinite(val) and val > 0:
                points.append((mat, val))
                continue
        except Exception:
            pass
        fred_id = FRED_SERIES.get(sym)
        if fred_id:
            try:
                val = _fetch_from_fred(fred_id) / 100.0
                if math.isfinite(val) and val > 0:
                    points.append((mat, val))
            except Exception:
                continue
    if not points:
        return DEFAULT_RF
    points.sort(key=lambda x: x[0])
    maturities = np.array([p[0] for p in points], dtype=float)
    rates = np.array([p[1] for p in points], dtype=float)
    if len(points) == 1:
        val = float(rates[0])
        return val if math.isfinite(val) and val > 0 else DEFAULT_RF
    T_clamped = np.clip(T, maturities.min(), maturities.max())
    val = float(np.interp(T_clamped, maturities, rates))
    return val if math.isfinite(val) and val > 0 else DEFAULT_RF


def main():
    """CLI entrypoint: read T from argv, compute r(T), print decimal value."""
    try:
        T = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    except Exception:
        T = 1.0
    try:
        r = compute_r(T)
    except Exception:
        r = DEFAULT_RF
    # Print only the decimal number so caller can parse easily.
    if not math.isfinite(r):
        r = DEFAULT_RF
    sys.stdout.write(f"{r:.6f}\n")


if __name__ == "__main__":
    main()
