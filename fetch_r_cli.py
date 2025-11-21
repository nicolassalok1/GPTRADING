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

RATE_SYMBOLS: Dict[str, float] = {
    "^IRX": 0.25,  # 13-week
    "^FVX": 5.0,   # 5-year
    "^TNX": 10.0,  # 10-year
}
MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0
DEFAULT_RF = float(os.getenv("DEFAULT_RF_RATE", "0.02"))


def _fetch_last_close(symbol: str) -> float:
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
    points: List[Tuple[float, float]] = []
    for sym, mat in RATE_SYMBOLS.items():
        try:
            pct = _fetch_last_close(sym)
            points.append((mat, pct / 100.0))
        except Exception:
            continue
    if not points:
        return DEFAULT_RF
    points.sort(key=lambda x: x[0])
    maturities = np.array([p[0] for p in points], dtype=float)
    rates = np.array([p[1] for p in points], dtype=float)
    if len(points) == 1:
        return float(rates[0])
    T_clamped = np.clip(T, maturities.min(), maturities.max())
    return float(np.interp(T_clamped, maturities, rates))


def main():
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
