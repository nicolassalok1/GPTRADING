"""
Utilities to fetch risk-free rates and dividend yield using yfinance.

- get_r(maturity_years): interpolates r(T) from ^IRX (13-week), ^FVX (5Y), ^TNX (10Y).
- get_q(ticker): returns dividend yield for a given equity ticker.
"""

from __future__ import annotations

import os
import time
import math
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
import yfinance as yf
import requests
from pathlib import Path

# Symbols for risk-free proxies
RATE_SYMBOLS: Dict[str, float] = {
    "^IRX": 0.25,  # 13-week T-Bill ≈ 0.25 years
    "^FVX": 5.0,   # 5-year
    "^TNX": 10.0,  # 10-year
}
FRED_SERIES: Dict[str, str] = {
    "^IRX": "DGS3MO",  # 3-month constant maturity
    "^FVX": "DGS5",    # 5-year
    "^TNX": "DGS10",   # 10-year
}

MAX_RETRIES = 3
SLEEP_BETWEEN = 1.0
DEFAULT_RF = float(os.getenv("DEFAULT_RF_RATE", "0.02"))


def _fetch_from_fred(series_id: str) -> float:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
    resp = requests.get(url, timeout=5)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    if len(lines) < 2:
        raise RuntimeError("Réponse FRED vide")
    last_val = lines[-1].split(",")[-1]
    if last_val.strip() == ".":
        raise RuntimeError("Pas de valeur FRED exploitable")
    return float(last_val)


def _fetch_last_close(symbol: str) -> float:
    """
    Fetch the latest close for a given symbol using yfinance,
    retrying on transient failures.
    """
    last_err: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="5d", interval="1d")
            if not hist.empty and "Close" in hist.columns:
                return float(hist["Close"].iloc[-1])
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(SLEEP_BETWEEN)
    if last_err:
        raise RuntimeError(f"Failed to fetch {symbol}: {last_err}") from last_err
    raise RuntimeError(f"Failed to fetch {symbol}: empty history")


def get_r(maturity_years: float) -> float:
    """
    Interpolate risk-free rate r(T) in decimal from ^IRX (≈0.25y), ^FVX (5y), ^TNX (10y).
    Values from yfinance are percentages; convert to decimal before interpolation.
    If USE_STATIC_RF_RATE=1, returns DEFAULT_RF_RATE. Otherwise calls the CLI helper
    fetch_r_cli.py (subprocess) to avoid import side-effects inside Streamlit.
    """
    if os.getenv("USE_STATIC_RF_RATE", "0").lower() in {"1", "true", "yes"}:
        return DEFAULT_RF

    cli_path = Path(__file__).resolve().parent / "fetch_r_cli.py"
    if cli_path.exists():
        try:
            res = subprocess.run(
                [sys.executable, str(cli_path), str(float(maturity_years))],
                capture_output=True,
                text=True,
                check=True,
                timeout=8,
            )
            val = float(res.stdout.strip())
            if math.isfinite(val) and val > 0:
                return val
        except Exception:
            pass

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
        # FRED fallback
        fred_id = FRED_SERIES.get(sym)
        if fred_id:
            try:
                val = _fetch_from_fred(fred_id) / 100.0
                if math.isfinite(val) and val > 0:
                    points.append((mat, val))
            except Exception:
                continue

    if not points:
        return DEFAULT_RF  # fallback

    # Sort by maturity
    points.sort(key=lambda x: x[0])
    maturities = np.array([p[0] for p in points], dtype=float)
    rates = np.array([p[1] for p in points], dtype=float)

    T = float(maturity_years)
    if len(points) == 1:
        val = float(rates[0])
        return val if math.isfinite(val) and val > 0 else DEFAULT_RF
    # Clamp to available range then linear interpolation
    T_clamped = np.clip(T, maturities.min(), maturities.max())
    val = float(np.interp(T_clamped, maturities, rates))
    return val if math.isfinite(val) and val > 0 else DEFAULT_RF


def get_q(ticker: str) -> float:
    """
    Return dividend yield (continuous approx) for the given ticker using yfinance.
    Falls back to 0.0 if unavailable.
    """
    last_err: Exception | None = None
    for _ in range(MAX_RETRIES):
        try:
            info = yf.Ticker(ticker).info or {}
            dy = info.get("dividendYield")
            if dy is None:
                return 0.0
            return float(dy)
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(SLEEP_BETWEEN)
    if last_err:
        raise RuntimeError(f"Failed to fetch dividend yield for {ticker}: {last_err}") from last_err
    return 0.0


if __name__ == "__main__":
    r = get_r(0.5)
    q = get_q("AAPL")
    print(r, q)
