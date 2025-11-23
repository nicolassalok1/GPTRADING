"""
Sample expired options and payoff/PnL calculators for local testing.

This file is standalone and does not depend on Streamlit. It builds a small
set of expired options (various products) and computes payoff and PnL at
expiration. Path-dependent structures (e.g. Asian, barrier) embed their
closing prices in `misc["closing_prices"]` so tests do not require network
access.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

DEFAULT_RF = 0.02  # fallback risk-free rate
DEFAULT_Q = 0.0    # fallback dividend yield


def _geometric_mean(values: List[float]) -> float:
    vals = [v for v in values if v > 0]
    if not vals:
        return 0.0
    log_sum = sum(math.log(v) for v in vals)
    return math.exp(log_sum / len(vals))


def compute_payoff(option: Dict) -> float:
    """
    Compute payoff at expiration for a single option entry.
    Supports vanilla, digital, Asian (arith/geom) and simple barrier knock-out.
    Expects `underlying_close` for vanilla/digital; `misc["closing_prices"]` for path-dependent.
    """
    product = str(option.get("product") or option.get("product_type") or "").lower()
    opt_type = str(option.get("option_type") or option.get("type") or "").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    quantity = float(option.get("quantity", 0) or 0.0)
    side = str(option.get("side", "long")).lower()
    underlying_close = float(option.get("underlying_close", 0.0) or 0.0)
    misc = option.get("misc") or {}
    closing_prices = misc.get("closing_prices") or []

    # Vanilla / digital
    if "digital" in product:
        payout = float(misc.get("payout", 1.0) or 1.0)
        payoff_per_unit = payout if (opt_type == "call" and underlying_close > strike) or (
            opt_type == "put" and underlying_close < strike
        ) else 0.0
    elif "asian" in product:
        if not closing_prices:
            return 0.0
        if "geom" in product:
            average_price = _geometric_mean(closing_prices)
        else:
            average_price = sum(closing_prices) / len(closing_prices)
        if opt_type == "call":
            payoff_per_unit = max(average_price - strike, 0.0)
        else:
            payoff_per_unit = max(strike - average_price, 0.0)
    elif "barrier" in product:
        # Simple knock-out check using path of closing prices.
        barrier = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
        direction = misc.get("knock", misc.get("direction", "out")).lower()
        barrier_type = misc.get("barrier_type", "up").lower()
        prices = closing_prices or [underlying_close]
        hit = False
        for p in prices:
            if barrier_type == "up" and p >= barrier:
                hit = True
                break
            if barrier_type == "down" and p <= barrier:
                hit = True
                break
        knocked_out = direction == "out" and hit
        knocked_in = direction == "in" and hit
        if knocked_out:
            payoff_per_unit = 0.0
        elif direction == "in" and not knocked_in:
            payoff_per_unit = 0.0
        else:
            if opt_type == "call":
                payoff_per_unit = max(underlying_close - strike, 0.0)
            else:
                payoff_per_unit = max(strike - underlying_close, 0.0)
    else:
        # Vanilla fallback
        if opt_type == "put" or product == "put":
            payoff_per_unit = max(strike - underlying_close, 0.0)
        else:
            payoff_per_unit = max(underlying_close - strike, 0.0)

    # PnL based on entry premium
    premium = float(option.get("avg_price", option.get("price", 0.0)) or 0.0)
    if side == "long":
        pnl_per_unit = payoff_per_unit - premium
    else:
        pnl_per_unit = premium - payoff_per_unit

    return pnl_per_unit * quantity


def sample_expired_options() -> Dict[str, Dict]:
    """Return a small dictionary of expired options covering several products."""
    return {
        "vanilla_call": {
            "underlying": "SPY",
            "option_type": "call",
            "product": "vanilla",
            "strike": 450.0,
            "quantity": 2,
            "avg_price": 5.0,
            "underlying_close": 480.0,
            "side": "long",
            "status": "expired",
        },
        "vanilla_put": {
            "underlying": "AAPL",
            "option_type": "put",
            "product": "vanilla",
            "strike": 170.0,
            "quantity": 1,
            "avg_price": 3.0,
            "underlying_close": 150.0,
            "side": "short",
            "status": "expired",
        },
        "asian_arith": {
            "underlying": "SPY",
            "option_type": "call",
            "product": "asian arithmetique",
            "strike": 100.0,
            "quantity": 1,
            "avg_price": 4.7,
            "underlying_close": 105.0,
            "side": "long",
            "status": "expired",
            "misc": {
                "closing_prices": [98, 102, 104, 106, 108],
                "method": "arith_avg",
            },
        },
        "asian_geom": {
            "underlying": "SPY",
            "option_type": "call",
            "product": "asian geometrique",
            "strike": 100.0,
            "quantity": 1,
            "avg_price": 4.7,
            "underlying_close": 105.0,
            "side": "long",
            "status": "expired",
            "misc": {
                "closing_prices": [98, 102, 104, 106, 108],
                "method": "geom_avg",
            },
        },
        "digital": {
            "underlying": "SPY",
            "option_type": "call",
            "product": "digital",
            "strike": 430.0,
            "quantity": 5,
            "avg_price": 1.0,
            "underlying_close": 440.0,
            "side": "long",
            "status": "expired",
            "misc": {"payout": 10.0},
        },
        "barrier_up_out": {
            "underlying": "SPY",
            "option_type": "call",
            "product": "barrier up-and-out",
            "strike": 100.0,
            "quantity": 1,
            "avg_price": 2.0,
            "underlying_close": 105.0,
            "side": "long",
            "status": "expired",
            "misc": {
                "barrier": 120.0,
                "barrier_type": "up",
                "knock": "out",
                "closing_prices": [95, 99, 102, 107, 110],
            },
        },
    }


def main():
    print("=== Sample expired options payoff/PnL ===")
    opts = sample_expired_options()
    for opt_id, opt in opts.items():
        pnl = compute_payoff(opt)
        print(f"{opt_id:20s} -> PnL_total={pnl:.4f}")


if __name__ == "__main__":
    main()
