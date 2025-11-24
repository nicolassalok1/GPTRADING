"""
Small CLI helper to download price history via yfinance and emit CSV to stdout.
Usage:
    python fetch_history_cli.py --ticker AAPL --period 1y --interval 1d
"""

import argparse
import sys
import yfinance as yf


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticker", required=True, help="Ticker symbol (ex: AAPL)")
    parser.add_argument("--period", default="1y", help="History period, default 1y")
    parser.add_argument("--interval", default="1d", help="Sampling interval, default 1d")
    args = parser.parse_args()

    try:
        df = yf.download(
            args.ticker,
            period=args.period,
            interval=args.interval,
            progress=False,
            auto_adjust=False,
            actions=False,
            threads=False,
        )
        if df.empty:
            print("", end="")
            return 2
        # Emit CSV to stdout
        df.to_csv(sys.stdout)
        return 0
    except Exception as exc:  # noqa: BLE001
        sys.stderr.write(f"error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
