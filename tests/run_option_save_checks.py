"""
Generate a JSON report validating that every option payload we support
is persisted (including its misc) via add_option_to_dashboard.
The results are written to tests/results/options_save_results.json.
"""

import ast
import json
import tempfile
from pathlib import Path


def _extract_functions(source_text, func_names):
    tree = ast.parse(source_text)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            segment = ast.get_source_segment(source_text, node)
            if segment:
                segments.append(segment)
    return "\n\n".join(segments)


def load_functions():
    src = Path("streamlit_app.py").read_text()
    code = "\n".join(
        [
            "import json, time, datetime",
            "from pathlib import Path",
            "def floor_3(x):\n    return float(f\"{float(x):.3f}\")",
            _extract_functions(
                src,
                {"load_options_book", "save_options_book", "add_option_to_dashboard"},
            ),
        ]
    )
    ns = {}
    exec(code, ns)  # nosec: executed from local source
    return ns


def build_payloads():
    """Canonical payloads for each supported structure."""
    common = {
        "expiration": "2025-12-31",
        "quantity": 1,
        "avg_price": 1.0,
        "side": "long",
        "underlying": "TST",
    }
    return {
        "asian_arith": {
            **common,
            "product_type": "Asian arithmétique",
            "option_type": "call",
            "strike": 100,
            "misc": {"method": "MC control variate", "n_obs": 20, "n_paths": 20000},
        },
        "asian_geom": {
            **common,
            "product_type": "Asian géométrique",
            "option_type": "put",
            "strike": 90,
            "misc": {"method": "closed_form_geom", "n_obs": 12},
        },
        "digital": {
            **common,
            "product_type": "Digital (cash-or-nothing)",
            "option_type": "call",
            "strike": 100,
            "misc": {"payout": 1.0, "style": "cash_or_nothing"},
        },
        "asset_or_nothing": {
            **common,
            "product_type": "Asset-or-nothing",
            "option_type": "put",
            "strike": 80,
            "misc": {"style": "asset_or_nothing"},
        },
        "forward_start": {
            **common,
            "product_type": "Forward-start",
            "option_type": "call",
            "strike": 100,
            "misc": {"T_start": 0.25, "k_factor": 1.1, "n_paths": 5000, "n_steps": 200},
        },
        "chooser": {
            **common,
            "product_type": "Chooser",
            "option_type": "call",
            "strike": 100,
            "misc": {"t_choice": 0.5},
        },
        "straddle": {
            **common,
            "product_type": "Straddle",
            "option_type": "call",
            "strike": 100,
            "misc": {},
            "legs": [
                {"option_type": "call", "strike": 100},
                {"option_type": "put", "strike": 100},
            ],
        },
        "strangle": {
            **common,
            "product_type": "Strangle",
            "option_type": "call",
            "strike": 95,
            "strike2": 105,
            "misc": {"strike_put": 95, "strike_call": 105, "wing": 10},
        },
        "call_spread": {
            **common,
            "product_type": "Call spread",
            "option_type": "call",
            "strike": 100,
            "strike2": 105,
            "misc": {"width": 5},
        },
        "put_spread": {
            **common,
            "product_type": "Put spread",
            "option_type": "put",
            "strike": 95,
            "strike2": 90,
            "misc": {"width": 5},
        },
        "butterfly": {
            **common,
            "product_type": "Butterfly",
            "option_type": "call",
            "strike": 95,
            "strike2": 105,
            "misc": {"wing": 5},
        },
        "condor": {
            **common,
            "product_type": "Condor",
            "option_type": "call",
            "strike": 90,
            "strike2": 110,
            "misc": {"wing_inner": 5, "wing_outer": 10},
        },
        "iron_bfly": {
            **common,
            "product_type": "Iron Butterfly",
            "option_type": "call",
            "strike": 95,
            "strike2": 105,
            "misc": {"wing": 5, "k_mid": 100},
        },
        "calendar": {
            **common,
            "product_type": "Calendar spread",
            "option_type": "put",
            "strike": 100,
            "misc": {"T_short": 0.5, "T_long": 1.0, "opt_kind": "put"},
        },
        "diagonal": {
            **common,
            "product_type": "Diagonal spread",
            "option_type": "call",
            "strike": 100,
            "strike2": 105,
            "misc": {
                "T_short": 0.5,
                "T_long": 1.0,
                "k_short": 100,
                "k_long": 105,
                "opt_kind": "call",
            },
        },
        "binary_barrier": {
            **common,
            "product_type": "Binary barrier up-out",
            "option_type": "call",
            "strike": 100,
            "misc": {
                "barrier_type": "up",
                "direction": "out",
                "barrier_level": 110,
                "payout": 1.0,
                "n_paths": 5000,
                "n_steps": 200,
            },
        },
        "barrier_up_out": {
            **common,
            "product_type": "Barrier up-and-out",
            "option_type": "call",
            "strike": 100,
            "misc": {"barrier_type": "up", "knock": "out", "barrier_level": 110},
        },
        "barrier_down_out": {
            **common,
            "product_type": "Barrier down-and-out",
            "option_type": "put",
            "strike": 100,
            "misc": {"barrier_type": "down", "knock": "out", "barrier_level": 90},
        },
        "barrier_up_in": {
            **common,
            "product_type": "Barrier up-and-in",
            "option_type": "call",
            "strike": 100,
            "misc": {"barrier_type": "up", "knock": "in", "barrier_level": 110},
        },
        "barrier_down_in": {
            **common,
            "product_type": "Barrier down-and-in",
            "option_type": "put",
            "strike": 100,
            "misc": {"barrier_type": "down", "knock": "in", "barrier_level": 90},
        },
        "lookback_fixed": {
            **common,
            "product_type": "Lookback fixed",
            "option_type": "call",
            "strike": 100,
            "misc": {"n_paths": 5000, "n_steps": 200},
        },
        "cliquet": {
            **common,
            "product_type": "Cliquet / Ratchet",
            "option_type": "call",
            "strike": 100,
            "misc": {"n_periods": 12, "cap": 0.05, "floor": 0.0, "n_paths": 3000},
        },
    }


def main():
    ns = load_functions()
    add_fn = ns["add_option_to_dashboard"]
    load_fn = ns["load_options_book"]

    tmpdir = tempfile.TemporaryDirectory()
    dbdir = Path(tmpdir.name) / "database"
    dbdir.mkdir(exist_ok=True)
    options_path = dbdir / "options_portfolio.json"
    legacy_path = dbdir / "options_book.json"
    options_path.write_text("{}")

    # Redirect storage to temp files
    ns["DB_DIR"] = dbdir
    ns["OPTIONS_BOOK_FILE"] = options_path
    ns["OPTIONS_BOOK_FILE_LEGACY"] = legacy_path

    results = []
    for name, payload in build_payloads().items():
        try:
            option_id = add_fn(dict(payload))
            book = load_fn()
            entry = book[option_id]
            results.append(
                {
                    "name": name,
                    "status": "ok",
                    "id": option_id,
                    "misc_keys": sorted(entry.get("misc", {}).keys()),
                    "strike": entry.get("strike"),
                    "strike2": entry.get("strike2"),
                    "product_type": entry.get("product_type"),
                }
            )
        except Exception as exc:  # pragma: no cover
            results.append({"name": name, "status": "error", "error": str(exc)})

    out_dir = Path("tests/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "options_save_results.json"
    out_file.write_text(json.dumps(results, indent=2))
    print(f"Wrote {len(results)} results to {out_file}")


if __name__ == "__main__":
    main()
