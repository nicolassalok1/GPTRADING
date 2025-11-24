import ast
import json
import tempfile
import unittest
from pathlib import Path


def _extract_functions(source_text, func_names):
    """Return source code for the selected top-level function definitions."""
    tree = ast.parse(source_text)
    segments = []
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in func_names:
            segment = ast.get_source_segment(source_text, node)
            if segment:
                segments.append(segment)
    return "\n\n".join(segments)


class OptionsSaveTests(unittest.TestCase):
    def setUp(self):
        # Load function sources from the main app without executing its UI.
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
        self.ns = {}
        exec(code, self.ns)  # nosec: executed from local source for testing

        # Use a temporary options file to avoid touching real data.
        self.tmpdir = tempfile.TemporaryDirectory()
        dbdir = Path(self.tmpdir.name) / "database"
        dbdir.mkdir(exist_ok=True)
        self.options_path = dbdir / "options_portfolio.json"
        self.legacy_path = dbdir / "options_book.json"
        self.options_path.write_text("{}")

        self.ns["DB_DIR"] = dbdir
        self.ns["OPTIONS_BOOK_FILE"] = self.options_path
        self.ns["OPTIONS_BOOK_FILE_LEGACY"] = self.legacy_path

    def tearDown(self):
        self.tmpdir.cleanup()

    def _save_and_load(self, payload):
        add_fn = self.ns["add_option_to_dashboard"]
        load_fn = self.ns["load_options_book"]
        option_id = add_fn(payload)
        book = load_fn()
        return option_id, book[option_id]

    def test_barrier_misc_saved(self):
        option_id, entry = self._save_and_load(
            {
                "underlying": "ABC",
                "option_type": "call",
                "product_type": "Barrier up-and-out",
                "strike": 100,
                "expiration": "2025-01-01",
                "quantity": 1,
                "avg_price": 2.5,
                "side": "long",
                "misc": {"barrier_type": "up", "knock": "out", "barrier_level": 110},
            }
        )
        self.assertIn("barrier_level", entry["misc"])
        self.assertEqual(entry["misc"]["knock"], "out")
        self.assertAlmostEqual(entry["strike"], 100.0)
        self.assertEqual(entry["option_type"], "call")
        self.assertTrue(option_id in entry["id"])

    def test_asian_arith_misc_saved(self):
        _, entry = self._save_and_load(
            {
                "underlying": "XYZ",
                "option_type": "call",
                "product_type": "Asian arithmétique",
                "strike": 50,
                "expiration": "2025-06-01",
                "quantity": 2,
                "avg_price": 1.1,
                "side": "long",
                "misc": {"method": "MC control variate", "n_obs": 20, "n_paths": 1000},
            }
        )
        self.assertEqual(entry["misc"]["method"], "MC control variate")
        self.assertEqual(entry["misc"]["n_obs"], 20)
        self.assertEqual(entry["quantity"], 2.0)

    def test_strangle_misc_saved(self):
        _, entry = self._save_and_load(
            {
                "underlying": "SPY",
                "option_type": "call",
                "product_type": "Strangle",
                "strike": 95,
                "strike2": 105,
                "expiration": "2025-03-15",
                "quantity": 3,
                "avg_price": 3.0,
                "side": "long",
                "misc": {"strike_put": 95, "strike_call": 105, "wing": 10},
            }
        )
        self.assertEqual(entry["misc"]["wing"], 10)
        self.assertEqual(entry["misc"]["strike_call"], 105)
        self.assertEqual(entry["misc"]["strike_put"], 95)
        self.assertEqual(entry["strike2"], 105.0)

    def test_calendar_misc_saved(self):
        _, entry = self._save_and_load(
            {
                "underlying": "MSFT",
                "option_type": "put",
                "product_type": "Calendar spread",
                "strike": 300,
                "expiration": "2025-09-01",
                "quantity": 1,
                "avg_price": 4.2,
                "side": "short",
                "misc": {"T_short": 0.5, "T_long": 1.0, "opt_kind": "put"},
            }
        )
        self.assertEqual(entry["misc"]["T_short"], 0.5)
        self.assertEqual(entry["side"], "short")

    def test_cliquet_misc_saved(self):
        _, entry = self._save_and_load(
            {
                "underlying": "EURUSD",
                "option_type": "call",
                "product_type": "Cliquet / Ratchet",
                "strike": 1.2,
                "expiration": "2026-01-01",
                "quantity": 5,
                "avg_price": 0.8,
                "side": "long",
                "misc": {"n_periods": 12, "cap": 0.05, "floor": 0.0, "n_paths": 3000},
            }
        )
        self.assertEqual(entry["misc"]["n_periods"], 12)
        self.assertAlmostEqual(entry["misc"]["cap"], 0.05)
        self.assertEqual(entry["quantity"], 5.0)


if __name__ == "__main__":
    unittest.main()
