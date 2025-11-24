"""
Streamlit-based trading and options pricing dashboard.

Responsibilities:
- Render the multi-tab UI (dashboard, trading systems, forwards, options pricing).
- Persist portfolios/systems/options in local JSON files under ./database.
- Call external services: Alpaca (positions/orders/prices), yfinance (price history),
  OpenAI ChatGPT, CBOE, and various pricing libraries (torch, tensorflow, scipy, etc.).

Side effects & assumptions:
- Reads environment variables (e.g., OPENAI_API_KEY) and local JSON files.
- Performs network I/O to market/pricing APIs; failures are surfaced in the UI.
- Writes to disk in the working directory; callers must ensure filesystem access.
"""

import streamlit as st
from openai import OpenAI
import os
import sys
from pathlib import Path
import json
import alpaca_trade_api as tradeapi
import time
from dotenv import load_dotenv
import pandas as pd
import requests
import datetime
import math
import yfinance as yf
import re

# Configuration
APP_DIR = Path(__file__).resolve().parent
DB_DIR = APP_DIR / "database"
DB_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR = APP_DIR / "scripts"
PRICING_DIR = APP_DIR / "pricing_scripts"
DATASETS_DIR = DB_DIR / "data"
DATASETS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PRICING_DIR))
from rates_utils import get_r, get_q
DATA_FILE = DB_DIR / "equities.json"
PORTFOLIO_FILE = DB_DIR / "portfolio.json"
SELL_SYSTEMS_FILE = DB_DIR / "sell_systems.json"
OPTIONS_BOOK_FILE = DB_DIR / "options_portfolio.json"
OPTIONS_BOOK_FILE_LEGACY = DB_DIR / "options_book.json"
OPTIONS_PORTFOLIO_FILE = OPTIONS_BOOK_FILE  # legacy name kept for compatibility
EXPIRED_OPTIONS_FILE = OPTIONS_BOOK_FILE    # legacy name kept for compatibility
CUSTOM_OPTIONS_FILE = DB_DIR / "custom_options.json"
FORWARDS_FILE = DB_DIR / "forwards.json"
LEGACY_EXPIRED_FILE = DB_DIR / "expired_options.json"
load_dotenv()

def run_app_options():
    """
    Render the options pricing tab (legacy monolith) inside the main Streamlit app.

    This function wires pricing models (Longstaff, Heston, lookback, etc.), visualization,
    and the ability to push priced structures into the unified dashboard JSON store.
    Side effects: imports heavy numerical libs, sets TensorFlow flags, reads/writes JSON
    files in ./database, and invokes networked data sources (yfinance, requests).
    """
    import io
    import math
    import os
    import subprocess
    import sys
    from pathlib import Path
    import time
    import re
    import base64
    import datetime

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import requests
    import streamlit as st
    import tensorflow as tf
    import yfinance as yf
    from rates_utils import get_r as get_r_interp, get_q as get_q_yf
    import torch
    import yfinance as yf
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy import stats
    from scipy.interpolate import griddata
    from scipy.linalg import lu_factor, lu_solve
    from scipy.stats import norm
    from typing import Callable

    from Longstaff.option import Option
    from Longstaff.pricing import (
        black_scholes_merton,
        crr_pricing,
        monte_carlo_simulation,
    )
    from Longstaff.process import GeometricBrownianMotion, HestonProcess
    from Lookback.european_call import european_call_option
    from Lookback.lookback_call import lookback_call_option
    from Heston.heston_torch import HestonParams, carr_madan_call_torch

    torch.set_default_dtype(torch.float64)
    HES_DEVICE = torch.device("cpu")
    MIN_IV_MATURITY = 0.1


    PLOTLY_CONFIG = {
        "displaylogo": False,
        "modeBarButtonsToRemove": ["sendDataToCloud"],
    }


    def render_add_to_dashboard_button(
        product_label: str,
        option_char: str,
        price_value: float,
        strike: float | None,
        maturity: float | None,
        key_prefix: str,
        *,
        strike2: float | None = None,
        spot: float | None = None,
        legs: list[dict] | None = None,
        misc: dict | None = None,
        expanded: bool = True,
    ) -> None:
        """Prompt to push a priced structure into the dashboard JSON."""
        if "add_option_to_dashboard" not in globals():
            st.info("Ajout au dashboard indisponible (fonction manquante).")
            return

        with st.expander(f"📥 Ajouter au dashboard ({product_label})", expanded=expanded):
            st.metric("Prix calculé", f"${price_value:.6f}")
            underlying = (
                st.session_state.get("heston_cboe_ticker")
                or st.session_state.get("tkr_common")
                or st.session_state.get("common_underlying")
                or st.session_state.get("ticker_default")
                or ""
            ).strip().upper()
            st.caption(f"Sous-jacent: {underlying or 'N/A'} (reprise de l’entête)")
            today = datetime.date.today()
            expiration_dt = today + datetime.timedelta(days=int((maturity or 0.0) * 365))
            qty = st.number_input("Quantité", min_value=1, value=1, step=1, key=f"{key_prefix}_qty")
            side = st.selectbox("Sens", ["long", "short"], index=0, key=f"{key_prefix}_side")
            strike_val = float(strike if strike is not None else st.session_state.get("common_strike", 0.0))
            strike2_val = float(strike2) if strike2 is not None else None
            st.caption(
                f"K (strike commun): {strike_val:.4f}"
                + (f" | K2: {strike2_val:.4f}" if strike2_val is not None else "")
            )
            if maturity is not None:
                st.caption(f"T (maturité commune, années): {float(maturity):.4f}")

            if st.button("Ajouter au dashboard", key=f"{key_prefix}_add"):
                base_misc = {
                    "structure": product_label,
                    "legs": legs,
                    "strike2": strike2_val,
                    "spot_at_pricing": float(spot or 0.0),
                }
                if isinstance(misc, dict):
                    base_misc.update(misc)
                misc_payload = base_misc
                payload = {
                    "underlying": underlying or "N/A",
                    "option_type": "call" if option_char.lower() == "c" else "put",
                    "product_type": product_label,
                    "type": product_label,
                    "strike": float(strike_val),
                    "strike2": float(strike2_val) if strike2_val is not None else None,
                    "expiration": expiration_dt.isoformat(),
                    "quantity": int(qty),
                    "avg_price": float(price_value),
                    "side": side,
                    "S0": float(spot or 0.0),
                    "maturity_years": maturity,
                    "legs": legs,
                    "T_0": today.isoformat(),
                    "price": float(price_value),
                    "misc": misc_payload,
                }
                try:
                    # Log the JSON write attempt (UI + server log)
                    st.caption(
                        f"[LOG] Écriture vers options_portfolio.json avec payload: {payload}"
                    )
                    print(f"[options] add_to_dashboard payload={payload}")
                    option_id = add_option_to_dashboard(payload)
                    st.success(
                        f"{product_label} ajouté au dashboard (id: {option_id}) "
                        f"et enregistré dans options_portfolio.json."
                    )
                    try:
                        st.rerun()
                    except Exception:
                        pass
                except Exception as exc:  # pragma: no cover - UI feedback
                    st.error(
                        f"Erreur lors de l'ajout au dashboard (écriture JSON) : {exc}"
                    )


    def simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=42):
        """
        Simulate GBM paths under the risk-neutral measure:
            dS_t = (r - q) S_t dt + sigma S_t dW_t
        Returns:
            S : array of shape (M+1, N_paths)
            dt: time step
        """
        dt = T / M
        rng = np.random.default_rng(seed)
        S = np.empty((M + 1, N_paths))
        S[0, :] = S0
        Z = rng.normal(size=(M, N_paths))
        drift = (r - q - 0.5 * sigma**2) * dt
        vol_term = sigma * np.sqrt(dt)
        for t in range(1, M + 1):
            S[t, :] = S[t - 1, :] * np.exp(drift + vol_term * Z[t - 1, :])
        return S, dt


    def price_bermudan_lsmc(
        S0,
        K,
        T,
        r,
        q,
        sigma,
        cpflag="p",
        M=50,
        N_paths=100_000,
        degree=3,
        n_ex_dates=6,
        seed: int = 42,
    ):
        """
        Longstaff–Schwartz Monte Carlo pricing for a Bermudan option
        under risk-neutral GBM (Black–Scholes dynamics).
        """
        S, dt = simulate_gbm_paths(S0, r, q, sigma, T, M, N_paths, seed=seed)
        disc = np.exp(-r * dt)

        if cpflag == "c":
            Y = np.maximum(S - K, 0.0)
        elif cpflag == "p":
            Y = np.maximum(K - S, 0.0)
        else:
            raise ValueError("cpflag must be 'c' or 'p'")

        C = Y[-1, :].copy()
        ex_indices = np.linspace(1, M - 1, max(1, n_ex_dates - 1), dtype=int)
        ex_set = set(ex_indices.tolist())

        for j in range(M - 1, 0, -1):
            C *= disc
            if j in ex_set:
                S_j = S[j, :]
                Y_j = Y[j, :]
                itm = Y_j > 0.0
                if np.any(itm):
                    X = np.vstack([S_j[itm] ** k for k in range(degree + 1)]).T
                    y_reg = C[itm]
                    beta, *_ = np.linalg.lstsq(X, y_reg, rcond=None)
                    X_all = np.vstack([S_j**k for k in range(degree + 1)]).T
                    C_hat = X_all @ beta
                    exercise = (Y_j > C_hat) & itm
                    C[exercise] = Y_j[exercise]

        C *= disc
        price = np.mean(C)
        return float(price)


    # ---------------------------------------------------------------------------
    #  Bermudan / European / American + Barrier (Crank–Nicolson)
    # ---------------------------------------------------------------------------

    class CrankNicolsonBS:
        """
        Solveur Crank–Nicolson pour la PDE de Black–Scholes en log(S).

        Typeflag:
            'Eu'  : option européenne
            'Am'  : option américaine (exercice possible à chaque date de grille)
            'Bmd' : option bermudéenne (exercice possible à certaines dates)
        cpflag:
            'c' : call
            'p' : put
        """

        def __init__(
            self,
            Typeflag: str,
            cpflag: str,
            S0: float,
            K: float,
            T: float,
            vol: float,
            r: float,
            d: float,
            n_spatial: int = 500,
            n_time: int = 600,
            exercise_step: int | None = None,
            n_exercise_dates: int | None = None,
            **_,
        ) -> None:
            self.Typeflag = Typeflag
            self.cpflag = cpflag
            self.S0 = float(S0)
            self.K = float(K)
            self.T = float(T)
            self.vol = float(vol)
            self.r = float(r)
            self.d = float(d)

            self.n_spatial = max(50, int(n_spatial))
            self.n_time = max(50, int(n_time))

            # Deux modes possibles pour la Bermudane :
            # - exercise_step       : exercice tous les 'exercise_step' pas
            # - n_exercise_dates    : nb de dates d'exercice (incluant T)
            # Si les deux sont donnés -> erreur, c'est ambigu.
            if exercise_step is not None and n_exercise_dates is not None:
                raise ValueError(
                    "Spécifie soit exercise_step, soit n_exercise_dates, pas les deux."
                )

            self.exercise_step = int(exercise_step) if exercise_step is not None else None
            self.n_exercise_dates = (
                int(n_exercise_dates) if n_exercise_dates is not None else None
            )

        # -------------------- utils --------------------

        def _resolve_params(
            self,
            Typeflag: str | None,
            cpflag: str | None,
            S0: float | None,
            K: float | None,
            T: float | None,
            vol: float | None,
            r: float | None,
            d: float | None,
        ):
            """Résout les paramètres effectifs sans casser les valeurs 0 éventuelles."""

            Typeflag = self.Typeflag if Typeflag is None else Typeflag
            cpflag = self.cpflag if cpflag is None else cpflag
            S0 = self.S0 if S0 is None else float(S0)
            K = self.K if K is None else float(K)
            T = self.T if T is None else float(T)
            vol = self.vol if vol is None else float(vol)
            r = self.r if r is None else float(r)
            d = self.d if d is None else float(d)
            return Typeflag, cpflag, S0, K, T, vol, r, d

        # -------------------- solveur CN --------------------

        def CN_option_info(
            self,
            Typeflag: str | None = None,
            cpflag: str | None = None,
            S0: float | None = None,
            K: float | None = None,
            T: float | None = None,
            vol: float | None = None,
            r: float | None = None,
            d: float | None = None,
        ) -> tuple[float, float, float, float]:
            """
            Résout la PDE et retourne (Price, Delta, Gamma, Theta).
            """

            Typeflag, cpflag, S0, K, T, vol, r, d = self._resolve_params(
                Typeflag, cpflag, S0, K, T, vol, r, d
            )

            Typeflag = Typeflag.strip()
            cpflag = cpflag.strip()
            if Typeflag not in {"Eu", "Am", "Bmd"}:
                raise ValueError("Typeflag doit être 'Eu', 'Am' ou 'Bmd'.")
            if cpflag not in {"c", "p"}:
                raise ValueError("cpflag doit être 'c' ou 'p'.")

            # Cas trivial T=0
            if T <= 0.0 or self.n_time <= 0:
                payoff0 = max(S0 - K, 0.0) if cpflag == "c" else max(K - S0, 0.0)
                return float(payoff0), 0.0, 0.0, 0.0

            if Typeflag == "Bmd":
                M_lsmc = max(1, min(self.n_time, 50))
                N_paths = 50_000
                n_ex_dates = self.n_exercise_dates or 6
                seed_base = 12345

                def _lsmc_price(s0_val: float, t_val: float) -> float:
                    return price_bermudan_lsmc(
                        S0=s0_val,
                        K=K,
                        T=max(t_val, 1e-6),
                        r=r,
                        q=d,
                        sigma=vol,
                        cpflag=cpflag,
                        M=M_lsmc,
                        N_paths=N_paths,
                        degree=3,
                        n_ex_dates=n_ex_dates,
                        seed=seed_base,
                    )

                price_bmd = _lsmc_price(S0, T)

                bump_s = max(1e-4, 0.01 * S0)
                price_up = _lsmc_price(S0 + bump_s, T)
                price_down = _lsmc_price(max(S0 - bump_s, 1e-6), T)
                delta = (price_up - price_down) / (2.0 * bump_s)
                gamma = (price_up - 2.0 * price_bmd + price_down) / (bump_s**2)

                theta = 0.0
                theta_h = min(max(1.0 / 365.0, 0.01 * T), max(T / 2.0, 1e-6))
                if T > theta_h:
                    price_short = _lsmc_price(S0, T - theta_h)
                    theta = (price_short - price_bmd) / theta_h

                return float(price_bmd), float(delta), float(gamma), float(theta)

            # ----- Grille en log(S) -----
            mu = r - d - 0.5 * vol * vol
            x_max = vol * np.sqrt(max(T, 1e-8)) * 5.0
            n_points = self.n_spatial
            dx = 2.0 * x_max / n_points

            X = np.linspace(-x_max, x_max, n_points + 1)
            max_log = np.log(np.finfo(float).max / max(S0, 1e-12))
            X_clipped = np.clip(X, -max_log, max_log)
            s_grid = S0 * np.exp(X_clipped)

            n_index = np.arange(0, n_points + 1)

            n_time = self.n_time
            dt = T / n_time

            a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
            b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
            c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

            main_diag_A = 1.0 - b - 2.0 * a
            upper_A = a + c
            lower_A = a - c

            main_diag_B = 1.0 + b + 2.0 * a
            upper_B = -a - c
            lower_B = -a + c

            A = np.zeros((n_points + 1, n_points + 1))
            B = np.zeros((n_points + 1, n_points + 1))

            np.fill_diagonal(A, main_diag_A)
            np.fill_diagonal(A[1:], lower_A[:-1])
            np.fill_diagonal(A[:, 1:], upper_A[:-1])
            A = np.nan_to_num(A, nan=0.0, posinf=1e6, neginf=-1e6)

            np.fill_diagonal(B, main_diag_B)
            np.fill_diagonal(B[1:], lower_B[:-1])
            np.fill_diagonal(B[:, 1:], upper_B[:-1])
            B = np.nan_to_num(B, nan=0.0, posinf=1e6, neginf=-1e6)

            lu_factor_A = lu_factor(A)

            # Payoff terminal
            if cpflag == "c":
                values = np.maximum(s_grid - K, 0.0)
            else:
                values = np.maximum(K - s_grid, 0.0)

            payoff = values.copy()
            values_prev_time = values.copy()

            S_max = s_grid[-1]
            S_min = s_grid[0]  # pas utilisé mais dispo si besoin

            # ----- Boucle backward -----
            for time_index in range(n_time):
                # Sauvegarde pour theta (un seul pas suffit)
                if time_index == n_time - 1:
                    values_prev_time = values.copy()

                rhs = B.dot(values)
                values = lu_solve(lu_factor_A, rhs)

                t_now = T - (time_index + 1) * dt
                tau = T - t_now  # temps restant à maturité

                # Conditions aux bords
                if cpflag == "c":
                    values[0] = 0.0
                    values[-1] = S_max - K * np.exp(-r * tau)
                else:
                    values[0] = K * np.exp(-r * tau)
                    values[-1] = 0.0

                # Gestion du style
                if Typeflag == "Am":
                    values = np.maximum(values, payoff)
                elif Typeflag == "Eu":
                    pass

            # ----- Grecs par différences finies -----
            middle_index = n_points // 2
            price = values[middle_index]

            s_plus = S0 * np.exp(dx)
            s_minus = S0 * np.exp(-dx)

            v_plus = values[middle_index + 1]
            v_0 = values[middle_index]
            v_minus = values[middle_index - 1]

            delta = (v_plus - v_minus) / (s_plus - s_minus)

            dVdS_plus = (v_plus - v_0) / (s_plus - S0)
            dVdS_minus = (v_0 - v_minus) / (S0 - s_minus)
            gamma = (dVdS_plus - dVdS_minus) / ((s_plus - s_minus) / 2.0)

            theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

            return float(price), float(delta), float(gamma), float(theta)


    def CN_Barrier_option(Typeflag, cpflag, S0, K, Hu, Hd, T, vol, r, d):
        """
        Pricing d'une option barrière par Crank–Nicolson.
        """

        mu = r - d - 0.5 * vol * vol
        x_max = vol * np.sqrt(T) * 5
        n_points = 500
        dx = 2 * x_max / n_points
        X = np.linspace(-x_max, x_max, n_points + 1)
        n_index = np.arange(0, n_points + 1)

        n_time = 600
        dt = T / n_time

        a = 0.25 * dt * ((vol**2) * (n_index**2) - mu * n_index)
        b = -0.5 * dt * ((vol**2) * (n_index**2) + r)
        c = 0.25 * dt * ((vol**2) * (n_index**2) + mu * n_index)

        main_diag_A = 1 - b - 2 * a
        upper_A = a + c
        lower_A = a - c

        main_diag_B = 1 + b + 2 * a
        upper_B = -a - c
        lower_B = -a + c

        A = np.zeros((n_points + 1, n_points + 1))
        B = np.zeros((n_points + 1, n_points + 1))

        np.fill_diagonal(A, main_diag_A)
        np.fill_diagonal(A[1:], lower_A[:-1])
        np.fill_diagonal(A[:, 1:], upper_A[:-1])

        np.fill_diagonal(B, main_diag_B)
        np.fill_diagonal(B[1:], lower_B[:-1])
        np.fill_diagonal(B[:, 1:], upper_B[:-1])

        Ainv = np.linalg.inv(A)

        s_grid = S0 * np.exp(X)
        if cpflag == "c":
            values = np.clip(s_grid - K, 0, 1e10)
        elif cpflag == "p":
            values = np.clip(K - s_grid, 0, 1e10)
        else:
            raise ValueError("cpflag doit être 'c' ou 'p'.")

        typeflag = Typeflag.upper()
        if typeflag in {"UNO", "UO"}:
            values = np.where(s_grid < Hu, values, 0.0)
        elif typeflag == "DNO":
            values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
        elif typeflag in {"DO"}:
            values = np.where(s_grid > Hd, values, 0.0)
        else:
            raise ValueError("Typeflag doit être 'UNO', 'UO', 'DO' ou 'DNO'.")

        values_prev_time = values.copy()

        for time_index in range(n_time):
            if time_index == n_time - 1:
                values_prev_time = values.copy()

            values = B.dot(values)
            values = Ainv.dot(values)

            s_grid = S0 * np.exp(X)
            if typeflag in {"UNO", "UO"}:
                values = np.where(s_grid < Hu, values, 0.0)
            elif typeflag == "DNO":
                values = np.where((s_grid > Hd) & (s_grid < Hu), values, 0.0)
            elif typeflag == "DO":
                values = np.where(s_grid > Hd, values, 0.0)

        middle_index = n_points // 2
        price = values[middle_index]

        s_plus = S0 * np.exp(dx)
        s_minus = S0 * np.exp(-dx)

        delta = (values[middle_index + 1] - values[middle_index - 1]) / (s_plus - s_minus)

        d_value_d_s_plus = (values[middle_index + 1] - values[middle_index]) / (s_plus - S0)
        d_value_d_s_minus = (values[middle_index] - values[middle_index - 1]) / (S0 - s_minus)
        gamma = (d_value_d_s_plus - d_value_d_s_minus) / ((s_plus - s_minus) / 2.0)

        theta = -(values[middle_index] - values_prev_time[middle_index]) / dt

        return float(price), float(delta), float(gamma), float(theta)


    # ---------------------------------------------------------------------------
    #  Helper Longstaff–Schwartz qui retourne le prix (version locale)
    # ---------------------------------------------------------------------------


    def longstaff_schwartz_price(option: Option, process, n_paths: int, n_steps: int) -> float:
        """
        Implémentation locale de l'algorithme LS, basée sur Longstaff/pricing.py,
        mais qui renvoie le prix comme float.
        """
        from numpy.polynomial import Polynomial

        simulated_paths = process.simulate(s0=option.s0, v0=option.v0, T=option.T, n=n_paths, m=n_steps)
        payoffs = option.payoff(s=simulated_paths)

        continuation_values = np.zeros_like(payoffs)
        continuation_values[-1] = payoffs[-1]

        dt = option.T / n_steps
        discount = np.exp(-process.mu * dt)

        for time_index in range(n_steps - 1, 0, -1):
            polynomial = Polynomial.fit(simulated_paths[time_index], discount * continuation_values[time_index + 1], 5)
            continuation = polynomial(simulated_paths[time_index])
            continuation_values[time_index] = np.where(
                payoffs[time_index] > continuation,
                payoffs[time_index],
                discount * continuation_values[time_index + 1],
            )

        price = discount * np.mean(continuation_values[1])
        return float(price)


    # ---------------------------------------------------------------------------
    #  Outils pour les heatmaps européennes
    # ---------------------------------------------------------------------------

    HEATMAP_GRID_SIZE = 11


    def _heatmap_axis(center: float, span: float, n_points: int = HEATMAP_GRID_SIZE) -> np.ndarray:
        lower = max(0.01, center - span)
        upper = max(lower, center + span)
        if np.isclose(lower, upper) or n_points == 1:
            return np.array([lower])
        return np.linspace(lower, upper, n_points)


    def _render_heatmap(
        matrix: np.ndarray,
        x_values: np.ndarray,
        y_values: np.ndarray,
        title: str,
        xlabel: str = "Spot",
        ylabel: str = "Strike",
    ) -> None:
        # Wrap every heatmap in an expander to avoid flooding the UI
        with st.expander(f"Afficher la heatmap : {title}", expanded=False):
            fig, ax = plt.subplots()
            image = ax.imshow(
                matrix,
                origin="lower",
                aspect="auto",
                extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
                cmap="viridis",
            )
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
            st.pyplot(fig)
            plt.close(fig)


    def _render_call_put_heatmaps(
        label: str, call_matrix: np.ndarray, put_matrix: np.ndarray, x_values: np.ndarray, y_values: np.ndarray
    ) -> None:
        col_call, col_put = st.columns(2)
        with col_call:
            st.write(f"Heatmap Call ({label})")
            _render_heatmap(call_matrix, x_values, y_values, f"Call ({label})")
        with col_put:
            st.write(f"Heatmap Put ({label})")
            _render_heatmap(put_matrix, x_values, y_values, f"Put ({label})")


    def _compute_bsm_heatmaps(
        s_values: np.ndarray, k_values: np.ndarray, maturity: float, rate: float, sigma: float
    ) -> tuple[np.ndarray, np.ndarray]:
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)
        for i, strike in enumerate(k_values):
            for j, spot in enumerate(s_values):
                option_call = Option(s0=spot, T=maturity, K=strike, call=True)
                option_put = Option(s0=spot, T=maturity, K=strike, call=False)
                call_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_call)
                put_matrix[i, j] = black_scholes_merton(r=rate, sigma=sigma, option=option_put)
        return call_matrix, put_matrix


    def _compute_mc_heatmaps(
        s_values: np.ndarray,
        k_values: np.ndarray,
        maturity: float,
        mu: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        process = GeometricBrownianMotion(mu=mu, sigma=sigma)
        discount = np.exp(-mu * maturity)
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)

        for j, spot in enumerate(s_values):
            simulated_paths = process.simulate(s0=spot, T=maturity, n=n_paths, m=n_steps, v0=None)
            terminal_prices = simulated_paths[-1]
            for i, strike in enumerate(k_values):
                call_matrix[i, j] = np.mean(np.maximum(terminal_prices - strike, 0)) * discount
                put_matrix[i, j] = np.mean(np.maximum(strike - terminal_prices, 0)) * discount

        return call_matrix, put_matrix


    def _vanilla_price_with_dividend(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        if T <= 0 or sigma <= 0 or K <= 0 or S0 <= 0:
            intrinsic = max(S0 - K, 0.0) if option_type.lower() in {"call", "c"} else max(K - S0, 0.0)
            return float(intrinsic)
        sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
        d2 = d1 - sqrt_T
        if option_type.lower() in {"call", "c"}:
            price = S0 * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S0 * np.exp(-dividend * T) * norm.cdf(-d1)
        return float(max(price, 0.0))


    def _digital_cash_or_nothing_price(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        payout: float = 1.0,
    ) -> float:
        if T <= 0 or sigma <= 0 or payout <= 0:
            return 0.0
        sqrt_T = sigma * np.sqrt(T)
        d2 = (np.log(S0 / K) + (r - dividend - 0.5 * sigma * sigma) * T) / sqrt_T
        disc = payout * np.exp(-r * T)
        if option_type.lower() in {"call", "c"}:
            return float(disc * norm.cdf(d2))
        return float(disc * norm.cdf(-d2))


    def _asset_or_nothing_price(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        if T <= 0 or sigma <= 0 or S0 <= 0:
            return 0.0
        sqrt_T = sigma * np.sqrt(T)
        d1 = (np.log(S0 / K) + (r - dividend + 0.5 * sigma * sigma) * T) / sqrt_T
        disc = S0 * np.exp(-dividend * T)
        if option_type.lower() in {"call", "c"}:
            return float(disc * norm.cdf(d1))
        return float(disc * norm.cdf(-d1))


    def _chooser_option_price(
        S0: float,
        K: float,
        T: float,
        t_choice: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        # Formula: call(T,K) + put(t_choice, K*exp(- (r - dividend)*(T - t_choice)))
        call_price = _vanilla_price_with_dividend("call", S0, K, T, r, dividend, sigma)
        tau = max(0.0, T - t_choice)
        if tau <= 0:
            return call_price
        K_adj = K * np.exp(-r * (T - t_choice))
        sqrt_tau = sigma * np.sqrt(tau)
        d1 = (np.log(S0 / K_adj) + (r - dividend + 0.5 * sigma * sigma) * tau) / sqrt_tau
        d2 = d1 - sqrt_tau
        put_piece = K_adj * np.exp(-r * t_choice) * norm.cdf(-d2) - S0 * np.exp(-dividend * t_choice) * norm.cdf(-d1)
        return float(max(call_price + put_piece, 0.0))


    def _forward_start_price_mc(
        S0: float,
        r: float,
        q: float,
        sigma: float,
        T_start: float,
        T_end: float,
        k: float,
        n_paths: int = 5000,
        n_steps: int = 200,
        option_type: str = "call",
    ) -> float:
        if T_end <= T_start or sigma <= 0 or n_paths <= 0 or n_steps <= 0:
            return 0.0
        dt = T_end / n_steps
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt)
        disc = np.exp(-r * T_end)
        payoffs = []
        step_choice = int(T_start / dt)
        for _ in range(n_paths):
            s = S0
            s_start = None
            for step in range(n_steps):
                z = np.random.normal()
                s *= np.exp(drift + diff * z)
                if s_start is None and step >= step_choice:
                    s_start = s
            s_start = s if s_start is None else s_start
            strike_dyn = k * s_start
            if option_type.lower() in {"call", "c"}:
                payoff = max(s - strike_dyn, 0.0)
            else:
                payoff = max(strike_dyn - s, 0.0)
            payoffs.append(payoff)
        return float(disc * np.mean(payoffs))


    def _binary_barrier_mc(
        option_type: str,
        barrier_type: str,
        direction: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        payout: float,
        n_paths: int,
        n_steps: int,
    ) -> float:
        if payout <= 0 or barrier <= 0 or n_paths <= 0 or n_steps <= 0:
            return 0.0
        dt = T / n_steps
        drift = (r - dividend - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt)
        disc = np.exp(-r * T)
        hits = []
        for _ in range(n_paths):
            s = S0
            touched = False
            for _ in range(n_steps):
                z = np.random.normal()
                s *= np.exp(drift + diff * z)
                if (barrier_type == "up" and s >= barrier) or (barrier_type == "down" and s <= barrier):
                    touched = True
                    break
            if direction == "out":
                pay = 0.0 if touched else payout
            else:  # in
                pay = payout if touched else 0.0
            # Optional vanilla style digital with strike K at maturity if not handling knock condition
            if pay == 0.0:
                if option_type.lower() in {"call", "c"}:
                    pay = payout if s >= K else 0.0
                else:
                    pay = payout if s <= K else 0.0
            hits.append(pay)
        return float(disc * np.mean(hits))


    def _cliquet_mc(
        S0: float,
        r: float,
        q: float,
        sigma: float,
        T: float,
        n_periods: int,
        cap: float,
        floor: float,
        n_paths: int = 2000,
        seed: int | None = None,
    ) -> float:
        if n_periods <= 0 or n_paths <= 0 or T <= 0:
            return 0.0
        rng = np.random.default_rng(seed)
        dt = T / n_periods
        drift = (r - q - 0.5 * sigma * sigma) * dt
        diff = sigma * np.sqrt(dt)
        disc = np.exp(-r * T)
        payoffs = []
        for _ in range(n_paths):
            s = S0
            coupons = []
            for _ in range(n_periods):
                z = rng.normal()
                s_next = s * np.exp(drift + diff * z)
                ret = (s_next / s) - 1.0
                coupons.append(np.clip(ret, floor, cap))
                s = s_next
            payoffs.append(sum(coupons))
        return float(disc * np.mean(payoffs))


    def _quanto_vanilla_price(
        option_type: str,
        S0: float,
        K: float,
        T: float,
        r_dom: float,
        q_for: float,
        sigma_asset: float,
        sigma_fx: float,
        rho: float,
    ) -> float:
        # Simple quanto adjustment on dividend: q* = q_for + rho*sigma_S*sigma_FX
        q_adj = q_for + rho * sigma_asset * sigma_fx
        return _vanilla_price_with_dividend(option_type, S0, K, T, r_dom, q_adj, sigma_asset)


    def _rainbow_two_asset_mc(
        payoff_on: str,
        S0_a: float,
        S0_b: float,
        sigma_a: float,
        sigma_b: float,
        rho: float,
        K: float,
        T: float,
        r: float,
        q_a: float,
        q_b: float,
        n_paths: int = 5000,
        n_steps: int = 200,
        option_type: str = "call",
    ) -> float:
        if n_paths <= 0 or n_steps <= 0 or T <= 0:
            return 0.0
        dt = T / n_steps
        disc = np.exp(-r * T)
        payoff_list = []
        for _ in range(n_paths):
            s_a, s_b = S0_a, S0_b
            for _ in range(n_steps):
                z1 = np.random.normal()
                z2 = np.random.normal()
                z_b = rho * z1 + np.sqrt(max(0.0, 1 - rho**2)) * z2
                s_a *= np.exp((r - q_a - 0.5 * sigma_a * sigma_a) * dt + sigma_a * np.sqrt(dt) * z1)
                s_b *= np.exp((r - q_b - 0.5 * sigma_b * sigma_b) * dt + sigma_b * np.sqrt(dt) * z_b)
            if payoff_on == "max":
                s_star = max(s_a, s_b)
            else:
                s_star = min(s_a, s_b)
            if option_type.lower() in {"call", "c"}:
                payoff = max(s_star - K, 0.0)
            else:
                payoff = max(K - s_star, 0.0)
            payoff_list.append(payoff)
        return float(disc * np.mean(payoff_list))

    def _barrier_closed_form_price(
        option_type: str,
        barrier_type: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        if barrier <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("Paramètres invalides pour la formule fermée barrière.")
        if barrier_type == "up" and S0 >= barrier:
            return 0.0
        if barrier_type == "down" and S0 <= barrier:
            return 0.0

        option_flag = option_type.lower()
        phi = 1.0 if option_flag in {"call", "c"} else -1.0
        eta = 1.0 if barrier_type == "down" else -1.0
        mu = (r - dividend - 0.5 * sigma * sigma) / (sigma * sigma)
        sigma_sqrt_T = sigma * np.sqrt(T)
        if sigma_sqrt_T == 0:
            return 0.0
        x1 = (np.log(S0 / K) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
        y1 = (np.log((barrier * barrier) / (S0 * K)) / sigma_sqrt_T) + (1.0 + mu) * sigma_sqrt_T
        power1 = (barrier / S0) ** (2.0 * (mu + eta))
        power2 = (barrier / S0) ** (2.0 * mu)
        term1 = phi * S0 * np.exp(-dividend * T) * (norm.cdf(phi * x1) - power1 * norm.cdf(eta * y1))
        term2 = phi * K * np.exp(-r * T) * (norm.cdf(phi * x1 - phi * sigma_sqrt_T) - power2 * norm.cdf(eta * y1 - eta * sigma_sqrt_T))
        price = term1 - term2
        return max(float(price), 0.0)


    def _knock_in_closed_form_price(
        option_type: str,
        barrier_type: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> float:
        vanilla = _vanilla_price_with_dividend(
            option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma
        )
        barrier_out_price = _barrier_closed_form_price(
            option_type=option_type,
            barrier_type=barrier_type,
            S0=S0,
            K=K,
            barrier=barrier,
            T=T,
            r=r,
            dividend=dividend,
            sigma=sigma,
        )
        return max(vanilla - barrier_out_price, 0.0)


    def _barrier_monte_carlo_price(
        option_type: str,
        barrier_type: str,
        S0: float,
        K: float,
        barrier: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        n_paths: int,
        n_steps: int,
        knock_in: bool = False,
    ) -> float:
        if barrier <= 0 or n_paths <= 0 or n_steps <= 0:
            raise ValueError("Paramètres invalides pour le Monte Carlo barrière.")
        option_type_lower = option_type.lower()
        if barrier_type == "up" and S0 >= barrier:
            if knock_in:
                return _vanilla_price_with_dividend(option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma)
            return 0.0
        if barrier_type == "down" and S0 <= barrier:
            if knock_in:
                return _vanilla_price_with_dividend(option_type=option_type, S0=S0, K=K, T=T, r=r, dividend=dividend, sigma=sigma)
            return 0.0
        dt = T / n_steps
        drift = (r - dividend - 0.5 * sigma * sigma) * dt
        diffusion = sigma * np.sqrt(dt)
        discount = np.exp(-r * T)
        payoffs = []
        for _ in range(n_paths):
            s = S0
            barrier_hit = False
            for _ in range(n_steps):
                z = np.random.normal()
                s *= np.exp(drift + diffusion * z)
                if barrier_type == "up" and s >= barrier:
                    barrier_hit = True
                    if not knock_in:
                        break
                elif barrier_type == "down" and s <= barrier:
                    barrier_hit = True
                    if not knock_in:
                        break
            if knock_in and not barrier_hit:
                payoffs.append(0.0)
                continue
            if not knock_in and barrier_hit:
                payoffs.append(0.0)
                continue
            if option_type_lower in {"call", "c"}:
                payoff = max(s - K, 0.0)
            else:
                payoff = max(K - s, 0.0)
            payoffs.append(payoff)
        return discount * (float(np.mean(payoffs)) if payoffs else 0.0)


    def _render_barrier_stock_paths(
        S0: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
        barrier: float,
        barrier_type: str,
        n_steps: int,
        title_suffix: str,
    ):
        """Display a few GBM trajectories with the active barrier level overlaid."""
        n_steps = max(5, int(n_steps))
        dt = T / n_steps if n_steps > 0 else T
        times = np.linspace(0.0, T, n_steps + 1)
        n_paths_plot = 5
        paths = np.empty((n_paths_plot, n_steps + 1))

        drift = (r - dividend - 0.5 * sigma * sigma) * dt
        vol_step = sigma * np.sqrt(dt)

        for i in range(n_paths_plot):
            shocks = np.random.normal(size=n_steps)
            log_path = np.empty(n_steps + 1)
            log_path[0] = np.log(S0)
            log_path[1:] = log_path[0] + np.cumsum(drift + vol_step * shocks)
            paths[i] = np.exp(log_path)

        fig, ax = plt.subplots(figsize=(7, 3))
        for i in range(n_paths_plot):
            ax.plot(times, paths[i], alpha=0.65, linewidth=1.4)

        is_up = barrier_type == "up"
        color = "crimson" if is_up else "steelblue"
        label = "Barrière haute" if is_up else "Barrière basse"
        ax.axhline(barrier, color=color, linestyle="--", linewidth=2.0, label=f"{label} = {barrier:.2f}")
        ax.set_xlabel("Temps (années)")
        ax.set_ylabel("Sous-jacent simulé")
        ax.set_title(f"Trajectoires simulées + barrière ({title_suffix})")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)


    def _compute_barrier_heatmap_matrix(
        option_type: str,
        barrier_type: str,
        strike_values: np.ndarray,
        offset_values: np.ndarray,
        S0: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix = np.zeros((len(strike_values), len(offset_values)))
        ratio_axis = np.zeros(len(offset_values))

        for j, offset in enumerate(offset_values):
            if barrier_type == "up":
                ratio = 1.1 + offset
            else:
                ratio = max(0.01, 0.9 - offset)
            ratio_axis[j] = ratio

            for i, strike in enumerate(strike_values):
                barrier = max(strike * ratio, 0.01)
                try:
                    price = _barrier_closed_form_price(
                        option_type=option_type,
                        barrier_type=barrier_type,
                        S0=S0,
                        K=float(strike),
                        barrier=float(barrier),
                        T=T,
                        r=r,
                        dividend=dividend,
                        sigma=sigma,
                    )
                except ValueError:
                    price = 0.0
                matrix[i, j] = price

        if np.any(np.diff(ratio_axis) < 0):
            order = np.argsort(ratio_axis)
            ratio_axis = ratio_axis[order]
            matrix = matrix[:, order]

        return matrix, ratio_axis


    def _compute_up_and_out_strike_heatmap(
        option_type: str,
        barrier: float,
        strike_values: np.ndarray,
        maturity_values: np.ndarray,
        spot: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> np.ndarray:
        """
        Construit une matrice de prix up-and-out selon (T, K) pour un spot fixe.
        """
        matrix = np.zeros((len(maturity_values), len(strike_values)))
        for i, maturity in enumerate(maturity_values):
            for j, strike in enumerate(strike_values):
                if strike <= 0.0:
                    matrix[i, j] = 0.0
                    continue
                try:
                    price = _barrier_closed_form_price(
                        option_type=option_type,
                        barrier_type="up",
                        S0=float(spot),
                        K=float(strike),
                        barrier=float(barrier),
                        T=float(maturity),
                        r=r,
                        dividend=dividend,
                        sigma=sigma,
                    )
                except ValueError:
                    price = 0.0
                matrix[i, j] = price
        return matrix


    def _compute_up_and_in_strike_heatmap(
        option_type: str,
        barrier: float,
        strike_values: np.ndarray,
        maturity_values: np.ndarray,
        spot: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> np.ndarray:
        matrix = np.zeros((len(maturity_values), len(strike_values)))
        for i, maturity in enumerate(maturity_values):
            for j, strike in enumerate(strike_values):
                if strike <= 0.0:
                    matrix[i, j] = 0.0
                    continue
                vanilla = _vanilla_price_with_dividend(option_type, spot, float(strike), float(maturity), r, dividend, sigma)
                try:
                    barrier_out = _barrier_closed_form_price(
                        option_type=option_type,
                        barrier_type="up",
                        S0=float(spot),
                        K=float(strike),
                        barrier=float(barrier),
                        T=float(maturity),
                        r=r,
                        dividend=dividend,
                        sigma=sigma,
                    )
                except ValueError:
                    matrix[i, j] = 0.0
                    continue
                matrix[i, j] = max(vanilla - barrier_out, 0.0)
        return matrix


    def _compute_lookback_exact_heatmap(
        s_values: np.ndarray,
        t_values: np.ndarray,
        t_current: float,
        rate: float,
        sigma: float,
    ) -> np.ndarray:
        matrix = np.zeros((len(t_values), len(s_values)))
        for i, maturity in enumerate(t_values):
            for j, spot in enumerate(s_values):
                lookback_opt = lookback_call_option(
                    T=float(maturity), t=float(t_current), S0=float(spot), r=float(rate), sigma=float(sigma)
                )
                matrix[i, j] = lookback_opt.price_exact()
        return matrix


    def _compute_lookback_mc_heatmap(
        s_values: np.ndarray,
        t_values: np.ndarray,
        t_current: float,
        rate: float,
        sigma: float,
        n_iters: int,
    ) -> np.ndarray:
        matrix = np.zeros((len(t_values), len(s_values)))
        for i, maturity in enumerate(t_values):
            for j, spot in enumerate(s_values):
                lookback_opt = lookback_call_option(
                    T=float(maturity), t=float(t_current), S0=float(spot), r=float(rate), sigma=float(sigma)
                )
                matrix[i, j] = lookback_opt.price_monte_carlo(n_iters)
        return matrix


    def _compute_down_in_heatmap(
        option_type: str,
        strike_values: np.ndarray,
        offset_values: np.ndarray,
        S0: float,
        T: float,
        r: float,
        dividend: float,
        sigma: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        matrix_out, ratio_axis = _compute_barrier_heatmap_matrix(
            option_type=option_type,
            barrier_type="down",
            strike_values=strike_values,
            offset_values=offset_values,
            S0=S0,
            T=T,
            r=r,
            dividend=dividend,
            sigma=sigma,
        )
        matrix_in = np.zeros_like(matrix_out)
        for i, strike in enumerate(strike_values):
            vanilla = _vanilla_price_with_dividend(
                option_type=option_type, S0=S0, K=float(strike), T=T, r=r, dividend=dividend, sigma=sigma
            )
            matrix_in[i, :] = np.maximum(vanilla - matrix_out[i, :], 0.0)
        return matrix_in, ratio_axis


    def _compute_american_ls_heatmaps(
        s_values: np.ndarray,
        k_values: np.ndarray,
        maturity: float,
        process,
        n_paths: int,
        n_steps: int,
        v0=None,
    ) -> tuple[np.ndarray, np.ndarray]:
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)
        for i, strike in enumerate(k_values):
            for j, spot in enumerate(s_values):
                option_call = Option(s0=spot, T=maturity, K=strike, v0=v0, call=True)
                option_put = Option(s0=spot, T=maturity, K=strike, v0=v0, call=False)
                call_matrix[i, j] = longstaff_schwartz_price(option_call, process, n_paths, n_steps)
                put_matrix[i, j] = longstaff_schwartz_price(option_put, process, n_paths, n_steps)
        return call_matrix, put_matrix


    def _compute_american_crr_heatmaps(
        s_values: np.ndarray,
        k_values: np.ndarray,
        maturity: float,
        rate: float,
        sigma: float,
        n_tree: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        call_matrix = np.zeros((len(k_values), len(s_values)))
        put_matrix = np.zeros_like(call_matrix)
        for i, strike in enumerate(k_values):
            for j, spot in enumerate(s_values):
                option_call = Option(s0=spot, T=maturity, K=strike, call=True)
                option_put = Option(s0=spot, T=maturity, K=strike, call=False)
                call_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_call, n=n_tree)
                put_matrix[i, j] = crr_pricing(r=rate, sigma=sigma, option=option_put, n=n_tree)
        return call_matrix, put_matrix


    def _build_crr_tree(option: Option, r: float, sigma: float, n_steps: int) -> tuple[np.ndarray, np.ndarray]:
        if n_steps <= 0:
            raise ValueError("n_steps doit être supérieur à 0.")
        dt = option.T / n_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u
        a = np.exp(r * dt)
        p = (a - d) / (u - d)
        q = 1 - p

        spot_tree = np.full((n_steps + 1, n_steps + 1), np.nan)
        value_tree = np.full_like(spot_tree, np.nan)

        for level in range(n_steps + 1):
            for up_moves in range(level + 1):
                spot_tree[level, up_moves] = option.s0 * (u**up_moves) * (d ** (level - up_moves))

        payoff_last = option.payoff(spot_tree[n_steps, : n_steps + 1])
        value_tree[n_steps, : n_steps + 1] = payoff_last
        discount = np.exp(-r * dt)

        for level in range(n_steps - 1, -1, -1):
            for up_moves in range(level + 1):
                continuation = discount * (
                    p * value_tree[level + 1, up_moves + 1] + q * value_tree[level + 1, up_moves]
                )
                exercise = option.payoff(np.array([spot_tree[level, up_moves]]))[0]
                value_tree[level, up_moves] = max(exercise, continuation)

        return spot_tree, value_tree


    def _format_tree_matrix(matrix: np.ndarray, precision: int = 4) -> np.ndarray:
        fmt = f"{{:.{precision}f}}"
        formatted = []
        for row in matrix:
            formatted.append([fmt.format(value) if not np.isnan(value) else "" for value in row])
        return np.array(formatted)


    def _plot_crr_tree(spots: np.ndarray, values: np.ndarray) -> plt.Figure:
        n_levels = spots.shape[0]
        fig_width = min(12, 4 + n_levels * 0.25)
        fig_height = min(10, 3 + n_levels * 0.25)
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        ax.set_axis_off()

        def _node_coords(level: int, index: int) -> tuple[float, float]:
            x = index - level / 2
            y = n_levels - 1 - level
            return x, y

        for level in range(n_levels - 1):
            for index in range(level + 1):
                if np.isnan(spots[level, index]):
                    continue
                x_curr, y_curr = _node_coords(level, index)
                x_down, y_down = _node_coords(level + 1, index)
                x_up, y_up = _node_coords(level + 1, index + 1)
                ax.plot([x_curr, x_down], [y_curr, y_down], color="lightgray", linewidth=0.8)
                ax.plot([x_curr, x_up], [y_curr, y_up], color="lightgray", linewidth=0.8)

        x_coords = []
        y_coords = []
        color_values = []
        spots_list = []
        option_list = []

        for level in range(n_levels):
            for index in range(level + 1):
                value = spots[level, index]
                option_value = values[level, index]
                if np.isnan(value) or np.isnan(option_value):
                    continue
                x, y = _node_coords(level, index)
                x_coords.append(x)
                y_coords.append(y)
                color_values.append(option_value)
                spots_list.append(value)
                option_list.append(option_value)

        scatter = ax.scatter(
            x_coords,
            y_coords,
            c=color_values,
            cmap="viridis",
            s=120,
            edgecolors="black",
            linewidths=0.5,
        )
        display_labels = n_levels - 1 <= 10
        if display_labels:
            for x, y, spot_value, option_value in zip(x_coords, y_coords, spots_list, option_list):
                ax.text(x, y + 0.25, f"S={spot_value:.2f}", ha="center", va="bottom", fontsize=7)
                ax.text(x, y - 0.25, f"V={option_value:.2f}", ha="center", va="top", fontsize=7)

        ax.set_ylim(-0.5, n_levels - 0.5)
        ax.set_xlim(min(x_coords, default=-1) - 1, max(x_coords, default=1) + 1)
        ax.set_title("Arbre CRR (couleur = valeur de l'option)")
        fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Valeur de l'option")
        fig.tight_layout()
        return fig


    # ---------------------------------------------------------------------------
    #  Modules Basket & Asian – helpers
    # ---------------------------------------------------------------------------


    @st.cache_data(show_spinner=False)
    def get_option_expiries(ticker: str):
        tk = yf.Ticker(ticker)
        return tk.options or []


    @st.cache_data(show_spinner=False)
    def get_option_surface_from_yf(ticker: str, expiry: str):
        tk = yf.Ticker(ticker)
        chain = tk.option_chain(expiry)

        frames = []
        for frame in [chain.calls, chain.puts]:
            tmp = frame[["strike", "impliedVolatility"]].rename(columns={"strike": "K", "impliedVolatility": "iv"})
            tmp["T"] = 0.0
            frames.append(tmp)
        df = pd.concat(frames, ignore_index=True)
        df = df.dropna(subset=["K", "iv"])
        return df


    @st.cache_data(show_spinner=False)
    def get_spot_and_hist_vol(ticker: str, period: str = "6mo", interval: str = "1d"):
        data = yf.download(ticker, period=period, interval=interval, progress=False)
        if data.empty:
            raise ValueError("Aucune donnée téléchargée.")
        close = data["Close"]
        spot = float(close.iloc[-1])
        log_returns = np.log(close / close.shift(1)).dropna()
        sigma = float(log_returns.std() * np.sqrt(252))
        hist_df = data.reset_index()
        hist_df["Date"] = pd.to_datetime(hist_df["Date"])
        return spot, sigma, hist_df


    def fetch_closing_prices(tickers, period="1mo", interval="1d"):
        if isinstance(tickers, str):
            tickers = [tickers]
        for var in ["YF_IMPERSONATE", "YF_SCRAPER_IMPERSONATE"]:
            try:
                os.environ.pop(var, None)
            except Exception:
                pass
        try:
            yf.set_config(proxy=None)
        except Exception:
            pass

        data = yf.download(
            tickers=tickers,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
        if data.empty:
            raise RuntimeError(f"Aucune donnée récupérée pour {tickers} sur {period}.")

        if isinstance(data.columns, pd.MultiIndex):
            prices = data["Adj Close"] if "Adj Close" in data.columns.levels[0] else data["Close"]
        else:
            if "Adj Close" in data.columns:
                prices = data[["Adj Close"]].copy()
            elif "Close" in data.columns:
                prices = data[["Close"]].copy()
            else:
                raise RuntimeError("Colonnes de prix introuvables dans les données yfinance.")
            prices.columns = tickers

        prices = prices.reset_index()
        return prices


    def compute_corr_from_prices(prices_df: pd.DataFrame):
        price_cols = [c for c in prices_df.columns if c.lower() != "date"]
        returns = np.log(prices_df[price_cols] / prices_df[price_cols].shift(1)).dropna(how="any")
        if returns.empty:
            raise RuntimeError("Pas assez de données pour calculer la corrélation.")
        return returns.corr()


    def load_closing_prices_with_tickers(path: Path) -> tuple[pd.DataFrame | None, list[str]]:
        if not path.exists():
            return None, []
        try:
            df = pd.read_csv(path)
        except Exception:
            return None, []
        ticker_cols: list[str] = []
        for col in df.columns:
            col_str = str(col).strip()
            if not col_str or col_str.lower() == "date":
                continue
            ticker_cols.append(col_str)
        return df, ticker_cols


    class BasketOption:
        def __init__(self, weights, prices, volatility, corr, strike, maturity, rate):
            self.weights = weights
            self.vol = volatility
            self.strike = strike
            self.mat = maturity
            self.rate = rate
            self.corr = corr
            self.prices = prices

        def get_mc(self, m_paths: int = 10000):
            b_ts = stats.multivariate_normal(np.zeros(len(self.weights)), cov=self.corr).rvs(size=m_paths)
            s_ts = self.prices * np.exp((self.rate - 0.5 * self.vol**2) * self.mat + self.vol * b_ts)
            if len(self.weights) > 1:
                payoffs = (np.sum(self.weights * s_ts, axis=1) - self.strike).clip(0)
            else:
                payoffs = np.maximum(s_ts - self.strike, np.zeros(m_paths))
            return float(np.exp(-self.rate * self.mat) * np.mean(payoffs))

        def get_bs_price(self):
            d1 = (np.log(self.prices / self.strike) + (self.rate + 0.5 * self.vol**2) * self.mat) / (
                self.vol * np.sqrt(self.mat)
            )
            d2 = d1 - self.vol * np.sqrt(self.mat)
            bs_price = stats.norm.cdf(d1) * self.prices - stats.norm.cdf(d2) * self.strike * np.exp(-self.rate * self.mat)
            return float(bs_price)


    class DataGen:
        def __init__(self, n_assets: int, n_samples: int):
            if n_samples <= 0:
                raise ValueError("n_samples needs to be positive")
            if n_assets <= 0:
                raise ValueError("n_assets needs to be positive")
            self.n_assets = n_assets
            self.n_samples = n_samples

        def generate(self, corr, strike_price: float, base_price: float, method: str = "bs"):
            mats = np.random.uniform(0.2, 1.1, size=self.n_samples)
            vols = np.random.uniform(0.01, 1.0, size=self.n_samples)
            rates = np.random.uniform(0.02, 0.1, size=self.n_samples)

            strikes = np.random.randn(self.n_samples) + strike_price
            prices = np.random.randn(self.n_samples) + base_price

            if self.n_assets > 1:
                weights = np.random.rand(self.n_samples * self.n_assets).reshape((self.n_samples, self.n_assets))
                weights /= np.sum(weights, axis=1)[:, np.newaxis]
            else:
                weights = np.ones((self.n_samples, self.n_assets))

            labels = []
            for i in range(self.n_samples):
                basket = BasketOption(
                    weights[i],
                    prices[i],
                    vols[i],
                    corr,
                    strikes[i],
                    mats[i],
                    rates[i],
                )
                if method == "bs":
                    labels.append(basket.get_bs_price())
                else:
                    labels.append(basket.get_mc())

            data = pd.DataFrame(
                {
                    "S/K": prices / strikes,
                    "Maturity": mats,
                    "Volatility": vols,
                    "Rate": rates,
                    "Labels": labels,
                    "Prices": prices,
                    "Strikes": strikes,
                }
            )
            for i in range(self.n_assets):
                data[f"Weight_{i}"] = weights[:, i]
            return data


    def simulate_dataset_notebook(n_assets: int, n_samples: int, method: str, corr: np.ndarray, base_price: float, base_strike: float):
        generator = DataGen(n_assets=n_assets, n_samples=n_samples)
        return generator.generate(corr=corr, strike_price=base_strike, base_price=base_price, method=method)


    @st.cache_data(show_spinner=False)
    def load_csv_bytes(file_bytes: bytes) -> pd.DataFrame:
        return pd.read_csv(io.BytesIO(file_bytes))


    def split_data_nn(data: pd.DataFrame, split_ratio: float = 0.7):
        feature_cols = ["S/K", "Maturity", "Volatility", "Rate"]
        target_col = "Labels"
        train = data.iloc[: int(split_ratio * len(data)), :]
        test = data.iloc[int(split_ratio * len(data)) :, :]
        x_train, y_train = train[feature_cols], train[target_col]
        x_test, y_test = test[feature_cols], test[target_col]
        return x_train, y_train, x_test, y_test


    def build_model_nn(input_dim: int) -> tf.keras.Model:
        inp = tf.keras.layers.Input(shape=(input_dim,))
        x = tf.keras.layers.Dense(32, activation="relu")(inp)
        x = tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(64, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        out = tf.keras.layers.Dense(1, activation="relu")(x)
        model = tf.keras.Model(inputs=inp, outputs=out)
        model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mean_squared_error"],
        )
        return model


    def price_basket_nn(model: tf.keras.Model, S: float, K: float, maturity: float, volatility: float, rate: float) -> float:
        S_over_K = S / K
        x = np.array([[S_over_K, maturity, volatility, rate]], dtype=float)
        return float(model.predict(x, verbose=0)[0, 0])


    def plot_heatmap_nn(
        model: tf.keras.Model,
        data: pd.DataFrame,
        spot_ref: float | None = None,
        strike_ref: float | None = None,
        maturity_fixed: float = 1.0,
    ):
        df = data.copy()
        if "Prices" not in df.columns and spot_ref is not None:
            df["Prices"] = spot_ref
        if "Strikes" not in df.columns and strike_ref is not None:
            df["Strikes"] = strike_ref

        if not {"Prices", "Strikes"}.issubset(df.columns):
            raise ValueError("Colonnes Prices et Strikes requises pour reproduire la heatmap du notebook.")

        s_min, s_max = df["Prices"].quantile([0.01, 0.99])
        k_min, k_max = df["Strikes"].quantile([0.01, 0.99])
        n_S, n_K = 50, 50
        s_vals = np.linspace(s_min, s_max, n_S)
        k_vals = np.linspace(k_min, k_max, n_K)

        K_grid, S_grid = np.meshgrid(k_vals, s_vals)
        s_over_k_grid = S_grid / K_grid

        sigma_ref = float(df["Volatility"].median())
        rate_ref = float(df["Rate"].median())

        X = np.stack(
            [
                s_over_k_grid.ravel(),
                np.full(s_over_k_grid.size, maturity_fixed),
                np.full(s_over_k_grid.size, sigma_ref),
                np.full(s_over_k_grid.size, rate_ref),
            ],
            axis=1,
        )
        prices_grid = model.predict(X, verbose=0).reshape(n_S, n_K)

        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(
            prices_grid,
            origin="lower",
            extent=[k_vals.min(), k_vals.max(), s_vals.min(), s_vals.max()],
            aspect="auto",
            cmap="viridis",
        )
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Spot S")
        ax.set_title("Heatmap du prix NN en fonction de S et K (T=1 an)")
        fig.colorbar(im, ax=ax, label="Prix NN")
        plt.tight_layout()
        return fig


    def build_grid(
        df: pd.DataFrame,
        spot: float,
        n_k: int = 200,
        n_t: int = 200,
        k_min: float | None = None,
        k_max: float | None = None,
        t_min: float | None = None,
        t_max: float | None = None,
        k_span: float | None = None,
        margin_frac: float = 0.02,
    ):
        if k_min is None or k_max is None:
            if k_span is not None:
                k_min = spot - k_span
                k_max = spot + k_span
            else:
                data_k_min = float(df["K"].min())
                data_k_max = float(df["K"].max())
                delta_k = data_k_max - data_k_min
                pad = delta_k * margin_frac
                k_min = data_k_min - pad
                k_max = data_k_max + pad

        if t_min is None:
            t_min = float(df["T"].min())
        if t_max is None:
            t_max = float(df["T"].max())

        if k_min >= k_max:
            raise ValueError("k_min doit être inférieur à k_max.")
        if t_min >= t_max:
            raise ValueError("t_min doit être inférieur à t_max.")

        k_vals = np.linspace(k_min, k_max, n_k)
        t_vals = np.linspace(t_min, t_max, n_t)

        df = df[(df["K"] >= k_min) & (df["K"] <= k_max)].copy()
        df = df[(df["T"] >= t_min) & (df["T"] <= t_max)]

        if df.empty:
            raise ValueError("Aucun point n'appartient au domaine défini par la grille.")

        df["K_idx"] = np.searchsorted(k_vals, df["K"], side="left").clip(0, n_k - 1)
        df["T_idx"] = np.searchsorted(t_vals, df["T"], side="left").clip(0, n_t - 1)

        grouped = df.groupby(["T_idx", "K_idx"])["iv"].mean().reset_index()

        iv_grid = np.full((n_t, n_k), np.nan, dtype=float)
        for _, row in grouped.iterrows():
            iv_grid[int(row["T_idx"]), int(row["K_idx"])] = row["iv"]

        k_grid, t_grid = np.meshgrid(k_vals, t_vals)
        return k_grid, t_grid, iv_grid


    def make_iv_surface_figure(k_grid, t_grid, iv_grid, title_suffix=""):
        fig = plt.figure(figsize=(12, 5))

        ax3d = fig.add_subplot(1, 2, 1, projection="3d")

        iv_flat = iv_grid[~np.isnan(iv_grid)]
        if iv_flat.size == 0:
            raise ValueError("La grille iv_grid ne contient aucune valeur non-NaN.")
        iv_mean = iv_flat.mean()
        iv_grid_filled = np.where(np.isnan(iv_grid), iv_mean, iv_grid)

        surf = ax3d.plot_surface(
            k_grid,
            t_grid,
            iv_grid_filled,
            rstride=1,
            cstride=1,
            linewidth=0.2,
            antialiased=True,
            cmap="viridis",
        )

        ax3d.set_xlabel("Strike K")
        ax3d.set_ylabel("Maturité T (années)")
        ax3d.set_zlabel("Implied vol")
        ax3d.set_title(f"Surface 3D de volatilité implicite{title_suffix}")

        fig.colorbar(surf, shrink=0.5, aspect=10, ax=ax3d, label="iv")

        ax2d = fig.add_subplot(1, 2, 2)
        im = ax2d.imshow(
            iv_grid_filled,
            extent=[k_grid.min(), k_grid.max(), t_grid.min(), t_grid.max()],
            origin="lower",
            aspect="auto",
            cmap="viridis",
        )
        ax2d.set_xlabel("Strike K")
        ax2d.set_ylabel("Maturité T (années)")
        ax2d.set_title(f"Heatmap IV{title_suffix}")
        fig.colorbar(im, ax=ax2d, label="iv")

        plt.tight_layout()
        return fig


    def btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps):
        delta_t = maturity / steps
        up = np.exp(sigma * np.sqrt(delta_t))
        down = 1.0 / up
        prob = (np.exp(rate * delta_t) - down) / (up - down)

        spot_paths = [spot]
        avg_paths = [spot]
        strike_paths = [strike]

        for _ in range(steps):
            spot_paths = [s * up for s in spot_paths] + [s * down for s in spot_paths]
            avg_paths = avg_paths + avg_paths
            strike_paths = strike_paths + strike_paths
            for index in range(len(avg_paths)):
                avg_paths[index] = avg_paths[index] + spot_paths[index]

        avg_paths = np.array(avg_paths) / (steps + 1)
        spot_paths = np.array(spot_paths)
        strike_paths = np.array(strike_paths)

        if strike_type == "fixed":
            if option_type == "C":
                payoff = np.maximum(avg_paths - strike_paths, 0.0)
            else:
                payoff = np.maximum(strike_paths - avg_paths, 0.0)
        else:
            if option_type == "C":
                payoff = np.maximum(spot_paths - avg_paths, 0.0)
            else:
                payoff = np.maximum(avg_paths - spot_paths, 0.0)

        option_price = payoff.copy()
        for _ in range(steps):
            length = len(option_price) // 2
            option_price = prob * option_price[:length] + (1 - prob) * option_price[length:]

        return float(option_price[0])


    def hw_btm_asian(strike_type, option_type, spot, strike, rate, sigma, maturity, steps, m_points):
        n_steps = steps
        delta_t = maturity / n_steps
        up = np.exp(sigma * np.sqrt(delta_t))
        down = 1.0 / up
        prob = (np.exp(rate * delta_t) - down) / (up - down)

        avg_grid = []
        strike_vec = np.array([strike] * m_points)

        for j_index in range(n_steps + 1):
            path_up_then_down = np.array(
                [spot * up**j * down**0 for j in range(n_steps - j_index)]
                + [spot * up**(n_steps - j_index) * down**j for j in range(j_index + 1)]
            )
            avg_max = path_up_then_down.mean()

            path_down_then_up = np.array(
                [spot * down**j * up**0 for j in range(j_index + 1)]
                + [spot * down**j_index * up**(j + 1) for j in range(n_steps - j_index)]
            )
            avg_min = path_down_then_up.mean()

            diff = avg_max - avg_min
            avg_vals = [avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)]
            avg_grid.append(avg_vals)

        avg_grid = np.round(avg_grid, 4)

        payoff = []
        for j_index in range(n_steps + 1):
            avg_vals = np.array(avg_grid[j_index])
            spot_vals = np.array([spot * up**(n_steps - j_index) * down**j_index] * m_points)

            if strike_type == "fixed":
                if option_type == "C":
                    pay = np.maximum(avg_vals - strike_vec, 0.0)
                else:
                    pay = np.maximum(strike_vec - avg_vals, 0.0)
            else:
                if option_type == "C":
                    pay = np.maximum(spot_vals - avg_vals, 0.0)
                else:
                    pay = np.maximum(avg_vals - spot_vals, 0.0)

            payoff.append(pay)

        payoff = np.round(np.array(payoff), 4)

        for n_index in range(n_steps - 1, -1, -1):
            avg_backward = []
            payoff_backward = []

            for j_index in range(n_index + 1):
                path_up_then_down = np.array(
                    [spot * up**j * down**0 for j in range(n_index - j_index)]
                    + [spot * up**(n_index - j_index) * down**j for j in range(j_index + 1)]
                )
                avg_max = path_up_then_down.mean()

                path_down_then_up = np.array(
                    [spot * down**j * up**0 for j in range(j_index + 1)]
                    + [spot * down**j_index * up**(j + 1) for j in range(n_index - j_index)]
                )
                avg_min = path_down_then_up.mean()

                diff = avg_max - avg_min
                avg_vals = np.array([avg_max - diff * k_index / (m_points - 1) for k_index in range(m_points)])
                avg_backward.append(avg_vals)

            avg_backward = np.round(np.array(avg_backward), 4)

            payoff_new = []
            for j_index in range(n_index + 1):
                avg_vals = avg_backward[j_index]
                pay_vals = np.zeros_like(avg_vals)

                avg_up = np.array(avg_grid[j_index])
                avg_down = np.array(avg_grid[j_index + 1])
                pay_up = payoff[j_index]
                pay_down = payoff[j_index + 1]

                for k_index, avg_k in enumerate(avg_vals):
                    if avg_k <= avg_up[0]:
                        fu = pay_up[0]
                    elif avg_k >= avg_up[-1]:
                        fu = pay_up[-1]
                    else:
                        idx = np.searchsorted(avg_up, avg_k) - 1
                        x0, x1 = avg_up[idx], avg_up[idx + 1]
                        y0, y1 = pay_up[idx], pay_up[idx + 1]
                        fu = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                    if avg_k <= avg_down[0]:
                        fd = pay_down[0]
                    elif avg_k >= avg_down[-1]:
                        fd = pay_down[-1]
                    else:
                        idx = np.searchsorted(avg_down, avg_k) - 1
                        x0, x1 = avg_down[idx], avg_down[idx + 1]
                        y0, y1 = pay_down[idx], pay_down[idx + 1]
                        fd = y0 + (y1 - y0) * (avg_k - x0) / (x1 - x0)

                    pay_vals[k_index] = (prob * fu + (1 - prob) * fd) * np.exp(-rate * delta_t)

                payoff_backward.append(pay_vals)

            avg_grid = avg_backward
            payoff = np.round(np.array(payoff_backward), 4)

        option_price = payoff[0].mean()
        return float(option_price)


    def bs_option_price(time, spot, strike, maturity, rate, sigma, option_kind):
        tau = maturity - time
        if tau <= 0:
            if option_kind == "call":
                return max(spot - strike, 0.0)
            return max(strike - spot, 0.0)

        d1 = (np.log(spot / strike) + (rate + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
        d2 = d1 - sigma * np.sqrt(tau)

        if option_kind == "call":
            price = spot * norm.cdf(d1) - strike * np.exp(-rate * tau) * norm.cdf(d2)
        else:
            price = strike * np.exp(-rate * tau) * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return float(price)


    def asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type):
        if n_obs < 1:
            return 0.0
        dt = maturity / n_obs
        nu = rate - 0.5 * sigma**2
        sigma_g_sq = (sigma**2) * (n_obs + 1) * (2 * n_obs + 1) / (6 * n_obs**2)
        sigma_g = np.sqrt(sigma_g_sq)
        mu_g = (nu * (n_obs + 1) / (2 * n_obs) + 0.5 * sigma_g_sq) * maturity
        d1 = (np.log(spot / strike) + mu_g + 0.5 * sigma_g_sq * maturity) / (sigma_g * np.sqrt(maturity))
        d2 = d1 - sigma_g * np.sqrt(maturity)
        df = np.exp(-rate * maturity)
        if option_type == "call":
            return float(df * (spot * np.exp(mu_g) * norm.cdf(d1) - strike * norm.cdf(d2)))
        else:
            return float(df * (strike * norm.cdf(-d2) - spot * np.exp(mu_g) * norm.cdf(-d1)))


    def asian_mc_control_variate(
        spot,
        strike,
        rate,
        sigma,
        maturity,
        n_obs,
        n_paths,
        option_type,
        antithetic=True,
        seed=None,
    ):
        if seed is not None:
            np.random.seed(seed)
        dt = maturity / n_obs
        drift = (rate - 0.5 * sigma**2) * dt
        vol_step = sigma * np.sqrt(dt)

        if antithetic:
            n_base = max(1, n_paths // 2)
            z_base = np.random.randn(n_obs, n_base)
            z = np.concatenate([z_base, -z_base], axis=1)
            n_eff = z.shape[1]
        else:
            z = np.random.randn(n_obs, n_paths)
            n_eff = n_paths

        log_s = np.log(spot) + np.cumsum(drift + vol_step * z, axis=0)
        s_paths = np.exp(log_s)

        arith_mean = s_paths.mean(axis=0)
        geom_mean = np.exp(np.log(s_paths).mean(axis=0))
        if option_type == "call":
            arith_payoff = np.maximum(arith_mean - strike, 0.0)
            geom_payoff = np.maximum(geom_mean - strike, 0.0)
        else:
            arith_payoff = np.maximum(strike - arith_mean, 0.0)
            geom_payoff = np.maximum(strike - geom_mean, 0.0)
        closed_geom = asian_geometric_closed_form(spot, strike, rate, sigma, maturity, n_obs, option_type)
        cov = np.cov(arith_payoff, geom_payoff)[0, 1]
        var_geom = np.var(geom_payoff)
        c = cov / var_geom if var_geom > 0 else 0.0
        control_estimator = arith_payoff - c * (geom_payoff - closed_geom)
        disc = np.exp(-rate * maturity)
        disc_payoff = disc * control_estimator
        price = np.mean(disc_payoff)
        stderr = np.std(disc_payoff, ddof=1) / np.sqrt(n_eff)
        return float(price), float(stderr), float(c)


    def compute_asian_price(
        strike_type: str,
        option_type: str,
        model: str,
        spot: float,
        strike: float,
        rate: float,
        sigma: float,
        maturity: float,
        steps: int,
        m_points: int | None,
    ):
        if model == "BTM naïf":
            return btm_asian(
                strike_type=strike_type,
                option_type=option_type,
                spot=spot,
                strike=strike,
                rate=rate,
                sigma=sigma,
                maturity=maturity,
                steps=int(steps),
            )
        m_points_val = int(m_points) if m_points is not None else 10
        return hw_btm_asian(
            strike_type=strike_type,
            option_type=option_type,
            spot=spot,
            strike=strike,
            rate=rate,
            sigma=sigma,
            maturity=maturity,
            steps=int(steps),
            m_points=m_points_val,
        )


    def ui_basket_surface(spot_common, maturity_common, rate_common, strike_common, key_prefix: str = "basket"):
        st.header("Basket – Pricing NN + corrélation (3 actifs)")
        render_unlock_sidebar_button("tab_basket", "🔓 Réactiver T (onglet Basket)")

        min_assets, max_assets = 2, 10
        closing_path = DATASETS_DIR / "closing_prices.csv"
        prices_df_cached, csv_tickers = load_closing_prices_with_tickers(closing_path)

        def _normalize_tickers(candidates: list[str]) -> list[str]:
            cleaned = [str(tk).strip().upper() for tk in candidates if str(tk).strip()]
            trimmed = cleaned[:max_assets]
            if len(trimmed) < min_assets:
                trimmed += ["SPY"] * (min_assets - len(trimmed))
            return trimmed

        def _k(suffix: str) -> str:
            return f"{key_prefix}_{suffix}"

        if "basket_tickers" not in st.session_state:
            default_list = csv_tickers if csv_tickers else ["AAPL", "SPY", "MSFT"]
            st.session_state["basket_tickers"] = _normalize_tickers(default_list)

        with st.container():
            st.subheader("Sélection des assets (2 à 10)")
            btn_col_add, btn_col_remove = st.columns(2)
            with btn_col_add:
                if st.button(
                    "Ajouter un asset",
                    key=_k("btn_add_asset"),
                    disabled=len(st.session_state["basket_tickers"]) >= max_assets,
                ):
                    st.session_state["basket_tickers"].append(
                        f"TICKER{len(st.session_state['basket_tickers']) + 1}"
                    )
            with btn_col_remove:
                if st.button(
                    "Retirer un asset",
                    key=_k("btn_remove_asset"),
                    disabled=len(st.session_state["basket_tickers"]) <= min_assets,
                ):
                    st.session_state["basket_tickers"].pop()

            tickers = []
            for i, default_tk in enumerate(st.session_state["basket_tickers"]):
                if i % 3 == 0:
                    cols = st.columns(3)
                col = cols[i % 3]
                with col:
                    tick = st.text_input(f"Ticker {i + 1}", value=default_tk, key=_k(f"corr_tk_dynamic_{i}"))
                    tickers.append(tick.strip().upper() or default_tk)
            tickers = tickers[:max_assets]
            if len(tickers) < min_assets:
                tickers += ["SPY"] * (min_assets - len(tickers))
            st.session_state["basket_tickers"] = tickers
        tickers = st.session_state["basket_tickers"]

        period = st.selectbox("Période yfinance", ["1mo", "3mo", "6mo", "1y"], index=0, key=_k("corr_period"))
        interval = st.selectbox("Intervalle", ["1d", "1h"], index=0, key=_k("corr_interval"))

        st.caption(
            "Le calcul de corrélation utilise les prix de clôture présents dans database/data/closing_prices.csv (régénéré via yfinance). "
            "En cas d'échec, une matrice de corrélation inventée sera utilisée."
        )
        regen_csv = st.button("Mettre à jour la Matrice de Corrélation", key=_k("btn_regen_closing"))
        try:
            if regen_csv or not closing_path.exists():
                prices_df_cached = fetch_closing_prices(tickers, period=period, interval=interval)
                closing_path.parent.mkdir(parents=True, exist_ok=True)
                prices_df_cached.to_csv(closing_path, index=False)
                csv_tickers = [c for c in prices_df_cached.columns if str(c).lower() != "date"]
                st.info(f"database/data/closing_prices.csv généré via yfinance ({len(prices_df_cached)} lignes)")
                if csv_tickers:
                    st.session_state["basket_tickers"] = _normalize_tickers(csv_tickers)
                    tickers = st.session_state["basket_tickers"]
        except Exception as exc:
            st.warning(f"Impossible de récupérer les prix de clôture : {exc}")

        corr_df = None
        try:
            if prices_df_cached is None:
                prices_df_cached, _ = load_closing_prices_with_tickers(closing_path)
            if prices_df_cached is None:
                raise FileNotFoundError("Impossible de charger database/data/closing_prices.csv.")
            corr_df = compute_corr_from_prices(prices_df_cached)
            st.success(f"Corrélation calculée à partir de {closing_path.name}")
            st.dataframe(corr_df)
        except Exception as exc:
            st.warning(f"Impossible de calculer la corrélation depuis database/data/closing_prices.csv : {exc}")
            corr_df = pd.DataFrame(
                [
                    [1.0, 0.6, 0.4],
                    [0.6, 1.0, 0.7],
                    [0.4, 0.7, 1.0],
                ],
                columns=tickers,
                index=tickers,
            )
            st.info("Utilisation d'une matrice de corrélation inventée pour la suite des calculs.")
            st.dataframe(corr_df)

        st.subheader("Dataset Basket pour NN")
        st.caption("Dataset généré automatiquement via DataGen (comme dans le notebook).")
        n_samples = st.slider("Taille du dataset simulé", 1000, 20000, 10000, 1000, key=_k("basket_n_samples"))
        method = st.selectbox("Méthode de pricing pour les labels", ["bs", "mc"], index=0, key=_k("basket_method"))

        df = simulate_dataset_notebook(
            n_assets=len(tickers),
            n_samples=int(n_samples),
            method=method,
            corr=corr_df.values,
            base_price=float(spot_common),
            base_strike=float(strike_common),
        )

        st.write("Aperçu :", df.head())
        st.write("Shape :", df.shape)

        split_ratio = st.slider("Train ratio", 0.5, 0.9, 0.7, 0.05, key=_k("basket_split_ratio"))
        epochs = st.slider("Epochs d'entraînement", 5, 200, 20, 5, key=_k("basket_epochs"))

        x_train, y_train, x_test, y_test = split_data_nn(df, split_ratio=split_ratio)
        Path("data").mkdir(parents=True, exist_ok=True)
        pd.concat([x_train, y_train], axis=1).to_csv(DATASETS_DIR / "train.csv", index=False)
        pd.concat([x_test, y_test], axis=1).to_csv(DATASETS_DIR / "test.csv", index=False)
        st.info("train.csv et test.csv régénérés pour la surface IV.")

        st.write(f"Train size: {x_train.shape[0]} | Test size: {x_test.shape[0]}")

        train_button = st.button("Entraîner le modèle NN", key=_k("btn_train_nn"))
        if not train_button:
            st.info("Clique sur 'Entraîner le modèle NN' pour lancer l'apprentissage.")
            return

        tf.keras.backend.clear_session()
        model = build_model_nn(input_dim=x_train.shape[1])
        train_logs: list[str] = []
        log_box = st.empty()

        class StreamlitLogger(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                logs = logs or {}
                msg = (
                    f"Epoch {epoch + 1}/{epochs} - loss: {logs.get('loss', float('nan')):.4f} - "
                    f"mse: {logs.get('mean_squared_error', float('nan')):.4f}"
                )
                if "val_loss" in logs or "val_mean_squared_error" in logs:
                    msg += (
                        f" - val_loss: {logs.get('val_loss', float('nan')):.4f} - "
                        f"val_mse: {logs.get('val_mean_squared_error', float('nan')):.4f}"
                    )
                train_logs.append(msg)
                log_box.text("\n".join(train_logs))

        with st.spinner("Entraînement du NN en cours…"):
            history = model.fit(
                x_train,
                y_train,
                epochs=epochs,
                validation_data=(x_test, y_test),
                verbose=0,
                callbacks=[StreamlitLogger()],
            )
        st.success("Entraînement terminé.")

        st.subheader("Courbe MSE NN")
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(history.history["mean_squared_error"], label="train")
        ax.plot(history.history["val_mean_squared_error"], label="val")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.4)
        st.pyplot(fig)

        st.subheader("Heatmap prix NN (S vs K)")
        try:
            with st.spinner("Calcul de la heatmap…"):
                heatmap_fig = plot_heatmap_nn(
                    model=model,
                    data=df,
                    spot_ref=float(spot_common),
                    strike_ref=float(strike_common),
                    maturity_fixed=1.0,
                )
            st.pyplot(heatmap_fig)
        except Exception as exc:
            st.warning(f"Impossible d'afficher la heatmap : {exc}")

        st.subheader("Surface IV (Strike, Maturité)")
        try:
            with st.spinner("Calcul de la surface IV…"):
                iv_df = df.copy()
                if "Strikes" in iv_df.columns:
                    iv_df["K"] = iv_df["Strikes"]
                else:
                    iv_df["K"] = spot_common / iv_df["S/K"].replace(0.0, np.nan)
                iv_df = iv_df.replace([np.inf, -np.inf], np.nan).dropna(subset=["K", "Maturity", "Volatility"])

                if iv_df.empty:
                    raise ValueError("Pas de données IV exploitables (S/K nuls ou manquants).")

                spot_ref_for_grid = float(iv_df["Prices"].mean()) if "Prices" in iv_df.columns else float(spot_common)

                grid_k, grid_t, grid_iv = build_grid(
                    df=iv_df.rename(columns={"Maturity": "T", "Volatility": "iv"}),
                    spot=spot_ref_for_grid,
                )
                iv_fig = make_iv_surface_figure(grid_k, grid_t, grid_iv, title_suffix=" (dataset NN)")
            st.pyplot(iv_fig)
        except Exception as exc:
            st.warning(f"Impossible d'afficher la surface IV : {exc}")


    def ui_asian_options(
        spot_default,
        sigma_common,
        maturity_common,
        strike_common,
        rate_common,
        key_prefix: str = "asian",
        option_char: str = "c",
    ):
        # Prefix keys to avoid clashes when the module is rendered in multiple tabs.
        def _k(suffix: str) -> str:
            return f"{key_prefix}_{suffix}"

        st.header("Options asiatiques (module Asian)")
        render_unlock_sidebar_button("tab_asian", "🔓 Réactiver T (onglet Asian)")
        render_general_definition_explainer(
            "🌏 Comprendre les options asiatiques",
            (
                "- **Spécificité du payoff** : pour une option asiatique arithmétique, le payoff dépend de la moyenne des prix du sous‑jacent observés à différentes dates entre `0` et `T`, plutôt que du seul `S_T`.\n"
                "- **Effet de lissage** : cette moyenne réduit l’impact des pics de volatilité ponctuels et donne un profil de risque plus \"lissé\" que pour une option européenne standard.\n"
                "- **Conséquences sur le prix** : à paramètres identiques, une option asiatique est généralement moins chère que son équivalent européen car elle réagit moins aux extrêmes de la trajectoire.\n"
                "- **Usage pratique** : ces produits sont fréquemment utilisés dans l’énergie, les matières premières ou les produits structurés pour lisser l’exposition à des prix très volatils.\n"
                "- **Objectif du module** : illustrer le pricing d’options asiatiques par simulation Monte Carlo, avec des variates antithétiques et un contrôle par une option de référence."
            ),
        )
        render_method_explainer(
            "🧮 Méthode Monte Carlo + control variate",
            (
                "- **Étape 1 – Paramétrage de la grille** : pour chaque couple `(K, T)` de la grille choisie, on fixe un nombre d’observations `n_obs` le long de `[0, T]` et un nombre de trajectoires Monte Carlo `n_paths_surface`.\n"
                "- **Étape 2 – Simulation des trajectoires de `S_t`** : pour un spot initial donné, on simule sous la mesure neutre au risque `n_paths_surface` trajectoires du sous‑jacent en découpant `[0, T]` en `n_obs` pas. À chaque pas, on applique le schéma d’Euler du GBM.\n"
                "- **Étape 3 – Utilisation des variates antithétiques** : pour chaque suite de chocs gaussiens utilisée pour générer une trajectoire, on génère une trajectoire \"miroir\" avec les chocs opposés. On obtient ainsi des paires de trajectoires fortement corrélées qui réduisent la variance de l’estimateur.\n"
                "- **Étape 4 – Calcul de la moyenne arithmétique** : sur chaque trajectoire, on calcule la moyenne arithmétique des `S_t` observés aux dates de la grille. Cette moyenne est ensuite utilisée pour déterminer le payoff asiatique (call ou put) à l’échéance.\n"
                "- **Étape 5 – Construction d’une variable de contrôle** : en parallèle, on calcule pour chaque trajectoire le payoff d’une option de référence (par exemple une option européenne ou une option asiatique géométrique) dont on connaît une formule de prix fermée.\n"
                "- **Étape 6 – Correction par control variate** : on corrige l’estimation brute du payoff asiatique en soustrayant la composante due à la variable de contrôle, puis en réajoutant l’espérance théorique de cette variable. Cela réduit significativement la variance de l’estimateur final.\n"
                "- **Étape 7 – Actualisation et moyenne** : les payoffs corrigés sont actualisés au taux `rate_common` jusqu’à la date présente et moyennés sur toutes les trajectoires.\n"
                "- **Étape 8 – Remplissage des surfaces** : on répète ce processus pour chaque point `(K, T)` de la grille, ce qui remplit deux matrices de prix (call et put) utilisées pour tracer les surfaces de prix asiatiques."
            ),
        )
        render_inputs_explainer(
            "🔧 Paramètres utilisés – module Asian",
            (
                "- **\"S0 (spot)\"** (via les paramètres communs) : niveau de départ des trajectoires asiatiques.\n"
                "- **\"K (strike)\"** : strike de référence utilisé pour centrer la plage de strikes.\n"
                "- **\"T (maturité, années)\"** : maturité de référence utilisée pour initialiser la plage de maturités.\n"
                "- **\"Taux sans risque r\"** : intervient dans l’actualisation et le drift neutre au risque.\n"
                "- **\"Volatilité σ\"** : volatilité utilisée pour simuler les trajectoires du sous‑jacent.\n"
                "- **\"K min\" / \"K max\"** : bornes de la plage de strikes sur l’axe horizontal des surfaces.\n"
                "- **\"T min (années)\" / \"T max (années)\"** : bornes de la plage de maturités sur l’axe vertical.\n"
                "- **\"Résolution en K\"** et **\"Résolution en T\"** : nombres de points de grille en strike et en maturité.\n"
                "- **\"Nombre de trajectoires Monte Carlo\"** : nombre de trajectoires utilisées pour estimer chaque point de la surface."
            ),
        )
        if spot_default is None:
            st.warning("Aucun téléchargement yfinance : utilisez le spot commun.")
            spot_default = 57830.0
        if sigma_common is None:
            sigma_common = 0.05

        col1, col2 = st.columns(2)
        with col1:
            spot_common = st.session_state.get("common_spot", spot_default)
            strike_common_local = st.session_state.get("common_strike", strike_common)
            st.info(f"Spot commun S0 = {spot_common:.4f}")
            st.info(f"Strike commun K = {strike_common_local:.4f}")
            st.info(f"Taux sans risque commun r = {rate_common:.4f}")
        with col2:
            sigma = sigma_common
            st.info(f"Volatilité commune σ = {sigma:.4f}")
            st.info("Pricing asiatique via Monte Carlo + control variate (méthode notebook).")

        with st.expander("📈 Prix asiatique arithmétique (MC + control variate)", expanded=False):
            progress = st.progress(0)
            try:
                n_obs_price = max(2, int(50 * float(maturity_common)))
                price_asian_call, _, _ = asian_mc_control_variate(
                    spot=float(spot_common),
                    strike=float(strike_common_local),
                    rate=float(rate_common),
                    sigma=float(sigma),
                    maturity=float(maturity_common),
                    n_obs=int(n_obs_price),
                    n_paths=20_000,
                    option_type="call",
                    antithetic=True,
                    seed=None,
                )
                progress.progress(100)
                st.success(f"Prix call asiatique arithmétique (MC + control variate) = {price_asian_call:.6f}")
                render_add_to_dashboard_button(
                    product_label="Asian arithmétique",
                    option_char=option_char,
                    price_value=price_asian_call,
                    strike=strike_common_local,
                    maturity=maturity_common,
                    key_prefix=_k("save_asian_arith"),
                    spot=spot_common,
                    misc={
                        "method": "MC control variate",
                        "n_obs": int(n_obs_price),
                        "n_paths": 20000,
                        "sigma": float(sigma),
                        "r": float(rate_common),
                        "q": float(d_common),
                    },
                )
            except Exception as exc:
                st.error(f"Erreur lors du pricing asiatique : {exc}")
            finally:
                progress.empty()
        st.caption(
            f"Paramètres utilisés pour le prix asiatique : "
            f"S0={spot_common:.4f}, K={strike_common_local:.4f}, "
            f"T={maturity_common:.4f}, r={rate_common:.4f}, σ={sigma:.4f}"
        )

        st.subheader("Heatmaps prix asiatiques (K vs T)")
        k_center = st.session_state.get("common_strike", strike_common)
        k_span = float(st.session_state.get("heatmap_span_value", 25.0))
        k_min = max(0.01, k_center - k_span)
        k_max = k_center + k_span
        col_k, col_t = st.columns(2)
        with col_k:
            st.caption(f"Domaine K commun (span): [{k_min:.2f}, {k_max:.2f}]")
        with col_t:
            t_center = st.session_state.get("common_maturity", maturity_common)
            t_span = float(st.session_state.get("heatmap_maturity_span_value", max(0.05, t_center * 0.5)))
            t_min = max(0.01, t_center - t_span)
            t_max = t_center + t_span
            st.caption(f"Domaine T commun (span): [{t_min:.2f}, {t_max:.2f}]")

        n_k = st.slider("Résolution en K", 10, 40, 20, 2, key=_k("n_k"))
        n_t = st.slider("Résolution en T", 10, 40, 20, 2, key=_k("n_t"))
        n_paths_surface = st.slider("Nombre de trajectoires Monte Carlo", 5_000, 50_000, 20_000, 5_000, key=_k("n_paths"))

        is_call_tab = option_char.lower() == "c"
        heatmap_label = "Call" if is_call_tab else "Put"
        with st.expander(f"Afficher la heatmap {heatmap_label}", expanded=False):
            k_vals = np.linspace(k_min, k_max, n_k)
            t_vals = np.linspace(t_min, t_max, n_t)
            prices = np.zeros((n_t, n_k), dtype=float)

            with st.spinner("Calcul de la surface de prix (MC asiatique)…"):
                progress_surface = st.progress(0)
                total_iters = len(t_vals) * len(k_vals)
                done = 0
                for i_t, t_val in enumerate(t_vals):
                    n_obs = max(2, int(50 * t_val))
                    for i_k, k_val in enumerate(k_vals):
                        price_val, _, _ = asian_mc_control_variate(
                            spot=float(spot_common),
                            strike=float(k_val),
                            rate=float(rate_common),
                            sigma=float(sigma),
                            maturity=float(t_val),
                            n_obs=int(n_obs),
                            n_paths=int(n_paths_surface),
                            option_type="call" if is_call_tab else "put",
                            antithetic=True,
                            seed=None,
                        )
                        prices[i_t, i_k] = price_val
                        done += 1
                        if total_iters > 0:
                            progress_surface.progress(int((done / total_iters) * 100))
                progress_surface.empty()

            fig, ax = plt.subplots(figsize=(7, 4))
            im = ax.imshow(
                prices,
                origin="lower",
                extent=[k_vals.min(), k_vals.max(), t_vals.min(), t_vals.max()],
                aspect="auto",
                cmap="viridis",
            )
            ax.set_xlabel("Strike K")
            ax.set_ylabel("Maturité T (années)")
            ax.set_title(f"{heatmap_label} asiatique arithmétique (MC + control variate)")
            fig.colorbar(im, ax=ax, label="Prix")
            fig.tight_layout()
            st.pyplot(fig)


    # ---------------------------------------------------------------------------
    #  Module Heston – pipeline complet
    # ---------------------------------------------------------------------------


    def heston_mc_pricer(
        S0: float,
        K: float,
        T: float,
        r: float,
        v0: float,
        theta: float,
        kappa: float,
        sigma_v: float,
        rho: float,
        n_paths: int = 50_000,
        n_steps: int = 100,
        option_type: str = "call",
    ) -> float:
        dt = T / n_steps
        sqrt_dt = math.sqrt(dt)
        S = np.full(n_paths, S0)
        v = np.full(n_paths, v0)
        for _ in range(n_steps):
            z1 = np.random.randn(n_paths)
            z2 = np.random.randn(n_paths)
            z_s = z1
            z_v = rho * z1 + math.sqrt(1 - rho**2) * z2
            v_pos = np.maximum(v, 0)
            S = S * np.exp((r - 0.5 * v_pos) * dt + np.sqrt(v_pos) * sqrt_dt * z_s)
            v = v + kappa * (theta - v_pos) * dt + sigma_v * np.sqrt(v_pos) * sqrt_dt * z_v
            v = np.maximum(v, 0)
        payoff = np.maximum(S - K, 0) if option_type == "call" else np.maximum(K - S, 0)
        return float(math.exp(-r * T) * np.mean(payoff))


    def download_options_cboe(symbol: str, option_type: str) -> tuple[pd.DataFrame, float, float, float]:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        payload = resp.json()
        data = payload.get("data", {})
        options = data.get("options", [])
        spot = float(data.get("current_price") or data.get("close") or np.nan)
        risk_free = float(data.get("risk_free_rate") or 0.02)
        dividend_yield = float(data.get("dividend_yield") or 0.0)
        now = pd.Timestamp.utcnow().tz_localize(None)
        pattern = re.compile(rf"^{symbol.upper()}(?P<expiry>\d{{6}})(?P<cp>[CP])(?P<strike>\d+)$")

        rows: list[dict] = []
        for opt in options:
            match = pattern.match(opt.get("option", ""))
            if not match:
                continue
            cp = match.group("cp")
            if (option_type == "call" and cp != "C") or (option_type == "put" and cp != "P"):
                continue
            expiry_dt = pd.to_datetime(match.group("expiry"), format="%y%m%d")
            T = (expiry_dt - now).total_seconds() / (365.0 * 24 * 3600)
            if T <= 0:
                continue
            T = round(T, 2)
            if T <= MIN_IV_MATURITY:
                continue
            strike = int(match.group("strike")) / 1000.0
            bid = float(opt.get("bid") or 0.0)
            ask = float(opt.get("ask") or 0.0)
            last = float(opt.get("last_trade_price") or 0.0)
            if bid > 0 and ask > 0:
                mid = 0.5 * (bid + ask)
            elif last > 0:
                mid = last
            else:
                mid = np.nan
            if np.isnan(mid) or mid <= 0:
                continue
            iv_val = opt.get("iv", np.nan)
            iv_val = float(iv_val) if iv_val not in (None, "") else np.nan
            rows.append(
                {
                    "S0": spot,
                    "K": strike,
                    "T": T,
                    ("C_mkt" if option_type == "call" else "P_mkt"): round(mid, 2),
                    "iv_market": iv_val,
                }
            )

        df = pd.DataFrame(rows)
        df = df[df["T"] > MIN_IV_MATURITY]
        return df, spot, risk_free, dividend_yield


    @st.cache_data(show_spinner=False)
    def load_cboe_data(symbol: str) -> tuple[pd.DataFrame, pd.DataFrame, float, float, float]:
        calls_df, spot_calls, rf_calls, div_calls = download_options_cboe(symbol, "call")
        puts_df, spot_puts, rf_puts, div_puts = download_options_cboe(symbol, "put")
        S0_ref = float(np.nanmean([spot_calls, spot_puts]))
        risk_free = float(np.nanmean([rf_calls, rf_puts]))
        dividend_yield = float(np.nanmean([div_calls, div_puts]))
        return calls_df, puts_df, S0_ref, risk_free, dividend_yield


    def prices_from_unconstrained(u: torch.Tensor, S0_t: torch.Tensor, K_t: torch.Tensor, T_t: torch.Tensor, r: float, q: float):
        params = HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])
        prices = []
        for S0_i, K_i, T_i in zip(S0_t, K_t, T_t):
            price_i = carr_madan_call_torch(S0_i, r, q, T_i, params, K_i)
            prices.append(price_i)
        return torch.stack(prices)


    def heston_nn_loss(
        u: torch.Tensor,
        S0_t: torch.Tensor,
        K_t: torch.Tensor,
        T_t: torch.Tensor,
        C_mkt_t: torch.Tensor,
        r: float,
        q: float,
        weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        model_prices = prices_from_unconstrained(u, S0_t, K_t, T_t, r, q)
        diff = model_prices - C_mkt_t
        if weights is not None:
            return 0.5 * (weights * diff**2).mean()
        return 0.5 * (diff**2).mean()


    def calibrate_heston_nn(
        df: pd.DataFrame,
        r: float,
        q: float,
        max_iters: int,
        lr: float,
        spot_override: float | None = None,
        progress_callback: Callable[[int, int, float], None] | None = None,
    ) -> HestonParams:
        if df.empty:
            raise ValueError("DataFrame vide.")
        df_clean = df.dropna(subset=["S0", "K", "T", "C_mkt"])
        df_clean = df_clean[(df_clean["T"] > MIN_IV_MATURITY) & (df_clean["C_mkt"] > 0.05)]
        df_clean = df_clean[df_clean.get("iv_market", 0) > 0]
        if df_clean.empty:
            raise ValueError("Pas de points pour la calibration")

        S0_ref = spot_override if spot_override is not None else float(df_clean["S0"].median())
        moneyness = df_clean["K"].values / S0_ref

        S0_t = torch.tensor(df_clean["S0"].values, dtype=torch.float64, device=HES_DEVICE)
        K_t = torch.tensor(df_clean["K"].values, dtype=torch.float64, device=HES_DEVICE)
        T_t = torch.tensor(df_clean["T"].values, dtype=torch.float64, device=HES_DEVICE)
        C_mkt_t = torch.tensor(df_clean["C_mkt"].values, dtype=torch.float64, device=HES_DEVICE)

        weights_np = 1.0 / (np.abs(moneyness - 1.0) + 1e-3)
        weights_np = np.clip(weights_np / weights_np.mean(), 0.5, 5.0)
        weights_t = torch.tensor(weights_np, dtype=torch.float64, device=HES_DEVICE)

        u = torch.zeros(5, dtype=torch.float64, device=HES_DEVICE, requires_grad=True)
        optimizer = torch.optim.Adam([u], lr=lr)

        for iteration in range(max_iters):
            optimizer.zero_grad()
            loss_val = heston_nn_loss(u, S0_t, K_t, T_t, C_mkt_t, r, q, weights=weights_t)
            loss_val.backward()
            optimizer.step()
            if progress_callback:
                progress_callback(iteration + 1, max_iters, float(loss_val.detach().cpu()))

        return HestonParams.from_unconstrained(u[0], u[1], u[2], u[3], u[4])


    def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(S - K, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)


    def bs_put(S: float, K: float, T: float, r: float, sigma: float) -> float:
        if T <= 0 or sigma <= 0:
            return max(K - S, 0)
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


    def implied_vol_option(price: float, S: float, K: float, T: float, r: float, option_type: str = "call", tol: float = 1e-6, max_iter: int = 100) -> float:
        if T < MIN_IV_MATURITY:
            return np.nan
        intrinsic = max(S - K, 0) if option_type == "call" else max(K - S, 0)
        if price <= intrinsic:
            return np.nan
        sigma = 0.3
        for _ in range(max_iter):
            price_est = bs_call(S, K, T, r, sigma) if option_type == "call" else bs_put(S, K, T, r, sigma)
            diff = price_est - price
            if abs(diff) < tol:
                return sigma
            d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
            vega = S * norm.pdf(d1) * math.sqrt(T)
            if vega < 1e-10:
                return np.nan
            sigma = sigma - diff / vega
            if sigma <= 0:
                return np.nan
        return np.nan


    def build_market_surface(
        df: pd.DataFrame,
        price_col: str,
        option_type: str,
        kk_grid: np.ndarray,
        tt_grid: np.ndarray,
        rf_rate: float,
    ) -> np.ndarray | None:
        df = df.dropna(subset=[price_col]).copy()
        df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
        if len(df) < 5:
            return None
        df["iv_calc"] = df.apply(
            lambda row: implied_vol_option(
                row[price_col], row["S0"], row["K"], row["T"], rf_rate, option_type
            ),
            axis=1,
        )
        df = df.dropna(subset=["iv_calc"])
        if df.empty:
            return None
        pts = df[["K", "T"]].to_numpy()
        vals = df["iv_calc"].to_numpy()
        surf = griddata(pts, vals, (kk_grid, tt_grid), method="linear")
        if surf is None or np.all(np.isnan(surf)):
            surf = griddata(pts, vals, (kk_grid, tt_grid), method="nearest")
        else:
            mask = np.isnan(surf)
            if mask.any():
                surf[mask] = griddata(pts, vals, (kk_grid[mask], tt_grid[mask]), method="nearest")
        return surf


    def build_market_price_grid(
        df: pd.DataFrame,
        price_col: str,
        kk_grid: np.ndarray,
        tt_grid: np.ndarray,
    ) -> np.ndarray | None:
        df = df.dropna(subset=[price_col]).copy()
        df = df[(df["T"] >= MIN_IV_MATURITY) & (df[price_col] > 0)]
        if len(df) < 5:
            return None
        pts = df[["K", "T"]].to_numpy()
        vals = df[price_col].to_numpy()
        grid = griddata(pts, vals, (kk_grid, tt_grid), method="linear")
        if grid is None or np.all(np.isnan(grid)):
            grid = griddata(pts, vals, (kk_grid, tt_grid), method="nearest")
        else:
            mask = np.isnan(grid)
            if mask.any():
                grid[mask] = griddata(pts, vals, (kk_grid[mask], tt_grid[mask]), method="nearest")
        return grid


    def render_section_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_general_definition_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_method_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_inputs_explainer(title: str, body: str) -> None:
        """No-op (explications cachées)."""
        return


    def render_unlock_sidebar_button(context_key: str, label: str) -> None:
        """Affiche un bouton permettant de réactiver l'input T lorsque Heston a verrouillé la barre latérale."""
        if st.session_state.get("heston_tab_locked"):
            if st.button(label, key=f"unlock_sidebar_{context_key}"):
                st.session_state["heston_tab_locked"] = False
                st.rerun()


    ASIAN_LATEX_DERIVATION = r"""
    **Modèle sous la mesure risque-neutre**

    Sous la mesure risque-neutre $\mathbb{Q}$, le sous-jacent suit
    \[
    dS_t = (r-q)\,S_t\,dt + \sigma S_t\,dW_t, \qquad S_0>0,
    \]
    où $r$ est le taux sans risque, $q$ le dividende continu et $W_t$ un mouvement brownien standard.
    La solution explicite s'écrit
    \[
    S_t = S_0 \exp\Big[(r-q-\tfrac12\sigma^2)t + \sigma W_t\Big].
    \]

    **Option asiatique géométrique**

    On définit la moyenne géométrique continue
    \[
    G_T = \exp\!\left(\frac{1}{T}\int_0^T \ln S_t\,dt\right),
    \]
    et le payoff d'un call géométrique $(G_T-K)^+$.
    En partant de
    \[
    \ln S_t = \ln S_0 + (r-q-\tfrac12\sigma^2)t + \sigma W_t,
    \]
    on montre que
    \[
    \ln G_T = \frac{1}{T}\int_0^T \ln S_t\,dt
    = \ln S_0 + (r-q-\tfrac12\sigma^2)\frac{T}{2}
      + \sigma Y,
    \]
    avec
    \[
    Y = \frac{1}{T}\int_0^T W_t\,dt \sim \mathcal{N}\!\Big(0,\tfrac{T}{3}\Big).
    \]
    Ainsi, $\ln G_T$ est gaussien de moyenne $\mu_G$ et variance $v_G$ :
    \[
    \mu_G = \ln S_0 + (r-q-\tfrac12\sigma^2)\frac{T}{2},\qquad
    v_G = \sigma^2\frac{T}{3},
    \]
    ce qui implique que $G_T$ est lognormal. On introduit une volatilité effective
    \[
    \tilde{\sigma} = \frac{\sigma}{\sqrt{3}},
    \]
    et un niveau initial ajusté $\tilde{S}_0$ (obtenu à partir de la moyenne de $\ln G_T$) de sorte que le pricing du call géométrique s'écrive sous une forme de type Black--Scholes :
    \[
    C_0^{\mathrm{geom}} = \tilde{S}_0 e^{-qT} N(d_1) - K e^{-rT} N(d_2),
    \]
    avec
    \[
    d_1 = \frac{\ln(\tilde{S}_0/K) + (r-q + \tfrac12 \tilde{\sigma}^2)T}{\tilde{\sigma}\sqrt{T}},
    \qquad
    d_2 = d_1 - \tilde{\sigma}\sqrt{T}.
    \]

    **Option asiatique arithmétique et PDE associée**

    La moyenne arithmétique continue est
    \[
    A_T = \frac{1}{T}\int_0^T S_t\,dt,
    \]
    et le payoff du call arithmétique est $(A_T-K)^+$.
    Comme ce payoff dépend du chemin complet, on introduit le processus d'intégrale
    \[
    I_t = \int_0^t S_u\,du,
    \]
    de sorte que $A_T = I_T/T$.
    Le couple $(S_t,I_t)$ est markovien et suit
    \[
    dS_t = (r-q)S_t\,dt + \sigma S_t\,dW_t, \qquad dI_t = S_t\,dt.
    \]

    On définit la fonction de valeur
    \[
    V(t,s,i) = \mathbb{E}^{\mathbb{Q}}\!\left[e^{-r(T-t)}\Big(\tfrac{I_T}{T} - K\Big)^+ \,\big|\, S_t=s,\,I_t=i\right],
    \]
    avec condition terminale
    \[
    V(T,s,i) = \Big(\tfrac{i}{T} - K\Big)^+.
    \]
    Le générateur infinitésimal du couple $(S_t,I_t)$ est
    \[
    \mathcal{L}V = (r-q)s\,V_s + \tfrac12\sigma^2 s^2\,V_{ss} + s\,V_i,
    \]
    et, par le théorème de Feynman--Kac, $V$ vérifie la PDE de valorisation
    \[
    \frac{\partial V}{\partial t}
     + (r-q)s \frac{\partial V}{\partial s}
     + \tfrac12 \sigma^2 s^2 \frac{\partial^2 V}{\partial s^2}
     + s \frac{\partial V}{\partial i}
     - r V = 0,
    \]
    sur $[0,T)\times (0,\infty)\times (0,\infty)$, avec la condition terminale ci‑dessus.
    Cette PDE n'admet pas de solution fermée simple et doit être résolue numériquement (schémas aux différences finies, méthodes spectrales, approches Monte Carlo avancées).
    """


    def render_math_derivation(title: str, body_md: str) -> None:
        """Affiche un menu déroulant contenant la dérivation mathématique, rendue avec LaTeX."""
        with st.expander(title):
            st.markdown(body_md)


    def render_pdf_derivation(title: str, pdf_path: str, download_name: str | None = None) -> None:
        """
        Affiche, dans un menu déroulant, un PDF (par exemple une dérivation LaTeX compilée).
        Le PDF est encodé en base64 et inclus dans une balise <iframe>.
        """
        from pathlib import Path as _Path

        with st.expander(title):
            path = _Path(pdf_path)
            if not path.exists():
                st.info(
                    f"Le fichier PDF '{pdf_path}' n'a pas été trouvé. "
                    "Placez le PDF compilé à cet emplacement pour l'afficher ici."
                )
                return

            with path.open("rb") as f:
                base64_pdf = base64.b64encode(f.read()).decode("utf-8")

            pdf_display = f"""
    <iframe
        src="data:application/pdf;base64,{base64_pdf}"
        width="100%"
        height="700"
        type="application/pdf"
    ></iframe>
    """
            st.markdown(pdf_display, unsafe_allow_html=True)



    def ui_heston_full_pipeline(auto_run: bool = False):


        # Ticker input + fetch action stacked vertically (pour éviter un layout en colonnes)
        ticker = st.text_input(
            "Ticker (sous-jacent)",
            value=st.session_state.get("tkr_common", "SPY"),
            key="heston_cboe_ticker",
            help="Code du sous-jacent coté au CBOE utilisé pour la calibration Heston.",
        ).strip().upper()
        st.session_state["tkr_common"] = ticker
        st.session_state["common_underlying"] = ticker
        rf_rate = float(st.session_state.get("common_rate", 0.02))
        div_yield = float(st.session_state.get("common_dividend", 0.0))
        fetch_btn = st.button("Récupérer les données du ticker", type="primary", key="heston_cboe_fetch")

        col_cfg1, col_cfg2 = st.columns(2)
        with col_cfg1:
            pass  # réservé pour d’autres réglages si nécessaire
        with col_cfg2:
            span_mc = float(st.session_state.get("heatmap_span_value", 20.0))
            n_maturities = 40


        state = st.session_state
        if "heston_calls_df" not in state:
            state.heston_calls_df = None
            state.heston_puts_df = None
            state.heston_S0_ref = None
            state.heston_calib_T_target = None

        auto_fetch = (
            not st.session_state.get("heston_cboe_loaded_once", False)
            and not st.session_state.get("heston_cboe_auto_fetched", False)
        )
        if auto_fetch:
            st.session_state["heston_cboe_auto_fetched"] = True
        fetch_btn = fetch_btn or auto_fetch
        st.divider()

        if fetch_btn:
            try:
                calls_df, puts_df, S0_ref, rf_rate, div_yield = load_cboe_data(ticker)
                state.heston_calls_df = calls_df
                state.heston_puts_df = puts_df
                state.heston_S0_ref = S0_ref
                st.session_state["common_rate"] = float(rf_rate)
                st.session_state["common_dividend"] = float(div_yield)
                st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
                st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
                maturity_list = sorted(calls_df["T"].round(2).unique().tolist())
                st.session_state["cboe_T_options"] = maturity_list
                st.session_state["sidebar_maturity_options"] = maturity_list
                span_sync = float(st.session_state.get("heatmap_span_value", 20.0))
                if maturity_list:
                    rnd_T = float(np.random.choice(maturity_list))
                else:
                    rnd_T = float(round(calls_df["T"].iloc[0], 2))
                eligible_calls = calls_df[
                    (calls_df["T"].round(2) == rnd_T)
                    & calls_df["K"].between(S0_ref - span_sync, S0_ref + span_sync)
                ]
                if eligible_calls.empty:
                    eligible_calls = calls_df[
                        calls_df["K"].between(S0_ref - span_sync, S0_ref + span_sync)
                    ]
                if eligible_calls.empty:
                    eligible_calls = calls_df
                chosen_row = eligible_calls.sample(1).iloc[0]
                chosen_K = float(chosen_row["K"])
                chosen_T = float(round(chosen_row["T"], 2))
                sigma_pick = float(chosen_row.get("iv_market") or np.nan)
                if not np.isfinite(sigma_pick):
                    sigma_pick = implied_vol_option(
                        float(chosen_row.get("C_mkt", np.nan)),
                        float(chosen_row.get("S0", S0_ref)),
                        chosen_K,
                        float(chosen_row["T"]),
                        rf_rate,
                        "call",
                    )
                if not np.isfinite(sigma_pick):
                    sigma_pick = float(st.session_state.get("sigma_common", 0.2))
                prefills = {
                    "S0_common": float(S0_ref),
                    "K_common": chosen_K,
                    "sigma_common": float(np.clip(sigma_pick, 0.01, 5.0)),
                }
                st.session_state["heston_sidebar_prefill"] = prefills
                st.session_state["heston_sidebar_placeholders"] = {
                    "S0_common": f"{prefills['S0_common']:.2f}",
                    "K_common": f"{prefills['K_common']:.2f}",
                    "sigma_common": f"{prefills['sigma_common']:.4f}",
                }
                st.session_state["heston_cboe_loaded_once"] = True
                st.rerun()
            except Exception as exc:
                st.error(f"❌ Erreur lors du téléchargement des données CBOE : {exc}")

        calls_df = state.heston_calls_df
        puts_df = state.heston_puts_df
        S0_ref = state.heston_S0_ref
        calib_T_target = state.heston_calib_T_target

        calib_band_range: tuple[float, float] | None = None
        calib_T_band = 0.4
        max_iters = 1000
        learning_rate = 0.005

        if calls_df is not None and puts_df is not None and S0_ref is not None:
            col_nn, col_modes = st.columns(2)
            with col_nn:
                st.subheader("🎯 Calibration NN Carr-Madan")
                calib_T_band = st.number_input(
                    "Largeur bande T (±)",
                    value=0.04,
                    min_value=0.01,
                    max_value=0.5,
                    step=0.01,
                    format="%.2f",
                    key="heston_cboe_calib_band",
                    help="Largeur de la bande de maturités autour de la cible utilisée pour la calibration.",
                )

                unique_T = sorted(calls_df["T"].round(2).unique().tolist())
                if unique_T:
                    if calib_T_target is None:
                        target_guess = max(MIN_IV_MATURITY, unique_T[0] + calib_T_band + 0.1)
                        idx_default = int(np.argmin(np.abs(np.array(unique_T) - target_guess)))
                    else:
                        try:
                            idx_default = unique_T.index(calib_T_target)
                        except ValueError:
                            idx_default = 0

                    calib_T_target = st.selectbox(
                        "Maturité T cible pour la calibration (Time to Maturity)",
                        unique_T,
                        index=idx_default,
                        format_func=lambda x: f"{x:.2f}",
                        key="heston_cboe_calib_target",
                        help="Maturité autour de laquelle la calibration Heston est centrée.",
                    )
                    state.heston_calib_T_target = calib_T_target
                else:
                    st.warning("Pas de maturités disponibles dans les données CBOE.")
                    calib_T_target = None

                with col_modes:
                    st.subheader("⚙️ Modes de calibration NN")
                    mode = st.radio(
                        "Choisir un mode",
                        ["Rapide", "Bonne", "Excellente"],
                        index=0,
                        horizontal=True,
                        key="heston_cboe_mode",
                        help="Choisit un compromis entre vitesse de calibration et précision de l’ajustement.",
                )
                if mode == "Rapide":
                    max_iters = 300
                    learning_rate = 0.01
                elif mode == "Bonne":
                    max_iters = 1000
                    learning_rate = 0.005
                else:
                    max_iters = 2000
                    learning_rate = 0.001
                st.markdown(
                    f"**Itérations NN** : `{max_iters}`  \n"
                    f"**Learning rate** : `{learning_rate}`"
                )

            if calib_T_target is not None:
                calib_band_range = (
                    max(MIN_IV_MATURITY, calib_T_target - calib_T_band),
                    calib_T_target + calib_T_band,
                )
            else:
                calib_band_range = None

        run_button = False
        if calls_df is not None and puts_df is not None and S0_ref is not None:
            # Only explicit click launches calibration; changing K/T won't auto-relance
            run_button = st.button("🚀 Lancer l'analyse", type="primary", width="stretch", key="heston_cboe_run")
            st.divider()

        if run_button:
            if calls_df is None or puts_df is None or S0_ref is None:
                st.error("Veuillez d'abord cliquer sur 'Récupérer les données du ticker'.")
                return
            if calib_band_range is None or calib_T_target is None:
                st.error("Veuillez choisir une maturité T cible après avoir chargé les données.")
                return

            try:
                st.info(f"📡 Données CBOE chargées pour {ticker} (cache)")
                st.success(f"{len(calls_df)} calls, {len(puts_df)} puts | S0 ≈ {S0_ref:.2f}")
                st.write(f"Maturité T cible pour la calibration : {calib_T_target:.2f} ans")

                st.info("🧠 Calibration ciblée...")
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                loss_log: list[float] = []

                def progress_cb(current: int, total: int, loss_val: float) -> None:
                    progress_bar.progress(current / total)
                    status_text.text(f"⏳ Iter {current}/{total} | Loss = {loss_val:.6f}")
                    loss_log.append(loss_val)

                calib_slice = calls_df[
                    (calls_df["T"].round(2).between(*calib_band_range))
                    & (calls_df["K"].between(S0_ref - span_mc, S0_ref + span_mc))
                    & (calls_df["C_mkt"] > 0.05)
                    & (calls_df["iv_market"] > 0)
                ]
                if len(calib_slice) < 5:
                    calib_slice = calls_df.copy()

                params_cm = calibrate_heston_nn(
                    calib_slice,
                    r=rf_rate,
                    q=div_yield,
                    max_iters=int(max_iters),
                    lr=learning_rate,
                    spot_override=S0_ref,
                    progress_callback=progress_cb,
                )
                progress_bar.empty()
                status_text.empty()

                params_dict = {
                    "kappa": float(params_cm.kappa.detach()),
                    "theta": float(params_cm.theta.detach()),
                    "sigma": float(params_cm.sigma.detach()),
                    "rho": float(params_cm.rho.detach()),
                    "v0": float(params_cm.v0.detach()),
                }
                state.heston_params_cm = params_cm
                state.heston_params_meta = {
                    "r": rf_rate,
                    "q": div_yield,
                    "S0_ref": float(S0_ref),
                }
                # Met à jour les paramètres Heston globaux, qui alimentent la sidebar
                st.session_state["heston_kappa_common"] = params_dict["kappa"]
                st.session_state["heston_theta_common"] = params_dict["theta"]
                st.session_state["heston_eta_common"] = params_dict["sigma"]
                st.session_state["heston_rho_common"] = params_dict["rho"]
                st.session_state["heston_v0_common"] = params_dict["v0"]
                st.success("✓ Calibration terminée")
                st.dataframe(pd.Series(params_dict, name="Paramètre").to_frame())

                # Fin : on s'arrête après calibration, pas de surfaces IV/heatmaps pour alléger l'affichage.

                st.balloons()
                st.success("🎉 Analyse terminée")

            except Exception as exc:
                st.error(f"❌ Erreur : {exc}")
                import traceback

                st.code(traceback.format_exc())
    # ---------------------------------------------------------------------------
    #  Application Streamlit unifiée
    # ---------------------------------------------------------------------------


    sidebar_prefill = st.session_state.pop("heston_sidebar_prefill", None)
    if sidebar_prefill:
        for key, value in sidebar_prefill.items():
            st.session_state[key] = value


    st.markdown(
        """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        div[data-testid="stStatusWidget"] {display: none;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Initial defaults for shared parameters
    placeholder_vals = st.session_state.get("heston_sidebar_placeholders", {})
    heston_tab_locked = st.session_state.get("heston_tab_locked", False)

    default_values = {
        "S0_common": 100.0,
        "K_common": 100.0,
        "T_common": 1.0,
        "sigma_common": 0.2,
        "r_common": 0.05,
        "d_common": 0.0,
        "heatmap_span": 25.0,
        "heston_kappa_common": 2.0,
        "heston_theta_common": 0.04,
        "heston_eta_common": 0.5,
        "heston_rho_common": -0.7,
        "heston_v0_common": 0.04,
    }
    for k, v in default_values.items():
        st.session_state.setdefault(k, v)
    st.session_state.setdefault("heston_cboe_loaded_once", False)

    ui_heston_full_pipeline()

    st.markdown("### Paramètres de Black-Scholes-Merton / Heston")

    # Si les données CBOE ne sont pas chargées, on continue avec les valeurs par défaut,
    # tout en signalant qu'une récupération est recommandée pour les modules Heston.
    if not st.session_state.get("heston_cboe_loaded_once", False):
        st.info('Clique sur "Récupérer les données du ticker" pour alimenter la calibration Heston (facultatif).')

    if not st.session_state.get("heston_cboe_loaded_once", False):
        st.info("Charge d'abord les données CBOE (via le bloc de calibration Heston) pour afficher les paramètres.")
        mat_options = [st.session_state.get("T_common", 1.0)]
    else:
        mat_options = st.session_state.get("cboe_T_options")
        if not mat_options:
            # Essaie de reconstruire la liste depuis les données CBOE déjà en cache
            calls_df = st.session_state.get("heston_calls_df")
            puts_df = st.session_state.get("heston_puts_df")
            frames = []
            if calls_df is not None:
                frames.append(calls_df[["T"]])
            if puts_df is not None:
                frames.append(puts_df[["T"]])
            if frames:
                mat_options = sorted(pd.concat(frames, axis=0)["T"].dropna().round(2).unique().tolist())
                st.session_state["cboe_T_options"] = mat_options
            else:
                mat_options = [st.session_state.get("T_common", 1.0)]

    col_left, col_right = st.columns(2)

    def _safe_float(val, default):
        try:
            return float(default if val is None else val)
        except Exception:
            return float(default)

    with col_left:
        # Spot provenant des données CBOE (chargées)
        S0_common = _safe_float(st.session_state.get("heston_S0_ref"), _safe_float(st.session_state.get("S0_common"), 100.0))
        st.markdown(f"**S0 (spot CBOE)** : {S0_common:.4f}")
        # La maturité de référence pour les calculs est la maturité cible choisie pour la calibration
        T_common = _safe_float(st.session_state.get("heston_calib_T_target"), _safe_float(st.session_state.get("T_common"), 1.0))
        st.session_state["T_common"] = T_common
        st.markdown(f"**T (maturité, années) — cible calibration** : {T_common:.4f}")
        # K issu des strikes CBOE pour T sélectionné
        K_common = st.session_state.get("K_common", float(S0_common))
        K_cboe_options: list[float] = []
        state = st.session_state
        calls_df = state.get("heston_calls_df")
        puts_df = state.get("heston_puts_df")
        sel_T = float(T_common)
        if calls_df is not None:
            K_cboe_options.extend(calls_df[calls_df["T"].round(2) == round(sel_T, 2)]["K"].tolist())
        if puts_df is not None:
            K_cboe_options.extend(puts_df[puts_df["T"].round(2) == round(sel_T, 2)]["K"].tolist())
        K_cboe_options = sorted(set(K_cboe_options))
        if K_cboe_options:
            # Choix du strike proche du spot privilégié
            default_idx = int(np.argmin(np.abs(np.array(K_cboe_options) - float(S0_common))))
            K_pick = st.selectbox(
                "K (strike) – CBOE",
                options=K_cboe_options,
                index=default_idx,
                format_func=lambda x: f"{x:.2f}",
                key="K_common_select",
                help="Strikes disponibles pour la maturité sélectionnée (T maître).",
            )
            K_common = float(K_pick)
            st.session_state["K_common"] = K_common
            # Cherche une IV correspondante et met à jour sigma
            iv_pick = np.nan
            price_pick = np.nan
            if calls_df is not None:
                sub = calls_df[calls_df["T"].round(2) == round(sel_T, 2)]
                row_call = sub.loc[(sub["K"] - K_common).abs().idxmin()] if not sub.empty else None
                if row_call is not None:
                    iv_pick = float(row_call.get("iv_market", np.nan))
                    price_pick = float(row_call.get("C_mkt", np.nan))
            if (not np.isfinite(iv_pick)) and puts_df is not None:
                subp = puts_df[puts_df["T"].round(2) == round(sel_T, 2)]
                row_put = subp.loc[(subp["K"] - K_common).abs().idxmin()] if not subp.empty else None
                if row_put is not None:
                    iv_pick = float(row_put.get("iv_market", np.nan))
                    if not np.isfinite(price_pick):
                        price_pick = float(row_put.get("P_mkt", np.nan))
            if not np.isfinite(iv_pick):
                # tentative de calcul à partir du prix si disponible
                S_ref = float(state.get("heston_S0_ref", S0_common))
                if np.isfinite(price_pick):
                    iv_pick = implied_vol_option(price_pick, S_ref, K_common, sel_T, float(state.get("common_rate", r_common)), "call")
            if np.isfinite(iv_pick) and iv_pick > 0:
                st.session_state["sigma_common"] = float(iv_pick)
                st.caption(f"IV CBOE retenue pour T={sel_T:.2f}, K={K_common:.2f} : σ ≈ {iv_pick:.4f}")
            else:
                st.caption("IV non trouvée pour ce strike/maturité, utilisez σ ci-dessous.")
        else:
            st.warning("Aucun strike CBOE pour la maturité sélectionnée. Ajustez T ou renseignez K/σ manuellement.")
            st.session_state["K_common"] = K_common
        # Volatilité déduite (ou fallback session)
        sigma_common = float(st.session_state.get("sigma_common", 0.2))
        st.markdown(f"**σ (IV déduite)** : {sigma_common:.4f}")
        r_common = max(float(st.session_state.get("common_rate", 0.0)), 1e-6)
        d_common = float(st.session_state.get("common_dividend", 0.0))
        st.markdown(f"**r (risk-free CBOE)** : {r_common:.4f}")
        st.markdown(f"**q (dividende continu CBOE)** : {d_common:.4f}")
        heatmap_span = float(25.0)
        st.markdown(f"**Span autour du spot (heatmaps)** : {heatmap_span:.1f}")

    with col_right:
        st.markdown("Paramètres Heston communs")
        heston_kappa_common = float(st.session_state.get("heston_kappa_common", 2.0))
        heston_theta_common = float(st.session_state.get("heston_theta_common", 0.04))
        heston_eta_common = float(st.session_state.get("heston_eta_common", 0.5))
        heston_rho_common = float(st.session_state.get("heston_rho_common", -0.7))
        heston_v0_common = float(st.session_state.get("heston_v0_common", 0.04))
        common_rdisp = float(st.session_state.get("common_rate", r_common))
        common_qdisp = float(st.session_state.get("common_dividend", d_common))

        st.markdown(
            f"""
            - κ = **{heston_kappa_common:.4f}**
            - θ = **{heston_theta_common:.4f}**
            - η = **{heston_eta_common:.4f}**
            - ρ = **{heston_rho_common:.4f}**
            - v0 = **{heston_v0_common:.4f}**
            - S₀ (calibration) = **{float(st.session_state.get("common_spot", S0_common)):.4f}**
            - r = **{common_rdisp:.4f}**
            - q = **{common_qdisp:.4f}**
            """
        )

    # Rafraîchir les valeurs en session (issues de la calibration ou des défauts)
    st.session_state["heston_kappa_common"] = heston_kappa_common
    st.session_state["heston_theta_common"] = heston_theta_common
    st.session_state["heston_eta_common"] = heston_eta_common
    st.session_state["heston_rho_common"] = heston_rho_common
    st.session_state["heston_v0_common"] = heston_v0_common

    heatmap_spot_values = _heatmap_axis(S0_common, heatmap_span)
    heatmap_strike_values = _heatmap_axis(K_common, heatmap_span)
    heatmap_maturity_span = float(max(0.01, T_common * 0.5))
    heatmap_maturity_values = _heatmap_axis(T_common, heatmap_maturity_span)

    common_spot_value = float(S0_common)
    common_maturity_value = float(T_common)
    common_strike_value = float(K_common)
    common_rate_value = float(r_common)
    common_sigma_value = float(sigma_common)

    st.session_state["common_spot"] = common_spot_value
    st.session_state["common_strike"] = common_strike_value
    st.session_state["common_maturity"] = common_maturity_value
    st.session_state["common_sigma"] = common_sigma_value
    st.session_state["common_rate"] = common_rate_value
    st.session_state["common_dividend"] = float(d_common)
    st.session_state["heatmap_span_value"] = float(heatmap_span)
    st.session_state["heatmap_maturity_span_value"] = float(heatmap_maturity_span)

    st.markdown("---")
    st.subheader("Historique 1 an du ticker (prix de clôture)")
    hist_fig = None
    tkr_hist = st.session_state.get("heston_cboe_ticker", st.session_state.get("tkr_common", "")).strip().upper()
    if not tkr_hist:
        st.info("Charge un ticker via la calibration Heston pour afficher l'historique 1 an.")
    else:
        hist_df = pd.DataFrame()
        try:
            cli_path = SCRIPTS_DIR / "fetch_history_cli.py"
            result = subprocess.run(
                [sys.executable, str(cli_path), "--ticker", tkr_hist, "--period", "1y", "--interval", "1d"],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0 and result.stdout:
                hist_df = pd.read_csv(io.StringIO(result.stdout))
                if "Date" in hist_df.columns:
                    hist_df["Date"] = pd.to_datetime(hist_df["Date"])
                    hist_df.set_index("Date", inplace=True)
            elif result.returncode != 0:
                st.warning("Impossible de récupérer l'historique 1 an (via CLI).")
        except Exception as _hist_err:
            st.warning(f"Impossible de récupérer l'historique 1 an : {_hist_err}")

        if not hist_df.empty and "Close" in hist_df.columns:
            hist_fig = go.Figure()
            hist_fig.add_trace(
                go.Scatter(
                    x=hist_df.index,
                    y=hist_df["Close"],
                    mode="lines",
                    name="Close",
                )
            )
            # Ensure datetime index for proper ticks
            idx_dt = pd.to_datetime(hist_df.index)
            start_dt = idx_dt.min()
            end_dt = idx_dt.max()
            start_label = start_dt.strftime("%Y-%m-%d") if hasattr(start_dt, "strftime") else str(start_dt)
            end_label = end_dt.strftime("%Y-%m-%d") if hasattr(end_dt, "strftime") else str(end_dt)

            hist_fig.update_layout(
                title=f"{tkr_hist} - Close (1 an)",
                xaxis_title="Date",
                yaxis_title="Prix",
                shapes=[
                    dict(
                        type="line",
                        x0=0,
                        x1=1,
                        y0=K_common,
                        y1=K_common,
                        xref="paper",
                        yref="y",
                        line=dict(color="red", width=2, dash="dash"),
                    )
                ],
                annotations=[
                    dict(
                        x=hist_df.index.max(),
                        y=K_common,
                        xanchor="left",
                        yanchor="bottom",
                        text=f"K = {K_common:.2f}",
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=0,
                        font=dict(color="red"),
                    )
                ],
            )
            st.plotly_chart(hist_fig, width="stretch")
        else:
            st.info("Pas d'historique disponible pour ce ticker.")

    def render_option_tabs_for_type(option_label: str, option_char: str):
        # Quick payoff helper for dropdown explanations.
        def _payoff_plot(x_vals, y_vals, title, strike_lines=None):
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode="lines", name="Payoff"))
            if strike_lines:
                for sl in strike_lines:
                    fig.add_shape(
                        type="line",
                        x0=sl,
                        x1=sl,
                        y0=min(y_vals),
                        y1=max(y_vals),
                        line=dict(color="red", dash="dash"),
                    )
            fig.update_layout(title=title, xaxis_title="Spot à maturité", yaxis_title="Payoff")
            return fig

        def _graph_desc(key_suffix: str) -> str:
            """Retourne une description textuelle (10 lignes) du payoff/produit."""
            desc_map = {
                "american_payoff": """Une option américaine peut être exercée à tout moment entre l’émission et l’échéance.
Le payoff terminal reste celui d’un call ou d’un put vanilla, mais l’acheteur peut l’activer plus tôt.
Sa valeur est toujours au moins égale à celle d’une européenne, car l’option d’exercice supplémentaire a un prix.
Un call américain sur un sous-jacent sans dividende ne s’exerce généralement pas avant T, mais un put peut être anticipé.
Le dividende pousse parfois à lever un call juste avant le détachement, et le taux rend l’exercice du put attractif.
La sensibilité au spot (delta) est souvent plus élevée près du strike, surtout pour les puts ITM.
La convexité (gamma) se concentre autour du strike et peut être accentuée par l’option d’exercice.
Le theta peut devenir moins négatif, voire positif pour certains puts ITM proches de l’échéance.
La volatilité implicite joue sur la valeur de l’option de flexibilité d’exercice, donc sur la prime.
Le profil final est celui d’un payoff vanilla au strike K, mais la décision d’exercice modifie la valeur actuelle.""",
                "bermuda_payoff": """Une option bermudéenne n’est exerçable que sur un ensemble de dates discrètes prédéfinies.
Elle se situe entre une européenne (exercice à T) et une américaine (exercice continu), offrant une flexibilité intermédiaire.
Si aucune date d’exercice anticipé n’est utilisée, le payoff terminal reste celui d’une vanilla au strike K.
Plus il y a de dates d’exercice, plus la valeur se rapproche de l’américaine, car la flexibilité augmente.
Le calendrier des dates influence le choix optimal d’exercice et donc le prix.
Le delta et le theta se situent entre ceux d’une européenne et d’une américaine.
La vega est proche de celle d’une européenne, mais l’impact de la vol sur l’exercice reste présent.
Les dividendes et le niveau des taux guident la décision d’exercer, surtout pour les puts.
Le produit convient aux investisseurs voulant un compromis entre flexibilité et coût.
Le payoff final reste celui d’un call ou d’un put vanilla si l’option n’est pas exercée avant l’échéance.""",
                "asian_graph": """Une option asiatique arithmétique paie sur la moyenne des prix observés, plutôt que sur le spot final.
Ce moyennage réduit l’impact des pics de volatilité et rend souvent l’option moins chère qu’une européenne.
Le nombre d’observations et leur calendrier déterminent la sensibilité au chemin et le niveau de prime.
Le delta est plus lissé qu’une vanilla, car chaque observation influence partiellement le payoff.
Le gamma est plus modéré et la vega généralement plus faible, car l’effet de la vol est dilué dans la moyenne.
Le theta dépend de la cadence des fixings et de la portion déjà observée.
Le payoff final reste de type call ou put, mais sur la moyenne plutôt que sur le spot terminal.
Les options asiatiques sont prisées pour couvrir un prix moyen d’achat ou de vente d’une marchandise.
Elles limitent le risque de mauvais timing tout en conservant une protection ou un levier.
Le graphe de payoff s’apparente à une vanilla mais décalée par l’effet de la moyenne.""",
                "asian_geo_graph": """Une option asiatique géométrique paie sur la moyenne géométrique des prix observés.
Cette moyenne est toujours inférieure ou égale à la moyenne arithmétique, ce qui rend la prime plus basse.
Une formule fermée existe souvent, ce qui facilite le calcul et la couverture.
Le produit est moins sensible aux extrêmes de prix que l’asiatique arithmétique.
Le delta reste lissé et le gamma modéré, car chaque observation pèse de manière multiplicative.
La vega est plus faible que pour l’asiatique arithmétique, l’effet de la vol étant atténué.
Le theta dépend du nombre d’observations et de la partie déjà réalisée.
La géométrique sert fréquemment de contrôle de variance pour des simulations Monte Carlo.
Le payoff final est de type call ou put sur cette moyenne géométrique.
Le graphe de payoff suit une forme vanilla, décalée par la moyenne géométrique plutôt que par le spot final.""",
                "lookback_graph": """Une lookback floating détermine son strike a posteriori en prenant l’extrême du spot pendant la vie de l’option.
Pour un call, le strike peut être le minimum observé ; pour un put, le maximum, ce qui protège contre un mauvais timing.
Le payoff final compare le spot terminal à cet extrême, offrant une convexité forte et une protection renforcée.
Le produit est fortement path-dependent, car chaque observation peut modifier l’extrême retenu.
Le delta dépend de la distance entre le spot et l’extrême ; il peut réagir vivement lors de nouveaux extrêmes.
La vega et le theta diffèrent d’une vanilla car le temps supplémentaire augmente la probabilité d’atteindre des extrêmes.
La maturité accroît la chance d’observer un nouveau max ou min, ce qui influence le prix.
La valorisation se fait souvent par Monte Carlo ou par formules fermées spécialisées.
Le risque de gap est important, car un saut peut fixer un extrême décisif.
Le payoff final reste comparable à une vanilla mais avec un strike déterminé par le parcours du sous-jacent.""",
                "lookback_fixed_graph": """Une lookback fixed fixe son strike dès l’origine, mais son payoff dépend du max ou du min atteint pendant la vie.
Le spot final est comparé à l’extrême historique, ce qui récompense une trajectoire favorable.
Le produit reste path-dependent, même si le strike ne bouge pas après l’émission.
Le delta reflète la distance entre le spot et l’extrême atteint, et peut évoluer brusquement lors d’un nouveau record.
La vega intègre l’incertitude sur l’atteinte de nouveaux extrêmes, ce qui rend le prix sensible à la vol.
Le theta dépend de la probabilité d’étendre l’extrême pendant le temps restant.
Les méthodes de valorisation vont des approches semi-fermées aux Monte Carlo selon les hypothèses.
Le payoff final ressemble à un call ou put, mais il intègre la meilleure ou la pire performance passée.
Le produit convient pour capturer une amélioration de prix sans renoncer au strike initial.
Le graphe de payoff indique une vanilla évaluée avec l’information d’un extrême déjà observé ou attendu.""",
                "forward_start_graph": """Une forward-start choisit son strike à une date future T_start, souvent comme k fois le spot observé ce jour-là.
Avant T_start, l’option reste en attente et vaut une promesse d’option avec strike inconnu.
Après T_start, elle devient une vanilla classique, mais le strike est déjà lié au marché de cette date.
Le produit réduit le risque de mauvais timing sur le strike, utile pour des émissions ou des primes différées.
Le delta est faible avant T_start, car le strike n’est pas encore fixé, puis il augmente après la fixation.
La vega est répartie entre la période avant et après T_start, avec une sensibilité distincte à la vol.
Le theta dépend du délai restant jusqu’à la fixation du strike, puis de la maturité résiduelle.
Le prix dépend du ratio k et de la distribution anticipée du spot à T_start.
La couverture nécessite de suivre le spot pré-fixation pour ajuster l’exposition.
Le payoff final est un call ou un put vanilla évalué avec un strike défini plus tard, mais payé à l’échéance finale.""",
                "cliquet_graph": """Une option cliquet cumule des coupons périodiques capés et floorés, souvent avec réinitialisation du strike.
Chaque période mesure un rendement qui est limité par un cap et un floor, puis ajouté au coupon cumulé.
Le produit est fortement path-dependent, car la séquence de rendements successifs détermine le total payé.
La vega et la theta sont réparties sur toutes les périodes, plutôt que concentrées sur une seule échéance.
Le delta évolue à chaque période selon la distance au cap et au floor et le niveau de spot.
La valorisation se fait souvent par Monte Carlo, car les dépendances entre périodes sont complexes.
Le produit est utilisé pour distribuer un rendement régulier avec des limites de performance.
Les caps/floors protègent contre des chocs excessifs tout en offrant une participation à la hausse ou à la baisse.
Le risque de gap est atténué par les plafonds, mais la corrélation des rendements successifs reste importante.
Le payoff final correspond à la somme des coupons ajustés, payés à l’échéance ou à chaque période selon la structure.""",
                "calendar_graph": """Un calendar spread combine une option courte à maturité proche et une option longue à maturité plus lointaine, au même strike.
Le pari principal est que la valeur temps de la jambe longue dépasse celle de la jambe courte pendant la vie du trade.
Le theta est souvent positif au départ, car on encaisse la prime de l’échéance courte plus vite.
La vega est généralement positive, car la jambe longue réagit plus à une hausse de volatilité.
Le payoff est non linéaire autour du strike et dépend du comportement après l’expiration de la jambe courte.
Le gamma est concentré près de l’échéance proche, mais l’option longue conserve de la convexité au-delà.
Le spread peut être débouclé ou roulé après la première échéance, selon le mouvement du spot et de la vol.
Le produit est sensible au skew et à la term structure de volatilité entre les deux maturités.
Le risque est borné par les primes nettes, mais un mouvement rapide peut coûter la jambe courte.
Le payoff final reflète la valeur résiduelle de la jambe longue moins la jambe courte arrivée à échéance.""",
                "diagonal_graph": """Un diagonal spread mélange une différence de maturité et de strike entre les deux jambes.
Il combine les idées d’un vertical spread (strikes différents) et d’un calendar spread (maturités différentes).
La position permet d’ajuster une vue directionnelle tout en jouant sur la volatilité et le passage du temps.
Le theta dépend des deux échéances : la jambe courte s’érode plus vite, la jambe longue porte plus de valeur temps.
La vega est mixte, car les deux maturités réagissent différemment à un choc de volatilité.
Le profil peut viser une zone de spot spécifique où le spread délivre sa valeur maximale.
Le skew entre strikes et la term structure entre maturités influencent directement le prix.
La gestion active est souvent nécessaire après expiration de la jambe courte ou en cas de mouvement fort.
Le risque est borné par la combinaison de primes, mais le profil reste asymétrique.
Le payoff final est un différentiel de deux vanillas de strikes et de maturités distincts.""",
                "digital_graph": """Une option digitale cash-or-nothing paie un montant fixe si la condition sur le spot est satisfaite à l’échéance.
Pour un call, le paiement est déclenché si S_T dépasse K ; pour un put, s’il passe en dessous.
Le payoff ressemble à un saut de niveau, sans dépendre de l’ampleur du mouvement au-delà du strike.
La vega est concentrée autour du strike, car la probabilité de franchissement dépend fortement de la volatilité.
Le theta peut être abrupt près de l’échéance, la probabilité de franchissement évoluant vite.
Le gamma est également pic autour de K, car une petite variation de spot change la probabilité de paiement.
Le produit sert souvent de brique de base pour des structurés à coupon élevé.
Le risque de gap est important, car un saut peut déclencher ou annuler le paiement d’un coup.
La couverture nécessite de gérer une exposition sensible au spot autour du strike.
Le payoff final est binaire : soit le montant fixe est versé, soit rien n’est payé.""",
                "asset_on_graph": """Une option asset-or-nothing verse le sous-jacent lui-même (ou rien) si la condition est remplie.
Pour un call, l’actif est livré si S_T dépasse K ; pour un put, si S_T passe en dessous.
Le payoff est proportionnel au spot dans le scénario favorable, sinon nul, ce qui combine linéarité et condition binaire.
La sensibilité au spot reste forte, car le paiement dépend directement du niveau final de l’actif.
Le gamma et la vega sont concentrés autour du strike, mais l’ampleur du paiement varie avec S_T.
Le theta se comporte différemment d’une digitale cash car le montant dépend du spot final.
Le produit peut être couvert via des combinaisons de vanillas et de positions sur le sous-jacent.
Le risque de gap crée une discontinuité de valeur s’il survient près du strike.
La valorisation reste proche d’une digitale, mais avec un paiement multiplicatif par S_T.
Le payoff final est soit la remise de l’actif, soit zéro, selon la condition sur le spot.""",
                "chooser_graph": """Une option chooser offre à son détenteur le droit de choisir, à une date t_choice, entre un call et un put.
Avant t_choice, la prime reflète cette flexibilité supplémentaire, supérieure à celle d’une simple vanilla.
Après t_choice, le contrat se transforme en l’option choisie, avec le strike et la maturité restants.
Le produit couvre une incertitude directionnelle jusqu’à une date clé, tout en gardant une protection.
Le delta et la vega sont mixtes avant t_choice, car le profil intègre les deux scénarios potentiels.
Le theta dépend du temps restant avant la décision, puis du temps restant après la conversion.
La valorisation s’appuie souvent sur une combinaison analytique de call et de put avec un facteur de décision.
Le risque de gap avant t_choice peut influencer la préférence pour le call ou le put.
Le produit convient quand l’investisseur veut reporter un choix directionnel tout en verrouillant un coût.
Le payoff final est celui du call ou du put choisi, exercé à l’échéance finale selon le type sélectionné.""",
                "quanto_graph": """Une option quanto paie sur un sous-jacent étranger mais règle dans une autre devise à un taux de change fixé.
Le produit neutralise le risque de change tout en conservant l’exposition au sous-jacent de départ.
Le payoff reste celui d’un call ou d’un put vanilla, mais converti à un taux garanti ou ajusté.
La corrélation entre l’actif et le FX peut être prise en compte dans le drift effectif.
Le delta s’exprime sur le sous-jacent étranger, tandis que l’exposition FX est neutralisée.
La vega porte sur la volatilité de l’actif, l’effet de la vol FX étant limité par la construction.
Le theta reflète la valeur temps dans la devise de règlement, avec le taux de change figé.
La couverture se fait via le sous-jacent étranger et éventuellement des dérivés FX pour ajuster le drift.
Le produit est recherché quand l’investisseur veut éviter le risque de conversion tout en gardant la vue sur l’actif.
Le payoff final est une vanilla en devise locale, calculée sur la performance de l’actif étranger.""",
                "rainbow_graph": """Une option rainbow multi-actifs paie sur l’extrême d’un panier, le plus souvent le max ou le min.
Un call sur le max profite du meilleur performeur ; un put sur le min protège contre le pire actif.
La corrélation entre actifs est déterminante : faible corrélation augmente l’attrait du call sur max et du put sur min.
La vega est répartie entre les actifs, chaque volatilité contribuant à la probabilité d’être l’extrême.
Le delta se déplace vers l’actif le plus susceptible de finir max ou min, évoluant avec le marché.
Le produit ne somme pas les actifs : il choisit un extrême, ce qui rend le payoff non additif.
Les poids peuvent être asymétriques pour refléter des préférences sur certains actifs.
La couverture nécessite de suivre plusieurs sous-jacents et leur corrélation.
Le risque est lié aux sauts d’un actif qui peut soudain devenir l’extrême.
Le payoff final correspond à la valeur de l’actif extrême, comparée au strike pour un call ou un put.""",
                "barrier_graph": """Une option barrière adapte un payoff vanilla en le conditionnant à un niveau franchi ou non.
Une barrière knock-in active l’option si elle est touchée, tandis qu’une knock-out l’éteint.
La direction (up ou down) et le type (call ou put) définissent la zone de surveillance.
Plus la barrière est proche, plus la prime baisse pour une knock-out ou augmente pour une knock-in.
Le produit est path-dependent : la chronologie des niveaux atteints compte, pas seulement le final.
La vega et le theta sont influencés par la proximité de la barrière et par le temps restant.
Le risque de gap est crucial, car un saut peut déclencher ou annuler l’option instantanément.
Les paramètres clés sont le niveau de barrière, son sens, la nature in/out et le style call/put.
Le payoff final est celui d’une vanilla si la condition de barrière est respectée, sinon il est nul ou s’active.
Le produit est prisé pour réduire le coût ou pour cibler un scénario de range précis.""",
                "binary_barrier_graph": """Une barrière digitale paie un montant fixe selon le franchissement d’un niveau et la position finale du spot.
Elle combine la logique d’une barrière (up/down, in/out) et d’un payoff binaire.
La probabilité de paiement dépend du chemin et de la vol, avec un gamma concentré autour de la barrière et du strike.
Une knock-out annule le paiement si la barrière est touchée, alors qu’une knock-in l’active.
Le payout fixe simplifie le montant mais rend le risque de saut très important.
Les paramètres essentiels sont la barrière, sa direction, son type in/out, le montant fixe et l’éventuel strike.
La vega et le theta dépendent de la proximité de la barrière et du temps restant.
La couverture est délicate car la valeur saute quand la barrière est franchie.
Le produit est courant dans les structurés à rendement élevé où la probabilité de survie est rémunérée.
Le payoff final est une fonction en escalier conditionnée par le scénario de barrière et le niveau final.""",
                "basket_graph": """Une option basket paie sur une combinaison pondérée de plusieurs actifs, souvent une moyenne.
La corrélation module la diversification : plus elle est faible, plus la volatilité du panier est réduite.
Le payoff final est de type call ou put sur la valeur agrégée du panier.
La vega dépend des vols individuelles et de la corrélation, chaque composante contribuant au risque global.
Le delta est réparti selon les pondérations et peut se déplacer si un actif domine la variation du panier.
Le produit est path-indépendant si le payoff ne dépend que de la valeur terminale du panier.
Il est utilisé pour exprimer une vue macro ou sectorielle tout en lissant le risque idiosyncratique.
Le prix est sensible à la structure de corrélation : une hausse de corrélation augmente la vol du panier.
La couverture nécessite des rééquilibrages sur chaque actif composant.
Le graphe de payoff ressemble à une vanilla appliquée à la valeur agrégée du panier.""",
                "european_graph": """Une option européenne n’est exerçable qu’à l’échéance, avec un payoff vanilla au strike K.
La prime dépend des paramètres classiques S0, K, T, r, dividende et volatilité implicite.
L’absence d’exercice anticipé simplifie la valorisation et la couverture.
Un call sur un actif sans dividende ne s’exerce jamais avant T, car la valeur temps est conservée.
Le delta, le gamma, le theta et la vega suivent les profils standards du modèle Black-Scholes-Merton.
La surface de prix se lit en fonction de S et K autour des valeurs communes choisies.
Le risque pour l’acheteur est limité à la prime payée, le gain potentiellement illimité pour un call.
La put-call parity relie les prix call et put via le terme forward S0·e^{(r-q)T}.
La volatilité implicite mesurée sur ces options sert de référence pour d’autres produits.
Le payoff final est linéaire au-dessus ou en dessous du strike selon qu’il s’agit d’un call ou d’un put.""",
                "straddle_graph": """Un straddle combine l’achat simultané d’un call et d’un put au même strike K.
Le profil est symétrique et gagne en cas de forte hausse ou de forte baisse du sous-jacent.
Le coût initial est élevé, car on paie deux primes, ce qui crée une zone de perte autour de K.
Le delta initial est proche de zéro mais évolue rapidement dès que le spot s’éloigne du strike.
Le gamma est élevé près du strike, offrant une forte convexité autour de K.
La vega est positive, car une hausse de volatilité augmente la valeur des deux jambes.
Le theta est négatif et peut être important, car deux options perdent de la valeur temps.
Le break-even se situe à K ± (somme des primes), définissant la zone de profit.
Le produit sert à parier sur la magnitude d’un mouvement sans biais directionnel.
Le payoff final forme un V double, somme des payoffs call et put au même strike.""",
                "strangle_graph": """Un strangle achète un put OTM et un call OTM à deux strikes différents, plus éloignés que le spot.
Le coût est inférieur à celui d’un straddle, mais il faut un mouvement plus grand pour être gagnant.
La zone de perte est plus large autour du spot actuel, car les strikes sont décalés.
Le delta initial est proche de zéro mais évolue lorsque le spot s’approche d’un des strikes.
Le gamma est moins concentré qu’un straddle mais reste sensible autour des deux strikes.
La vega est positive, l’option profitant d’une hausse de volatilité qui augmente la probabilité de franchir un strike.
Le theta est négatif mais généralement plus faible qu’un straddle en valeur absolue.
Les break-even se situent autour de chaque strike plus la prime correspondante.
Le produit convient pour parier sur un mouvement important à moindre coût qu’un straddle.
Le payoff final présente deux régions de profit au-delà des strikes, avec une zone plate de perte entre eux.""",
                "call_spread_graph": """Un bull call spread achète un call à strike bas K1 et vend un call à strike plus haut K2.
La jambe courte finance partiellement la jambe longue, réduisant le coût par rapport à un call nu.
Le gain est plafonné au-dessus de K2, car la jambe vendue limite la participation à la hausse.
Le delta est positif mais inférieur à celui d’un call nu, et le gamma est modéré.
La vega est réduite, car une partie du risque de volatilité est vendue via la jambe courte.
Le theta peut être moins négatif, la prime encaissée compensant la perte de valeur temps.
La perte maximale est limitée à la prime nette payée, bornant le risque.
Le produit convient à une vue haussière modérée, avec budget de prime réduit.
Le profil reste directionnel, mais avec un plafond de gain défini par l’écart des strikes.
Le payoff final monte entre K1 et K2 puis se stabilise une fois le plafond atteint.""",
                "put_spread_graph": """Un bear put spread achète un put à strike haut K1 et vend un put à strike plus bas K2.
La jambe courte réduit le coût, mais le gain potentiel est plafonné sous K2.
Le delta est négatif mais plus faible en valeur absolue qu’un put nu, et le gamma est modéré.
La vega est atténuée, car une partie du risque de volatilité est cédée via la jambe vendue.
Le theta peut être moins négatif, la prime encaissée compensant l’érosion de la jambe longue.
La perte maximale est la prime nette payée, tandis que le gain maximal est (K1-K2) moins la prime.
Le produit cible une baisse modérée du sous-jacent plutôt qu’un scénario extrême.
La sensibilité au skew put est importante car les deux strikes peuvent se pricier différemment.
Le profil est borné : perte limitée au-dessus de K1, gain plafonné sous K2.
Le payoff final descend entre K1 et K2 puis se stabilise une fois le plancher atteint.""",
                "butterfly_graph": """Une butterfly call classique combine deux calls vendus au strike central et deux calls achetés aux ailes.
Le profil crée un pic de gain autour du strike médian, avec un coût relativement faible.
Le delta est proche de zéro au centre, tandis que le gamma est élevé près du strike central.
La vega est négative, car on est net vendeur de volatilité sur la zone centrale.
Le theta peut être positif autour du strike central, car la vente des options centrales s’érode plus vite.
Le risque est limité à la prime nette payée si le spot s’éloigne fortement.
Le gain maximal est atteint si le spot termine autour du strike médian à l’échéance.
Le produit convient pour parier sur un range étroit et une volatilité en baisse.
La largeur des ailes détermine l’ampleur du plateau de risque et de gain.
Le payoff final forme une tente autour du strike central, avec des pertes limitées en dehors.""",
                "condor_graph": """Un condor utilise quatre options à strikes échelonnés pour créer un plateau de gain plus large qu’une butterfly.
Les deux options centrales sont vendues, tandis que les deux ailes sont achetées pour borner le risque.
Le profil est plus plat et plus étendu au centre qu’une butterfly, avec un pic de gain moins aigu.
Le delta reste proche de zéro autour du centre, et le gamma est plus doux.
La vega est généralement négative, car la construction reste globalement short vol.
Le theta peut être positif dans la zone centrale, la valeur temps des jambes courtes se dégradant plus vite.
Le gain est limité entre les strikes intermédiaires, la perte est limitée aux extrêmes.
La construction peut être en crédit ou en débit selon le placement des strikes et le niveau de vol.
Le produit vise un range modéré plus large qu’une butterfly, avec un risque borné.
Le payoff final offre un plateau central avec des pentes plus douces vers les pertes aux ailes.""",
                "iron_condor_graph": """Un iron condor combine un put spread vendu et un call spread vendu pour encaisser un crédit initial.
Le plateau de gain se situe entre les strikes courts, tant que le spot reste dans le corridor.
La perte est limitée en dehors, bornée par la largeur des ailes longues achetées.
Le delta est proche de zéro au centre, le gamma modéré, ce qui convient aux vues de range.
La vega est négative, car la position est globalement short vol, et le theta est souvent positif au départ.
Le placement des ailes longues et courtes ajuste le compromis entre crédit encaissé et largeur de range.
Une hausse de volatilité ou un mouvement violent peut menacer le plateau et réduire le gain attendu.
Le trade peut être débouclé avant échéance si la vol se comprime et que le spot reste dans le range.
Le produit convient pour monétiser une attente de stagnation avec risque borné.
Le payoff final présente un plateau central de profit et des pertes plafonnées de part et d’autre.""",
                "iron_butterfly_graph": """Un iron butterfly vend un straddle au strike central et achète des ailes de protection plus éloignées.
Le crédit initial est maximal si le spot reste proche du strike central à l’échéance.
Les ailes longues bornent la perte en cas de mouvement extrême, rendant le risque limité.
Le delta est proche de zéro autour du centre, mais le gamma est élevé près du strike central.
La vega est négative et le theta positif, car la position vend de la valeur temps au cœur.
Le produit vise un range étroit et une volatilité en baisse, avec un crédit supérieur à l’iron condor.
Les break-even se situent autour du strike central plus ou moins la largeur des ailes moins le crédit net.
La gestion active peut être requise si le spot s’éloigne trop du centre.
La construction reste en crédit, avec une perte maximale bornée par l’écart ailes/centre.
Le payoff final est une tente inversée centrée sur le strike, avec profit au centre et pertes plafonnées aux extrêmes.""",
            }
            text = desc_map.get(key_suffix, "")
            if not text:
                return ""
            lines = text.strip("\n").splitlines()
            if len(lines) < 10:
                lines += [""] * (10 - len(lines))
            cleaned = [re.sub(r"^\s*\d+\)\s*", "", ln) for ln in lines[:10]]
            return "\n".join(cleaned)

        def _render_option_text(label: str, key_suffix: str):
            """Affiche un expander texte décrivant le payoff et la spécificité."""
            desc = _graph_desc(key_suffix)
            if not desc:
                return
            with st.expander(f"ℹ️ À propos – {label}", expanded=False):
                st.markdown(desc.replace("\n", "  \n"))

        def _render_payoff_dropdown(
            product: str,
            description: str,
            payoff_func,
            *,
            strike2_factor: float = 1.05,
            strike_lines_override: list[float] | Callable[[float, float], list[float]] | None = None,
            desc_key_suffix: str | None = None,
        ):
            with st.expander(f"🧭 Comprendre le payoff – {product}", expanded=False):
                K = float(st.session_state.get("common_strike_value", common_strike_value))
                K2 = float(st.session_state.get("common_strike_value", common_strike_value) * strike2_factor)
                xs = np.linspace(max(0.1, K * 0.5), K * 1.5, 100)
                ys = [payoff_func(x, K, K2) for x in xs]
                if strike_lines_override is None:
                    strikes_to_plot = [K]
                elif callable(strike_lines_override):
                    strikes_to_plot = strike_lines_override(K, K2)
                else:
                    strikes_to_plot = strike_lines_override
                fig = _payoff_plot(xs, ys, f"Payoff {product}", strike_lines=strikes_to_plot)
                st.plotly_chart(fig, width="stretch", key=_k(f"payoff_{product}"))
                st.caption(description)
                desc_key = desc_key_suffix or f"{product.lower().replace(' ', '_').replace('-', '_').replace('/', '_')}_graph"
                desc = _graph_desc(desc_key)
                if desc:
                    st.markdown(desc.replace("\n", "  \n"))

        # Helper to avoid duplicate Streamlit keys across Call/Put tabs.
        def _k(base: str) -> str:
            return f"{base}_{option_label.lower()}"
        # Helper for advanced structures, reused across dedicated tabs.
        def _render_structure_panel(structure_name: str):
            ks = structure_name.lower().replace(" ", "_").replace("/", "_").replace("-", "_")
            def kk(suffix: str) -> str:
                return _k(f"{ks}_{suffix}")

            st.subheader(structure_name)
            if structure_name == "Iron Condor":
                render_method_explainer(
                    "🪂 Construction de l’iron condor",
                    (
                        "- Quatre jambes européennes : long put bas, short put plus proche du spot, short call plus proche du spot, long call haut.\n"
                        "- Prix obtenu en sommant les primes BSM (positions achetées > positif ; positions vendues > négatif).\n"
                        "- Risque limité à l’écart entre ailes et jambes courtes, profit maximal égal au crédit net encaissé."
                    ),
                )
                wing_inner = st.number_input(
                    "Écart des strikes courts (autour de K commun)",
                    value=max(1.0, common_strike_value * 0.05),
                    min_value=0.1,
                    step=0.1,
                    key=kk("ic_wing_inner"),
                )
                wing_outer = st.number_input(
                    "Largeur des ailes (écart entre strike court et aile longue)",
                    value=max(1.0, common_strike_value * 0.05),
                    min_value=0.1,
                    step=0.1,
                    key=kk("ic_wing_outer"),
                )
                if st.button("Calculer l'Iron Condor (BSM)", key=kk("btn_iron_condor")):
                    K_mid = float(common_strike_value)
                    k_put_long = max(0.01, K_mid - (wing_inner + wing_outer))
                    k_put_short = max(0.01, K_mid - wing_inner)
                    k_call_short = K_mid + wing_inner
                    k_call_long = K_mid + wing_inner + wing_outer
                    try:
                        premium_put_long = _vanilla_price_with_dividend("put", common_spot_value, k_put_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        premium_put_short = _vanilla_price_with_dividend("put", common_spot_value, k_put_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        premium_call_short = _vanilla_price_with_dividend("call", common_spot_value, k_call_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        premium_call_long = _vanilla_price_with_dividend("call", common_spot_value, k_call_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        net_premium = premium_put_long - premium_put_short - premium_call_short + premium_call_long
                        credit = max(-net_premium, 0.0)
                        width = float(wing_outer)
                        max_profit = credit
                        max_loss = max(0.0, width - credit)
                        be_low = k_put_short - credit
                        be_high = k_call_short + credit
                        st.success(
                            f"Prime nette (achat>+ / vente>−) = {net_premium:.6f} "
                            f"{'(crédit)' if net_premium < 0 else '(débit)'}\n\n"
                            f"Max profit ≈ {max_profit:.6f} | Max perte ≈ {max_loss:.6f}\n"
                            f"Strikes : Put long {k_put_long:.2f} / Put short {k_put_short:.2f} / "
                            f"Call short {k_call_short:.2f} / Call long {k_call_long:.2f}\n"
                            f"Break-even bas ≈ {be_low:.4f} | Break-even haut ≈ {be_high:.4f}"
                        )
                    except Exception as exc:
                        st.error(f"Erreur lors du calcul Iron Condor : {exc}")
                return

            if structure_name == "Digital (cash-or-nothing)":
                payout = st.number_input("Payout", value=1.0, min_value=0.0, step=0.1, key=kk("payout"))
                if st.button("Pricer le digital", key=kk("btn")):
                    price = _digital_cash_or_nothing_price(
                        option_type=option_char,
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                        payout=payout,
                    )
                    st.success(f"Prix digital ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Digital (cash-or-nothing)",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_digital"),
                        spot=common_spot_value,
                        misc={
                            "payout": payout,
                            "style": "cash_or_nothing",
                        },
                    )
                return

            if structure_name == "Asset-or-nothing":
                if st.button("Pricer l'asset-or-nothing", key=kk("btn")):
                    price = _asset_or_nothing_price(
                        option_type=option_char,
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                    )
                    st.success(f"Prix asset-or-nothing ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Asset-or-nothing",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_asset_on"),
                        spot=common_spot_value,
                        misc={
                            "style": "asset_or_nothing",
                        },
                    )
                return

            if structure_name == "Forward-start option":
                t_start = st.number_input("T start (années)", value=float(common_maturity_value * 0.25), min_value=0.0, max_value=float(common_maturity_value * 0.9), step=0.05, key=kk("t_start"))
                k_fs = st.number_input("Facteur de strike (k)", value=1.0, min_value=0.1, step=0.05, key=kk("k"))
                n_paths_fs = st.number_input("Trajectoires MC", value=5000, min_value=500, step=500, key=kk("paths"))
                n_steps_fs = st.number_input("Pas de temps", value=200, min_value=20, step=10, key=kk("steps"))
                with st.expander(f"📈 Prix forward-start ({option_label})", expanded=False):
                    with st.spinner("Simulation Monte Carlo..."):
                        price = _forward_start_price_mc(
                            S0=common_spot_value,
                            r=common_rate_value,
                            q=float(d_common),
                            sigma=common_sigma_value,
                            T_start=t_start,
                            T_end=common_maturity_value,
                            k=k_fs,
                            n_paths=int(n_paths_fs),
                            n_steps=int(n_steps_fs),
                            option_type=option_char,
                        )
                    st.success(f"Prix forward-start ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Forward-start",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_forward_start"),
                        spot=common_spot_value,
                        misc={
                            "T_start": float(t_start),
                            "k_factor": float(k_fs),
                            "n_paths": int(n_paths_fs),
                            "n_steps": int(n_steps_fs),
                        },
                    )
                return

            if structure_name == "Chooser option":
                t_choice = st.number_input("Date de choix (années)", value=float(common_maturity_value * 0.5), min_value=0.0, max_value=float(max(0.01, common_maturity_value)), step=0.05, key=kk("t"))
                if st.button("Pricer le chooser", key=kk("btn")):
                    price = _chooser_option_price(
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        t_choice=t_choice,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                    )
                    st.success(f"Prix chooser = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Chooser",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_chooser"),
                        spot=common_spot_value,
                        misc={
                            "t_choice": float(t_choice),
                        },
                    )
                return

            if structure_name == "Straddle":
                if st.button("Pricer le straddle", key=kk("btn")):
                    price = _vanilla_price_with_dividend("call", common_spot_value, common_strike_value, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) + _vanilla_price_with_dividend("put", common_spot_value, common_strike_value, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix straddle (K={common_strike_value:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Straddle",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_straddle"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": common_strike_value},
                            {"option_type": "put", "strike": common_strike_value},
                        ],
                    )
                return

            if structure_name == "Strangle":
                wing = st.number_input("Écart strike strangle", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("wing"))
                if st.button("Pricer le strangle", key=kk("btn")):
                    k_put = max(0.01, common_strike_value - wing)
                    k_call = common_strike_value + wing
                    price = _vanilla_price_with_dividend("put", common_spot_value, k_put, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) + _vanilla_price_with_dividend("call", common_spot_value, k_call, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix strangle (Put {k_put:.2f} / Call {k_call:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Strangle",
                        option_char=option_char,
                        price_value=price,
                        strike=k_put,
                        strike2=k_call,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_strangle"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "put", "strike": k_put},
                            {"option_type": "call", "strike": k_call},
                        ],
                        misc={
                            "strike_put": k_put,
                            "strike_call": k_call,
                            "wing": wing,
                        },
                    )
                return

            if structure_name == "Call spread":
                width = st.number_input("Écart strikes (vertical call spread)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("width"))
                if st.button("Pricer le call spread", key=kk("btn")):
                    k_long = common_strike_value
                    k_short = common_strike_value + width
                    price = _vanilla_price_with_dividend("call", common_spot_value, k_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) - _vanilla_price_with_dividend("call", common_spot_value, k_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix call spread (long {k_long:.2f}, short {k_short:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Call spread",
                        option_char=option_char,
                        price_value=price,
                        strike=k_long,
                        strike2=k_short,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_call_spread"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": k_long, "side": "long"},
                            {"option_type": "call", "strike": k_short, "side": "short"},
                        ],
                        misc={"width": width},
                    )
                return

            if structure_name == "Put spread":
                width = st.number_input("Écart strikes (vertical put spread)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("width"))
                if st.button("Pricer le put spread", key=kk("btn")):
                    k_short = max(0.01, common_strike_value - width)
                    k_long = common_strike_value
                    price = _vanilla_price_with_dividend("put", common_spot_value, k_long, common_maturity_value, common_rate_value, float(d_common), common_sigma_value) - _vanilla_price_with_dividend("put", common_spot_value, k_short, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix put spread (long {k_long:.2f}, short {k_short:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Put spread",
                        option_char=option_char,
                        price_value=price,
                        strike=k_long,
                        strike2=k_short,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_put_spread"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "put", "strike": k_long, "side": "long"},
                            {"option_type": "put", "strike": k_short, "side": "short"},
                        ],
                        misc={"width": width},
                    )
                return

            if structure_name == "Butterfly":
                wing = st.number_input("Largeur des ailes (butterfly)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("wing"))
                if st.button("Pricer le butterfly", key=kk("btn")):
                    k1 = max(0.01, common_strike_value - wing)
                    k2 = common_strike_value
                    k3 = common_strike_value + wing
                    price = (
                        _vanilla_price_with_dividend("call", common_spot_value, k1, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - 2 * _vanilla_price_with_dividend("call", common_spot_value, k2, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        + _vanilla_price_with_dividend("call", common_spot_value, k3, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    )
                    st.success(f"Prix butterfly (K1={k1:.2f}, K2={k2:.2f}, K3={k3:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Butterfly",
                        option_char=option_char,
                        price_value=price,
                        strike=k1,
                        strike2=k3,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_bfly"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": k1, "side": "long"},
                            {"option_type": "call", "strike": k2, "side": "short", "qty": 2},
                            {"option_type": "call", "strike": k3, "side": "long"},
                        ],
                        misc={"wing": wing},
                    )
                return

            if structure_name == "Condor":
                wing_inner = st.number_input("Écart strikes intérieurs", value=max(1.0, common_strike_value * 0.03), min_value=0.01, step=0.1, key=kk("inner"))
                wing_outer = st.number_input("Largeur d'aile condor", value=max(1.0, common_strike_value * 0.06), min_value=0.01, step=0.1, key=kk("outer"))
                if st.button("Pricer le condor", key=kk("btn")):
                    K1 = max(0.01, common_strike_value - (wing_inner + wing_outer))
                    K2 = max(0.01, common_strike_value - wing_inner)
                    K3 = common_strike_value + wing_inner
                    K4 = common_strike_value + wing_inner + wing_outer
                    price = (
                        _vanilla_price_with_dividend("call", common_spot_value, K1, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("call", common_spot_value, K2, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("call", common_spot_value, K3, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        + _vanilla_price_with_dividend("call", common_spot_value, K4, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    )
                    st.success(f"Prix condor (K1={K1:.2f}, K2={K2:.2f}, K3={K3:.2f}, K4={K4:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Condor",
                        option_char=option_char,
                        price_value=price,
                        strike=K1,
                        strike2=K4,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_condor"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "call", "strike": K1, "side": "long"},
                            {"option_type": "call", "strike": K2, "side": "short"},
                            {"option_type": "call", "strike": K3, "side": "short"},
                            {"option_type": "call", "strike": K4, "side": "long"},
                        ],
                        misc={
                            "wing_inner": wing_inner,
                            "wing_outer": wing_outer,
                        },
                    )
                return

            if structure_name == "Iron Butterfly":
                wing = st.number_input("Largeur des ailes (iron fly)", value=max(1.0, common_strike_value * 0.05), min_value=0.01, step=0.1, key=kk("wing"))
                if st.button("Pricer l'iron butterfly", key=kk("btn")):
                    K_mid = common_strike_value
                    K_low = max(0.01, K_mid - wing)
                    K_high = K_mid + wing
                    price = (
                        _vanilla_price_with_dividend("put", common_spot_value, K_low, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("put", common_spot_value, K_mid, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        - _vanilla_price_with_dividend("call", common_spot_value, K_mid, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                        + _vanilla_price_with_dividend("call", common_spot_value, K_high, common_maturity_value, common_rate_value, float(d_common), common_sigma_value)
                    )
                    st.success(f"Prix iron butterfly (K={K_low:.2f}/{K_mid:.2f}/{K_high:.2f}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Iron Butterfly",
                        option_char=option_char,
                        price_value=price,
                        strike=K_low,
                        strike2=K_high,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_iron_bfly"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": "put", "strike": K_low, "side": "long"},
                            {"option_type": "put", "strike": K_mid, "side": "short"},
                            {"option_type": "call", "strike": K_mid, "side": "short"},
                            {"option_type": "call", "strike": K_high, "side": "long"},
                        ],
                        misc={
                            "wing": wing,
                            "k_mid": K_mid,
                        },
                    )
                return

            if structure_name == "Calendar spread":
                T_short = st.number_input("Maturité courte", value=float(max(0.1, common_maturity_value * 0.5)), min_value=0.01, key=kk("t_short"))
                T_long = st.number_input("Maturité longue", value=float(common_maturity_value), min_value=T_short + 0.01, key=kk("t_long"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                if st.button("Pricer le calendar", key=kk("btn")):
                    long_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, common_strike_value, T_long, common_rate_value, float(d_common), common_sigma_value)
                    short_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, common_strike_value, T_short, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix calendar ({opt_kind}) = {long_leg - short_leg:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Calendar spread",
                        option_char=option_char,
                        price_value=long_leg - short_leg,
                        strike=common_strike_value,
                        maturity=T_long,
                        key_prefix=kk("save_calendar"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": opt_kind, "strike": common_strike_value, "side": "long", "tenor": T_long},
                            {"option_type": opt_kind, "strike": common_strike_value, "side": "short", "tenor": T_short},
                        ],
                        misc={
                            "T_short": float(T_short),
                            "T_long": float(T_long),
                            "opt_kind": opt_kind,
                        },
                    )
                return

            if structure_name == "Diagonal spread":
                T_short = st.number_input("Maturité courte", value=float(max(0.1, common_maturity_value * 0.5)), min_value=0.01, key=kk("t_short"))
                T_long = st.number_input("Maturité longue", value=float(common_maturity_value), min_value=T_short + 0.01, key=kk("t_long"))
                k_short = st.number_input("Strike court", value=float(common_strike_value), min_value=0.01, key=kk("k_short"))
                k_long = st.number_input("Strike long", value=float(common_strike_value * 1.05), min_value=0.01, key=kk("k_long"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                if st.button("Pricer le diagonal", key=kk("btn")):
                    long_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, k_long, T_long, common_rate_value, float(d_common), common_sigma_value)
                    short_leg = _vanilla_price_with_dividend(opt_kind, common_spot_value, k_short, T_short, common_rate_value, float(d_common), common_sigma_value)
                    st.success(f"Prix diagonal ({opt_kind}) = {long_leg - short_leg:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Diagonal spread",
                        option_char=option_char,
                        price_value=long_leg - short_leg,
                        strike=k_short,
                        strike2=k_long,
                        maturity=T_long,
                        key_prefix=kk("save_diagonal"),
                        spot=common_spot_value,
                        legs=[
                            {"option_type": opt_kind, "strike": k_long, "side": "long", "tenor": T_long},
                            {"option_type": opt_kind, "strike": k_short, "side": "short", "tenor": T_short},
                        ],
                        misc={
                            "T_short": float(T_short),
                            "T_long": float(T_long),
                            "k_short": float(k_short),
                            "k_long": float(k_long),
                            "opt_kind": opt_kind,
                        },
                    )
                return

            if structure_name == "Binary barrier (digital)":
                barrier_type = st.selectbox("Barrière", ["up", "down"], key=kk("barrier_type"))
                direction = st.selectbox("Knock", ["out", "in"], key=kk("direction"))
                payout = st.number_input("Payout", value=1.0, min_value=0.0, step=0.1, key=kk("payout"))
                base_level = common_spot_value * (1.1 if barrier_type == "up" else 0.9)
                barrier_level = st.number_input("Niveau barrière", value=float(base_level), min_value=0.0001, key=kk("level"))
                n_paths_bb = st.number_input("Trajectoires MC", value=5000, min_value=500, step=500, key=kk("paths"))
                n_steps_bb = st.number_input("Pas de temps", value=200, min_value=20, step=10, key=kk("steps"))
                if st.button("Pricer la binary barrière", key=kk("btn")):
                    price = _binary_barrier_mc(
                        option_type=option_char,
                        barrier_type=barrier_type,
                        direction=direction,
                        S0=common_spot_value,
                        K=common_strike_value,
                        barrier=barrier_level,
                        T=common_maturity_value,
                        r=common_rate_value,
                        dividend=float(d_common),
                        sigma=common_sigma_value,
                        payout=payout,
                        n_paths=int(n_paths_bb),
                        n_steps=int(n_steps_bb),
                    )
                    st.success(f"Prix binary barrière = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label=f"Binary barrier {barrier_type}-{direction}",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_binary_barrier"),
                        spot=common_spot_value,
                        legs=[{"option_type": option_char, "strike": common_strike_value, "barrier": barrier_level}],
                        misc={
                            "barrier_type": barrier_type,
                            "direction": direction,
                            "barrier_level": barrier_level,
                            "payout": payout,
                            "n_paths": int(n_paths_bb),
                            "n_steps": int(n_steps_bb),
                        },
                    )
                return

            if structure_name == "Asian géométrique":
                n_obs_geo = st.number_input("Observations", value=12, min_value=1, step=1, key=kk("obs"))
                with st.expander(f"📈 Prix Asian géométrique ({option_label})", expanded=False):
                    with st.spinner("Calcul en cours..."):
                        price = asian_geometric_closed_form(
                            spot=common_spot_value,
                            strike=common_strike_value,
                            rate=common_rate_value,
                            sigma=common_sigma_value,
                            maturity=common_maturity_value,
                            n_obs=int(n_obs_geo),
                            option_type="call" if option_char == "c" else "put",
                        )
                    st.success(f"Prix asian géométrique ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Asian géométrique",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_asian_geo"),
                        spot=common_spot_value,
                        misc={
                            "method": "closed_form_geom",
                            "n_obs": int(n_obs_geo),
                            "sigma": float(common_sigma_value),
                            "r": float(common_rate_value),
                            "q": float(d_common),
                        },
                    )
                return

            if structure_name == "Lookback fixed (MC)":
                n_paths_lb = st.number_input("Trajectoires MC", value=5000, min_value=500, step=500, key=kk("paths"))
                n_steps_lb = st.number_input("Pas de temps", value=200, min_value=10, step=10, key=kk("steps"))
                with st.expander(f"📈 Prix Lookback fixed (MC) ({option_label})", expanded=False):
                    with st.spinner("Simulation Monte Carlo..."):
                        dt = common_maturity_value / n_steps_lb
                        drift = (common_rate_value - float(d_common) - 0.5 * common_sigma_value**2) * dt
                        diff = common_sigma_value * math.sqrt(dt)
                        disc = math.exp(-common_rate_value * common_maturity_value)
                        payoffs = []
                        for _ in range(int(n_paths_lb)):
                            s = common_spot_value
                            s_max = s_min = s
                            for _ in range(int(n_steps_lb)):
                                z = np.random.normal()
                                s *= math.exp(drift + diff * z)
                                s_max = max(s_max, s)
                                s_min = min(s_min, s)
                            if option_char == "c":
                                payoff = max(s_max - common_strike_value, 0.0)
                            else:
                                payoff = max(common_strike_value - s_min, 0.0)
                            payoffs.append(payoff)
                        price = disc * float(np.mean(payoffs)) if payoffs else 0.0
                    st.success(f"Prix lookback fixed ({option_label}) = {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Lookback fixed",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_lookback"),
                        spot=common_spot_value,
                        misc={
                            "n_paths": int(n_paths_lb),
                            "n_steps": int(n_steps_lb),
                        },
                    )
                return

            if structure_name == "Cliquet / Ratchet (MC)":
                n_periods = st.number_input("Nombre de périodes", value=12, min_value=1, step=1, key=kk("periods"))
                cap = st.number_input("Cap par période", value=0.05, min_value=-1.0, step=0.01, key=kk("cap"))
                floor = st.number_input("Floor par période", value=0.0, min_value=-1.0, step=0.01, key=kk("floor"))
                n_paths_cliq = st.number_input("Trajectoires MC", value=3000, min_value=500, step=500, key=kk("paths"))
                with st.expander("📈 Prix cliquet / ratchet (MC)", expanded=False):
                    with st.spinner("Simulation Monte Carlo..."):
                        price = _cliquet_mc(
                            S0=common_spot_value,
                            r=common_rate_value,
                            q=float(d_common),
                            sigma=common_sigma_value,
                            T=common_maturity_value,
                            n_periods=int(n_periods),
                            cap=float(cap),
                            floor=float(floor),
                            n_paths=int(n_paths_cliq),
                        )
                    st.success(f"Prix cliquet/ratchet ≈ {price:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Cliquet / Ratchet",
                        option_char=option_char,
                        price_value=price,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=kk("save_cliquet"),
                        spot=common_spot_value,
                        misc={
                            "n_periods": int(n_periods),
                            "cap": float(cap),
                            "floor": float(floor),
                            "n_paths": int(n_paths_cliq),
                        },
                    )
                return

            if structure_name == "Quanto option":
                sigma_fx = st.number_input("Vol FX", value=0.1, min_value=0.0, step=0.01, key=kk("sigma_fx"))
                rho_qt = st.number_input("Corrélation S/FX", value=0.0, min_value=-1.0, max_value=1.0, step=0.05, key=kk("rho"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                if st.button("Pricer la quanto", key=kk("btn")):
                    price = _quanto_vanilla_price(
                        option_type=opt_kind,
                        S0=common_spot_value,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r_dom=common_rate_value,
                        q_for=float(d_common),
                        sigma_asset=common_sigma_value,
                        sigma_fx=sigma_fx,
                        rho=rho_qt,
                    )
                    st.success(f"Prix quanto ({opt_kind}) = {price:.6f}")
                return

            if structure_name == "Rainbow option":
                S0_b = st.number_input("Spot actif B", value=float(common_spot_value), min_value=0.01, key=kk("S0b"))
                sigma_b = st.number_input("Vol B", value=float(common_sigma_value), min_value=0.0001, key=kk("sigb"))
                rho_ab = st.number_input("Corrélation A/B", value=0.2, min_value=-1.0, max_value=1.0, step=0.05, key=kk("rho"))
                payoff_on = st.selectbox("Sous-jacent du payoff", ["max", "min"], key=kk("payoff"))
                opt_kind = st.selectbox("Type", ["call", "put"], key=kk("type"))
                n_paths_r = st.number_input("Trajectoires MC", value=4000, min_value=500, step=500, key=kk("paths"))
                n_steps_r = st.number_input("Pas de temps", value=150, min_value=10, step=10, key=kk("steps"))
                if st.button("Pricer le rainbow", key=kk("btn")):
                    price = _rainbow_two_asset_mc(
                        payoff_on=payoff_on,
                        S0_a=common_spot_value,
                        S0_b=S0_b,
                        sigma_a=common_sigma_value,
                        sigma_b=sigma_b,
                        rho=rho_ab,
                        K=common_strike_value,
                        T=common_maturity_value,
                        r=common_rate_value,
                        q_a=float(d_common),
                        q_b=float(d_common),
                        n_paths=int(n_paths_r),
                        n_steps=int(n_steps_r),
                        option_type=opt_kind,
                    )
                    st.success(f"Prix rainbow ({payoff_on}) = {price:.6f}")
                return

        # Helper to render the relevant heatmap for the current Call/Put tab.
        def _render_heatmaps_for_current_option(label: str, call_matrix, put_matrix, x_vals, y_vals):
            if option_char == "c":
                st.write(f"Heatmap Call ({label})")
                _render_heatmap(call_matrix, x_vals, y_vals, f"Call ({label})")
            else:
                st.write(f"Heatmap Put ({label})")
                _render_heatmap(put_matrix, x_vals, y_vals, f"Put ({label})")
        # Heston Carr–Madan pricer helpers
        def _heston_params_from_state() -> HestonParams:
            return HestonParams(
                torch.tensor(float(st.session_state.get("heston_kappa_common", 2.0)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_theta_common", 0.04)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_eta_common", 0.5)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_rho_common", -0.7)), device=HES_DEVICE),
                torch.tensor(float(st.session_state.get("heston_v0_common", 0.04)), device=HES_DEVICE),
            )

        def _carr_madan_price(S0: float, K: float, T: float, r: float, q: float, opt_char: str, params: HestonParams) -> float:
            call_price = float(carr_madan_call_torch(S0, r, q, T, params, K))
            if opt_char == "c":
                return call_price
            # Put via parité call-put
            return float(call_price - S0 * math.exp(-q * T) + K * math.exp(-r * T))
        (
            tab_grp_vanilla,
            tab_grp_path,
            tab_grp_barrier,
            tab_grp_spreads,
            tab_grp_calendar,
            tab_grp_exotics,
            tab_grp_basket,
        ) = st.tabs(
            [
                "Vanilla / Early exercise",
                "Path-dependent",
                "Barrières",
                "Spreads & Wings",
                "Calendriers",
                "Exotiques",
                "Basket",
            ]
        )

        with tab_grp_vanilla:
            tab_european, tab_american, tab_bermudan = st.tabs(["Européenne", "Américaine", "Bermuda"])

        with tab_grp_path:
            (
                tab_asian,
                tab_asian_geo,
                tab_lookback,
                tab_lookback_fixed,
                tab_forward_start,
                tab_cliquet,
            ) = st.tabs(["Asian", "Asian géométrique", "Lookback", "Lookback fixed", "Forward-start", "Cliquet / Ratchet"])

        with tab_grp_barrier:
            tab_barrier, tab_binary_barrier = st.tabs(["Barrière", "Binary barrière"])

        with tab_grp_spreads:
            (
                tab_straddle,
                tab_strangle,
                tab_call_spread,
                tab_put_spread,
                tab_butterfly,
                tab_condor,
                tab_iron_condor,
                tab_iron_bfly,
            ) = st.tabs(["Straddle", "Strangle", "Call spread", "Put spread", "Butterfly", "Condor", "Iron Condor", "Iron Butterfly"])

        with tab_grp_calendar:
            tab_calendar, tab_diagonal = st.tabs(["Calendar spread", "Diagonal spread"])

        with tab_grp_exotics:
            tab_digital, tab_asset_on, tab_chooser, tab_quanto, tab_rainbow = st.tabs(["Digital", "Asset-or-nothing", "Chooser", "Quanto", "Rainbow"])

        with tab_european:
            st.header("Option européenne")
            _render_option_text("Option européenne", "european_graph")

            eu_run_flag = st.session_state.get(_k("run_eu_done"), False)
            if not eu_run_flag:
                if st.button("🚀 Lancer tous les pricings Euro", key=_k("run_all_euro"), type="primary"):
                    try:
                        # Pré-calcul BSM
                        opt_type_pre = "call" if option_char == "c" else "put"
                        price_bsm_pre = _vanilla_price_with_dividend(
                            option_type=opt_type_pre,
                            S0=common_spot_value,
                            K=common_strike_value,
                            T=common_maturity_value,
                            r=common_rate_value,
                            dividend=float(d_common),
                            sigma=common_sigma_value,
                        )
                        st.session_state["eu_price_bsm"] = price_bsm_pre
                        # Pré-calcul Heston CM
                        params_heston_pre = _heston_params_from_state()
                        price_cm_pre = _carr_madan_price(
                            S0=float(common_spot_value),
                            K=float(common_strike_value),
                            T=float(common_maturity_value),
                            r=float(common_rate_value),
                            q=float(d_common),
                            opt_char=option_char,
                            params=params_heston_pre,
                        )
                        st.session_state["eu_price_heston"] = price_cm_pre
                    except Exception:
                        pass
                    st.session_state[_k("run_eu_done")] = True
                    st.rerun()
                st.info("Clique sur le bouton pour afficher immédiatement les prix et les dropdowns d’ajout.")

            if eu_run_flag:
                params_heston = _heston_params_from_state()
                st.subheader("Heston (référence)")
                render_method_explainer(
                    "🧮 Méthode Heston pour les options européennes",
                    (
                        "- **Étape 1 – Choix du cadre probabiliste** : on modélise le sous‑jacent `S_t` et la variance instantanée `v_t` sous la mesure neutre au risque. `S_t` suit une diffusion où le terme de diffusion dépend de `√v_t`, et `v_t` suit un processus de type CIR avec rappel vers `θ`.\n"
                        "- **Étape 2 – Spécification des paramètres de Heston** : on travaille avec cinq paramètres structurants : `κ` (vitesse de rappel de la variance), `θ` (variance de long terme), `σ_v` (volatilité de la variance), `ρ` (corrélation entre chocs sur `S_t` et `v_t`) et `v0` (variance initiale).\n"
                        "- **Étape 3 – Préparation des données de marché** : les données CBOE sont téléchargées, nettoyées et ramenées sous forme de points `(S0, K, T, C_mkt)` ou `(P_mkt)`, en filtrant les maturités trop courtes et les prix non exploitables.\n"
                        "- **Étape 4 – Construction d’un pricer rapide** : pour un jeu de paramètres Heston donné, on évalue les prix de calls européens via la méthode de Carr–Madan (transformée de Fourier) implémentée en `carr_madan_call_torch`, ce qui permet d’avoir un pricer différentiable dans PyTorch.\n"
                        "- **Étape 5 – Définition de la fonction de perte** : on compare les prix modèle aux prix de marché sur l’ensemble des points, via une fonction de perte de type somme pondérée des carrés des écarts, éventuellement avec des poids pour privilégier certaines zones du smile.\n"
                        "- **Étape 6 – Optimisation / calibration** : à partir d’un vecteur de paramètres non contraints `u`, on reconstruit des paramètres Heston admissibles (positivité, contraintes de Feller) puis on minimise la perte par descente de gradient ou quasi‑Newton (itérations jusqu’à `max_iters` avec un pas `learning_rate`).\n"
                        "- **Étape 7 – Exploitation des paramètres calibrés** : une fois les paramètres calibrés obtenus, on peut pricer des options européennes, dériver des surfaces d’IV et comparer au BSM / MC.\n"
                    ),
                )
                render_inputs_explainer(
                    "🔧 Paramètres utilisés – Heston européen",
                    (
                        "- **\"S0 (spot)\"** : niveau actuel du sous‑jacent, utilisé pour centrer la grille de strikes.\n"
                        "- **\"K (strike)\"** : strike de référence saisi dans la barre latérale.\n"
                        "- **\"T (maturité, années)\"** : maturité commune pour les surfaces.\n"
                        "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : paramètres de taux.\n"
                        "- **\"Ticker (sous-jacent)\"** : code CBOE utilisé pour la collecte des options.\n"
                        "- **\"Largeur bande T (±)\" et \"Maturité T cible\" : bornes de calibration.\n"
                    ),
                )

                st.caption("Pricing direct avec Carr–Madan (Heston calibré).")
                with st.expander(f"📈 Prix Heston Carr–Madan ({option_label})", expanded=False):
                    price_cm = st.session_state.get("eu_price_heston")
                    if price_cm is None:
                        try:
                            with st.spinner("Calcul Heston Carr–Madan..."):
                                price_cm = _carr_madan_price(
                                    S0=float(common_spot_value),
                                    K=float(common_strike_value),
                                    T=float(common_maturity_value),
                                    r=float(common_rate_value),
                                    q=float(d_common),
                                    opt_char=option_char,
                                    params=params_heston,
                                )
                            st.session_state["eu_price_heston"] = price_cm
                        except Exception as exc:
                            st.error(f"Erreur Carr–Madan : {exc}")
                            price_cm = None
                    if price_cm is not None:
                        st.success(f"Prix Heston (Carr–Madan) {option_label} = {price_cm:.6f}")
                        misc_heston = {
                            "heston_params": {
                                "kappa": float(st.session_state.get("heston_kappa_common", 2.0)),
                                "theta": float(st.session_state.get("heston_theta_common", 0.04)),
                                "eta": float(st.session_state.get("heston_eta_common", 0.5)),
                                "rho": float(st.session_state.get("heston_rho_common", -0.7)),
                                "v0": float(st.session_state.get("heston_v0_common", 0.04)),
                            }
                        }
                        render_add_to_dashboard_button(
                            product_label="Vanilla (Heston CM)",
                            option_char=option_char,
                            price_value=price_cm,
                            strike=common_strike_value,
                            maturity=common_maturity_value,
                            key_prefix=_k("save_heston_cm"),
                            spot=common_spot_value,
                            misc=misc_heston,
                        )

                with st.expander("Visualisations Heston (Carr–Madan)", expanded=False):
                    try:
                        with st.spinner("Calcul heatmap & surface IV Heston…"):
                            k_vals = heatmap_strike_values
                            t_vals = heatmap_maturity_values

                            call_matrix = np.zeros((len(t_vals), len(k_vals)), dtype=float)
                            put_matrix = np.zeros_like(call_matrix)
                            for i_t, t_val in enumerate(t_vals):
                                for j_k, k_val in enumerate(k_vals):
                                    call_matrix[i_t, j_k] = _carr_madan_price(
                                        S0=float(common_spot_value),
                                        K=float(k_val),
                                        T=float(t_val),
                                        r=float(common_rate_value),
                                        q=float(d_common),
                                        opt_char="c",
                                        params=params_heston,
                                    )
                                    put_matrix[i_t, j_k] = _carr_madan_price(
                                        S0=float(common_spot_value),
                                        K=float(k_val),
                                        T=float(t_val),
                                        r=float(common_rate_value),
                                        q=float(d_common),
                                        opt_char="p",
                                        params=params_heston,
                                    )

                            # Surface IV sur la base du type d’option courant (call/put)
                            k_grid, t_grid = np.meshgrid(k_vals, t_vals)
                            price_grid = call_matrix if option_char == "c" else put_matrix
                            iv_grid = np.full_like(price_grid, np.nan, dtype=float)
                            for i_t, t_val in enumerate(t_vals):
                                for j_k, k_val in enumerate(k_vals):
                                    iv_grid[i_t, j_k] = implied_vol_option(
                                        price=float(price_grid[i_t, j_k]),
                                        S=float(common_spot_value),
                                        K=float(k_val),
                                        T=float(t_val),
                                        r=float(common_rate_value),
                                        option_type="call" if option_char == "c" else "put",
                                    )
                            # Disposition responsive : deux colonnes (Call/Put heatmap prix et surface IV) qui se superposent sur mobile.
                            col_heatmap, col_iv = st.columns(2)
                            with col_heatmap:
                                _render_heatmaps_for_current_option(
                                    "Heston Carr–Madan (K, T)",
                                    call_matrix,
                                    put_matrix,
                                    k_vals,
                                    t_vals,
                                )
                            with col_iv:
                                iv_fig = make_iv_surface_figure(k_grid, t_grid, iv_grid, title_suffix=" (Heston Carr–Madan)")
                                st.pyplot(iv_fig)
                    except Exception as exc:
                        st.error(f"Erreur calcul heatmap / surface IV Heston : {exc}")

                st.divider()
                st.subheader("Black–Scholes–Merton (prix ponctuel + heatmaps)")
                render_unlock_sidebar_button("eu_bsm", "🔓 Réactiver T (onglet BSM)")
                render_method_explainer(
                    "🧮 Méthode Black–Scholes–Merton (BSM)",
                    (
                        "- **Étape 1 – Mise sous la mesure neutre au risque** : on suppose GBM avec volatilité constante `σ` et drift `r-d`.\n"
                        "- **Étape 2 – Calcul des quantités intermédiaires** : `d1`, `d2` pour chaque `(S, K)`.\n"
                        "- **Étape 3 – Formule de prix** : call/put fermés.\n"
                        "- **Étape 4 – Construction des heatmaps** : matrices de prix call/put sur la grille Spot × Strike.\n"
                    ),
                )
                render_inputs_explainer(
                    "🔧 Paramètres utilisés – BSM",
                    (
                        "- **\"S0 (spot)\"** et **\"K (strike)\"** : centres de la grille.\n"
                        "- **\"T (maturité, années)\"**, **\"r\"**, **\"d\"**, **\"σ\"** : paramètres du modèle.\n"
                        "- **\"Span autour du spot (heatmaps)\"** : amplitude de la grille.\n"
                    ),
                )
                cpflag_eu_bsm = option_label
                st.caption("Type fixé par l’onglet Call / Put en haut de page.")
                with st.expander(f"📈 Prix BSM ({cpflag_eu_bsm})", expanded=False):
                    price_bsm = st.session_state.get("eu_price_bsm")
                    if price_bsm is None:
                        with st.spinner("Calcul BSM..."):
                            opt_type = "call" if option_char == "c" else "put"
                            price_bsm = _vanilla_price_with_dividend(
                                option_type=opt_type,
                                S0=common_spot_value,
                                K=common_strike_value,
                                T=common_maturity_value,
                                r=common_rate_value,
                                dividend=float(d_common),
                                sigma=common_sigma_value,
                            )
                        st.session_state["eu_price_bsm"] = price_bsm
                    st.success(f"Prix BSM ({cpflag_eu_bsm}) = {price_bsm:.6f}")
                    render_add_to_dashboard_button(
                        product_label="Vanilla (BSM)",
                        option_char=option_char,
                        price_value=price_bsm,
                        strike=common_strike_value,
                        maturity=common_maturity_value,
                        key_prefix=_k("save_bsm"),
                        spot=common_spot_value,
                    )
                st.caption(
                    f"Paramètres utilisés pour le prix unique BSM : "
                    f"S0={common_spot_value:.4f}, K={common_strike_value:.4f}, "
                    f"T={common_maturity_value:.4f}, r={common_rate_value:.4f}, "
                    f"d={float(d_common):.4f}, σ={common_sigma_value:.4f}"
                )
            payoff_desc_euro = (
                "Call : payoff = max(S - K, 0). Put : payoff = max(K - S, 0). "
                "La ligne rouge marque le strike K, la ligne pointillée montre un strike voisin pour situer la zone de gain."
            )
            _render_payoff_dropdown(
                f"Option européenne ({option_label})",
                payoff_desc_euro,
                lambda s, K, K2: max(s - K, 0.0) if option_char == "c" else max(K - s, 0.0),
            )

        with tab_american:
            st.header("Option américaine")
            cpflag_am = option_label
            cpflag_am_char = option_char
            _render_option_text("Option américaine", "american_payoff")

            with st.expander("📊 Graph payoff (américaine)", expanded=False):
                xs = np.linspace(max(0.1, K_common * 0.5), K_common * 1.5, 120)
                ys = [max(x - K_common, 0.0) if option_char == "c" else max(K_common - x, 0.0) for x in xs]
                fig_pay = _payoff_plot(xs, ys, f"Payoff américaine ({cpflag_am})", strike_lines=[K_common])
                st.plotly_chart(fig_pay, width="stretch")

            am_run_flag = st.session_state.get(_k("run_am_done"), False)
            if not am_run_flag:
                if st.button("🚀 Lancer tous les pricings Américain", key=_k("run_all_am"), type="primary"):
                    st.session_state[_k("run_am_done")] = True
                    st.rerun()
                st.info("Clique sur le bouton pour afficher immédiatement les prix et les dropdowns d’ajout.")

            if am_run_flag:
                st.subheader("Longstaff–Schwartz (Heston)")
                render_method_explainer(
                    "🧮 L-S Heston",
                    (
                        "- Simulation Monte Carlo avec variance stochastique calibrée Heston.\n"
                        "- Régression backward pour l’exercice optimal.\n"
                    ),
                )
                heston_ready = bool(st.session_state.get("heston_cboe_loaded_once", False))
                n_paths_am_hes = st.number_input(
                    "Trajectoires Monte Carlo (Heston)",
                    value=1000,
                    min_value=100,
                    key=_k("n_paths_am_hes"),
                    disabled=not heston_ready,
                )
                n_steps_am_hes = st.number_input(
                    "Pas de temps (Heston)",
                    value=50,
                    min_value=1,
                    key=_k("n_steps_am_hes"),
                    disabled=not heston_ready,
                )
                v0_am_hes = None
                process_am_hes = None
                if heston_ready:
                    kappa_am = float(st.session_state.get("heston_kappa_common", 2.0))
                    theta_am = float(st.session_state.get("heston_theta_common", 0.04))
                    eta_am = float(st.session_state.get("heston_eta_common", 0.5))
                    rho_am = float(st.session_state.get("heston_rho_common", -0.7))
                    v0_am_hes = float(st.session_state.get("heston_v0_common", 0.04))
                    process_am_hes = HestonProcess(
                        mu=r_common - d_common, kappa=kappa_am, theta=theta_am, eta=eta_am, rho=rho_am
                    )
                    st.caption(
                        f"Paramètres Heston : κ={kappa_am:.4f}, θ={theta_am:.4f}, η={eta_am:.4f}, ρ={rho_am:.4f}, v0={v0_am_hes:.4f}"
                    )
                else:
                    st.caption("Heston désactivé : fais la calibration Heston pour l’activer.")

                with st.expander(f"📈 Pricing Heston L-S ({cpflag_am})", expanded=False):
                    if not heston_ready:
                        st.info("Calibration Heston requise.")
                    else:
                        with st.spinner("Calcul Longstaff–Schwartz Heston..."):
                            try:
                                option_ls = Option(
                                    s0=S0_common,
                                    T=T_common,
                                    K=K_common,
                                    v0=v0_am_hes,
                                    call=(cpflag_am == "Call"),
                                )
                                price_ls = longstaff_schwartz_price(
                                    option=option_ls,
                                    process=process_am_hes,
                                    n_paths=int(n_paths_am_hes),
                                    n_steps=int(n_steps_am_hes),
                                )
                                st.success(f"Prix américain Heston L-S ({cpflag_am}) = {price_ls:.6f}")
                                render_add_to_dashboard_button(
                                    product_label="American (Heston L-S)",
                                    option_char=option_char,
                                    price_value=price_ls,
                                    strike=K_common,
                                    maturity=T_common,
                                    key_prefix=_k("save_am_heston_ls"),
                                    spot=S0_common,
                                )
                            except Exception as exc:
                                st.error(f"Erreur Longstaff–Schwartz Heston : {exc}")
                with st.expander("Heatmap Heston (L-S)", expanded=False):
                    if heston_ready:
                        with st.spinner("Calcul des heatmaps Heston L-S"):
                            call_heatmap_ls, put_heatmap_ls = _compute_american_ls_heatmaps(
                                heatmap_spot_values,
                                heatmap_strike_values,
                                T_common,
                                process_am_hes,
                                int(n_paths_am_hes),
                                int(n_steps_am_hes),
                                v0_am_hes,
                            )
                        _render_heatmaps_for_current_option(
                            "Heston L-S",
                            call_heatmap_ls,
                            put_heatmap_ls,
                            heatmap_spot_values,
                            heatmap_strike_values,
                        )
                    else:
                        st.info("Heatmap désactivée : calibration Heston requise.")

                st.divider()

                st.subheader("Longstaff–Schwartz (GBM)")
                render_method_explainer(
                    "🧮 L-S GBM",
                    (
                        "- Simulation GBM (volatilité constante) + régression backward.\n"
                    ),
                )
                n_paths_am_gbm = st.number_input(
                    "Trajectoires Monte Carlo (GBM)",
                    value=1000,
                    min_value=100,
                    key=_k("n_paths_am_gbm"),
                )
                n_steps_am_gbm = st.number_input(
                    "Pas de temps (GBM)",
                    value=50,
                    min_value=1,
                    key=_k("n_steps_am_gbm"),
                )
                process_am_gbm = GeometricBrownianMotion(mu=r_common - d_common, sigma=sigma_common)

                with st.expander(f"📈 Pricing GBM L-S ({cpflag_am})", expanded=False):
                    with st.spinner("Calcul Longstaff–Schwartz GBM..."):
                        try:
                            option_ls = Option(
                                s0=S0_common,
                                T=T_common,
                                K=K_common,
                                v0=None,
                                call=(cpflag_am == "Call"),
                            )
                            price_ls = longstaff_schwartz_price(
                                option=option_ls,
                                process=process_am_gbm,
                                n_paths=int(n_paths_am_gbm),
                                n_steps=int(n_steps_am_gbm),
                            )
                            st.success(f"Prix américain GBM L-S ({cpflag_am}) = {price_ls:.6f}")
                            render_add_to_dashboard_button(
                                product_label="American (GBM L-S)",
                                option_char=option_char,
                                price_value=price_ls,
                                strike=K_common,
                                maturity=T_common,
                                key_prefix=_k("save_am_gbm_ls"),
                                spot=S0_common,
                            )
                        except Exception as exc:
                            st.error(f"Erreur Longstaff–Schwartz GBM : {exc}")
                with st.expander("Heatmap GBM (L-S)", expanded=False):
                    with st.spinner("Calcul des heatmaps GBM L-S"):
                        call_heatmap_ls, put_heatmap_ls = _compute_american_ls_heatmaps(
                            heatmap_spot_values,
                            heatmap_strike_values,
                            T_common,
                            process_am_gbm,
                            int(n_paths_am_gbm),
                            int(n_steps_am_gbm),
                            None,
                        )
                    _render_heatmaps_for_current_option(
                        "GBM L-S",
                        call_heatmap_ls,
                        put_heatmap_ls,
                        heatmap_spot_values,
                        heatmap_strike_values,
                    )

                st.divider()

                st.subheader("Arbre binomial CRR")
                render_method_explainer(
                    "🌳 Arbre CRR",
                    (
                        "- Discrétisation de l’horizon en `n_tree_am` pas.\n"
                        "- Recursion backward avec exercice optimal.\n"
                    ),
                )
                n_tree_am = st.number_input(
                    "Nombre de pas de l'arbre",
                    value=10,
                    min_value=5,
                    key=_k("n_tree_am"),
                    help="Nombre de pas de temps utilisés dans l’arbre binomial CRR.",
                )
                option_am_crr = Option(s0=S0_common, T=T_common, K=K_common, call=cpflag_am == "Call")
                int_n_tree = int(n_tree_am)
                if int_n_tree > 10:
                    st.info("L'affichage peut devenir difficile à lire pour un nombre de pas supérieur à 10.")
                with st.expander(f"📈 Pricing CRR ({cpflag_am}) + heatmap", expanded=False):
                    with st.spinner("Calcul du prix CRR"):
                        try:
                            option_am_single = Option(
                                s0=S0_common,
                                T=T_common,
                                K=K_common,
                                call=(cpflag_am == 'Call'),
                            )
                            price_crr_single = crr_pricing(
                                r=r_common,
                                sigma=sigma_common,
                                option=option_am_single,
                                n=int_n_tree,
                            )
                            st.success(f"Prix américain CRR ({cpflag_am}) ≈ {price_crr_single:.6f} (avec {int_n_tree} pas)")
                            render_add_to_dashboard_button(
                                product_label="American (CRR)",
                                option_char=option_char,
                                price_value=price_crr_single,
                                strike=K_common,
                                maturity=T_common,
                                key_prefix=_k("save_am_crr"),
                                spot=S0_common,
                            )
                        except Exception as exc:
                            st.error(f"Erreur CRR : {exc}")
                    with st.spinner("Construction de l'arbre CRR"):
                        spot_tree, value_tree = _build_crr_tree(
                            option=option_am_crr, r=r_common, sigma=sigma_common, n_steps=int_n_tree
                        )
                    st.write("**Représentation graphique**")
                    fig_tree = _plot_crr_tree(spot_tree, value_tree)
                    st.pyplot(fig_tree)
                    plt.close(fig_tree)

                    with st.spinner("Calcul de la heatmap CRR"):
                        call_heatmap_crr, put_heatmap_crr = _compute_american_crr_heatmaps(
                            heatmap_spot_values,
                            heatmap_strike_values,
                            T_common,
                            r_common,
                            sigma_common,
                            int_n_tree,
                        )
                    _render_heatmaps_for_current_option(
                        "CRR",
                        call_heatmap_crr,
                        put_heatmap_crr,
                        heatmap_spot_values,
                        heatmap_strike_values,
                    )
                st.caption(
                    f"Paramètres utilisés pour CRR : "
                    f"S0={S0_common:.4f}, K={K_common:.4f}, T={T_common:.4f}, "
                    f"r={r_common:.4f}, σ={sigma_common:.4f}, n={int_n_tree}"
                )


        with tab_lookback:
            _render_option_text("Lookback floating (approx payoff)", "lookback_graph")
            st.header("Options lookback (floating strike)")
            render_unlock_sidebar_button("tab_lookback", "🔓 Réactiver T (onglet Lookback)")
            lb_run_flag = st.session_state.get(_k("run_path_lookback_done"), False)
            if not lb_run_flag:
                if st.button("🚀 Lancer tous les pricings Lookback", key=_k("run_path_lookback_btn"), type="primary"):
                    st.session_state[_k("run_path_lookback_done")] = True
                    lb_run_flag = True
                    st.rerun()
            if lb_run_flag:
                render_general_definition_explainer(
                    "🔍 Comprendre les options lookback",
                    (
                        "- **Payoff dépendant du chemin** : une option lookback ne dépend plus uniquement de `S_T`, mais de l'historique complet de la trajectoire du sous‑jacent (par exemple de son maximum ou de son minimum atteint avant l'échéance).\n"
                        "- **Floating strike** : dans cet onglet, on considère des structures où le strike effectif est défini à partir d'un extrême de la trajectoire, par exemple le maximum historique pour un put, ou le minimum pour un call.\n"
                        "- **Intérêt intuitif** : ce type d'option permet de \"regarder en arrière\" pour déterminer le niveau de référence du contrat, offrant une protection renforcée contre des mouvements extrêmes défavorables.\n"
                        "- **Dimension temporelle** : plus la maturité est longue, plus le sous‑jacent a de chances de visiter des extrêmes éloignés, ce qui impacte directement le niveau du payoff.\n"
                        "- **Objectif de cet onglet** : comparer une formule fermée (lorsqu'elle est disponible) à une approche Monte Carlo pour des options lookback, et visualiser l'effet des paramètres via des heatmaps Spot × Maturité."
                    ),
                )
                st.caption(
                    "Les heatmaps affichent les prix lookback sur un carré Spot × Maturité centré autour des valeurs définies dans la barre latérale."
                )

                st.subheader("Formule exacte")
                render_method_explainer(
                    "📗 Méthode analytique pour lookback",
                    (
                        "- **Étape 1 – Choix du modèle sous‑jacent** : on se place dans le cadre Black–Scholes standard avec volatilité constante `σ`, taux sans risque `r` et éventuellement dividende continu. Le sous‑jacent suit un mouvement brownien géométrique.\n"
                        "- **Étape 2 – Caractérisation des extrêmes** : on utilise des résultats de théorie des processus stochastiques sur la distribution du maximum (ou minimum) d’un mouvement brownien géométrique sur un horizon `[0, T]`.\n"
                        "- **Étape 3 – Réécriture du payoff** : le payoff lookback (par exemple basé sur `max_t S_t` ou `min_t S_t`) est réécrit de manière à isoler des termes qui ressemblent à des payoffs d’options européennes classiques, plus des termes correctifs dépendant des extrêmes.\n"
                        "- **Étape 4 – Intégration analytique** : à partir de cette réécriture, on calcule l’espérance neutre au risque de ce payoff en intégrant par rapport aux densités des extrêmes et du sous‑jacent. On obtient des formules fermées impliquant des fonctions de répartition de la loi normale et des combinaisons exponentielles.\n"
                        "- **Étape 5 – Implémentation numérique** : les formules fermées sont implémentées sous forme de fonctions vectorisées qui prennent en entrée `(S0, T, σ, r, …)` et renvoient directement le prix de l’option lookback pour chaque point de la grille Spot × Maturité.\n"
                        "- **Étape 6 – Construction de la heatmap** : pour chaque valeur de `S0` et `T` de la grille, la formule analytique est évaluée, ce qui remplit une matrice de prix. Cette matrice est ensuite affichée sous forme de carte de chaleur.\n"
                        "- **Étape 7 – Rôle de benchmark** : cette solution analytique sert de référence \"exacte\" pour valider la méthode Monte Carlo : en comparant les deux surfaces, on quantifie l’erreur de simulation et on ajuste le nombre d’itérations ou la granularité temporelle si nécessaire."
                    ),
                )
                render_inputs_explainer(
                    "🔧 Paramètres utilisés – Lookback exact",
                    (
                        "- **\"S0 (spot)\"** : fixe le centre de l’axe des spots de la heatmap sur lequel la formule exacte est évaluée.\n"
                        "- **\"T (maturité, années)\"** : fournit les maturités à partir desquelles on construit l’axe vertical de la heatmap.\n"
                        "- **\"t (temps courant)\"** : champ numérique permettant de considérer une option lookback déjà en cours de vie (temps écoulé depuis l’émission).\n"
                        "- **\"Taux sans risque r\"** : utilisé pour actualiser l’espérance du payoff dans la formule fermée.\n"
                        "- **\"Volatilité σ\"** : volatilité constante supposée par le modèle BSM sous‑jacent."
                    ),
                )
                t0_lb = st.number_input(
                    "t (temps courant)",
                    value=0.0,
                    min_value=0.0,
                    key=_k("t0_lb_exact"),
                    help="Temps déjà écoulé depuis l’émission de l’option lookback (en années).",
                )
                r_lb = max(r_common, 1e-6)
                with st.expander("📈 Prix lookback exact", expanded=False):
                    try:
                        with st.spinner("Calcul du prix exact..."):
                            lookback_opt = lookback_call_option(
                                T=float(T_common),
                                t=float(t0_lb),
                                S0=float(common_spot_value),
                                r=float(r_lb),
                                sigma=float(sigma_common),
                            )
                            price_lb_exact = float(lookback_opt.price_exact())
                        st.success(f"Prix lookback (formule exacte) = {price_lb_exact:.6f}")
                    except Exception as exc:
                        st.error(f"Erreur lookback (formule exacte) : {exc}")
                st.caption(
                    f"Paramètres utilisés pour le prix lookback exact : "
                    f"S0={common_spot_value:.4f}, T={T_common:.4f}, r={r_lb:.4f}, σ={sigma_common:.4f}, t={t0_lb:.4f}"
                )
                with st.spinner("Calcul de la heatmap exacte"):
                    heatmap_lb_exact = _compute_lookback_exact_heatmap(
                        heatmap_spot_values,
                        heatmap_maturity_values,
                        t0_lb,
                        r_lb,
                        sigma_common,
                    )
                st.write("Heatmap Lookback (formule exacte)")
                _render_heatmap(heatmap_lb_exact, heatmap_spot_values, heatmap_maturity_values, "Prix Lookback (Exact)")

                st.divider()

                st.subheader("Monte Carlo lookback")
                render_method_explainer(
                    "🎲 Méthode Monte Carlo pour lookback",
                    (
                        "- **Étape 1 – Grille temporelle** : on découpe l’horizon `[0, T]` en un certain nombre de pas de temps. Plus la grille est fine, mieux on détecte les extrêmes du sous‑jacent.\n"
                        "- **Étape 2 – Simulation des trajectoires** : on simule, sous la mesure neutre au risque, de nombreuses trajectoires `S_t` via un GBM avec volatilité constante `σ`, en appliquant à chaque pas un choc gaussien.\n"
                        "- **Étape 3 – Suivi de l’extrême** : pour chaque trajectoire, on met à jour à chaque pas le maximum (ou le minimum) atteint jusqu’alors. Cette valeur représente l’\"historique condensé\" de la trajectoire pour le payoff lookback.\n"
                        "- **Étape 4 – Évaluation du payoff** : à la date finale, on calcule le payoff en fonction de cet extrême (par exemple `max(M_T - K, 0)` où `M_T = max_{0≤t≤T} S_t`), ou les variantes floating strike selon le type de contrat.\n"
                        "- **Étape 5 – Actualisation** : on actualise le payoff obtenu sur chaque trajectoire au taux sans risque `r_common` jusqu’à la date présente.\n"
                        "- **Étape 6 – Moyenne Monte Carlo** : le prix est obtenu en moyennant ces payoffs actualisés sur l’ensemble des trajectoires simulées.\n"
                        "- **Étape 7 – Construction de la heatmap** : on répète l’algorithme pour toutes les combinaisons `(S0, T)` de la grille, de sorte à remplir une matrice de prix lookback Monte Carlo comparable à la surface analytique.\n"
                        "- **Étape 8 – Analyse d’erreur** : en comparant cette surface MC à la surface exacte, on évalue la qualité de la simulation (variabilité statistique, biais de discretisation des extrêmes) et on ajuste `n_iters_lb` ou la taille des pas de temps si nécessaire."
                    ),
                )
                render_inputs_explainer(
                    "🔧 Paramètres utilisés – Lookback Monte Carlo",
                    (
                        "- **\"S0 (spot)\"** : centre de l’axe des spots sur lequel les trajectoires lookback sont simulées.\n"
                        "- **\"T (maturité, années)\"** : ensemble des maturités pour lesquelles on simule les trajectoires et construit la heatmap.\n"
                        "- **\"t (temps courant) MC\"** : temps déjà écoulé avant le début de la période de simulation, pour traiter des options en cours de vie.\n"
                        "- **\"Taux sans risque r\"** : intervient dans le drift neutre au risque et l’actualisation des payoffs.\n"
                        "- **\"Volatilité σ\"** : volatilité supposée constante dans les trajectoires Monte Carlo.\n"
                        "- **\"Itérations Monte Carlo\"** : nombre de trajectoires simulées pour chaque couple `(S0, T)`."
                    ),
                )
                t0_lb_mc = st.number_input(
                    "t (temps courant) MC",
                    value=0.0,
                    min_value=0.0,
                    key=_k("t0_lb_mc"),
                    help="Temps déjà écoulé avant la période de simulation Monte Carlo (en années).",
                )
                n_iters_lb = st.number_input(
                    "Itérations Monte Carlo",
                    value=1000,
                    min_value=100,
                    key=_k("n_iters_lb_mc"),
                    help="Nombre de trajectoires lookback simulées pour chaque couple (S0, T).",
                )
                r_lb_mc = max(r_common, 1e-6)
                with st.expander("📈 Prix lookback Monte Carlo", expanded=False):
                    progress = st.progress(0)
                    try:
                        lookback_opt_mc = lookback_call_option(
                            T=float(T_common),
                            t=float(t0_lb_mc),
                            S0=float(common_spot_value),
                            r=float(r_lb_mc),
                            sigma=float(sigma_common),
                        )
                        progress.progress(40)
                        price_lb_mc = float(lookback_opt_mc.price_monte_carlo(int(n_iters_lb)))
                        progress.progress(80)
                        st.success(f"Prix lookback (Monte Carlo) = {price_lb_mc:.6f}")
                    except Exception as exc:
                        st.error(f"Erreur lookback Monte Carlo : {exc}")
                    finally:
                        progress.empty()
                st.caption(
                    f"Paramètres utilisés pour le prix lookback MC : "
                    f"S0={common_spot_value:.4f}, T={T_common:.4f}, r={r_lb_mc:.4f}, σ={sigma_common:.4f}, "
                    f"t={t0_lb_mc:.4f}, N_iters={int(n_iters_lb)}"
                )
                if st.checkbox("Afficher la heatmap Lookback (Monte Carlo)", value=False, key=_k("show_lb_mc_heatmap")):
                    progress = st.progress(0)
                    with st.spinner("Calcul de la heatmap Monte Carlo"):
                        heatmap_lb_mc = _compute_lookback_mc_heatmap(
                            heatmap_spot_values,
                            heatmap_maturity_values,
                            t0_lb_mc,
                            r_lb_mc,
                            sigma_common,
                            int(n_iters_lb),
                        )
                        progress.progress(100)
                    st.write("Heatmap Lookback (Monte Carlo)")
                    _render_heatmap(heatmap_lb_mc, heatmap_spot_values, heatmap_maturity_values, "Prix Lookback (MC)")
                    progress.empty()


        with tab_barrier:
            st.header("Options barrière")
            render_unlock_sidebar_button("tab_barrier", "🔓 Réactiver T (onglet Barrière)")
            _render_option_text("Barrière (MC)", "barrier_graph")
            barrier_run_flag = st.session_state.get(_k("run_barrier_done"), False)
            if not barrier_run_flag:
                if st.button("🚀 Lancer tous les pricings Barrière", key=_k("run_barrier_btn"), type="primary"):
                    st.session_state[_k("run_barrier_done")] = True
                    st.rerun()

            if barrier_run_flag:
                render_general_definition_explainer(
                    "🚧 Comprendre les options barrière",
                    (
                        "- **Principe de base** : une option barrière est activée ou désactivée en fonction du franchissement d'un niveau de prix prédéfini (`Hu` ou `Hd`). La trajectoire du sous‑jacent entre `0` et `T` devient donc déterminante.\n"
                        "- **Knock-out** : l'option cesse d'exister dès que la barrière est touchée ; le droit d'exercer à l'échéance est alors perdu.\n"
                        "- **Knock-in** : à l’inverse, l’option ne \"prend naissance\" que si la barrière a été franchie au moins une fois avant l’échéance.\n"
                        "- **Up / Down** : on distingue les barrières **Up** (situées au‑dessus du spot initial) des barrières **Down** (situées en dessous), ce qui permet de modéliser des scénarios de protection ou de conditionnalité différentes.\n"
                        "- **Sensibilité au chemin** : ces produits sont très sensibles au maillage temporel : plus les pas sont grossiers, plus on risque de manquer des franchissements de barrière entre deux dates de simulation.\n"
                        "- **Objectif de l'onglet** : montrer comment le prix réagit aux combinaisons `S0`, `K`, `T`, `Hu/Hd`, `σ` et au type de barrière (in/out, up/down) via des simulations Monte Carlo."
                    ),
                )
                (
                    tab_barrier_up_out,
                    tab_barrier_down_out,
                    tab_barrier_up_in,
                    tab_barrier_down_in,
                ) = st.tabs(["Up-and-out", "Down-and-out", "Up-and-in", "Down-and-in"])

                with tab_barrier_up_out:
                    st.subheader("Up-and-out")
                    with st.expander("🧭 Schéma payoff / barrière", expanded=False):
                        sample_spot = np.linspace(max(1.0, S0_common * 0.5), S0_common * 1.6, 80)
                        payoff_curve = [max(s - K_common, 0.0) for s in sample_spot]
                        Hu_up_plot = st.session_state.get(_k("Hu_up"), None)
                        fig_pay = go.Figure()
                        fig_pay.add_trace(go.Scatter(x=sample_spot, y=payoff_curve, name="Payoff call"))
                        if Hu_up_plot is not None:
                            fig_pay.add_shape(
                                type="line",
                                x0=Hu_up_plot,
                                x1=Hu_up_plot,
                                y0=0,
                                y1=max(payoff_curve) if payoff_curve else 1,
                                line=dict(color="red", dash="dash"),
                            )
                        fig_pay.update_layout(
                            title="Payoff call avec barrière up-and-out (ligne rouge = barrière Hu)",
                            xaxis_title="Spot",
                            yaxis_title="Payoff",
                        )
                        st.plotly_chart(fig_pay, width="stretch", key=_k("barrier_up_out_schema"))
                    render_method_explainer(
                        "⬆️ Méthode Monte Carlo – Up-and-out",
                        (
                            "- **Étape 1 – Définition du niveau de barrière** : on fixe une barrière haute `Hu` strictement au‑dessus du spot `S0_common`. Le contrat stipule qu’en cas de franchissement de `Hu` avant `T`, l’option est annulée.\n"
                            "- **Étape 2 – Simulation des trajectoires** : on simule des trajectoires `S_t` sous la mesure neutre au risque (GBM) en discrétisant `[0, T_common]` en `n_steps_up` pas de temps.\n"
                            "- **Étape 3 – Détection du knock‑out** : pour chaque trajectoire, on initialise un indicateur `knocked_out = False`. À chaque pas, si `S_t ≥ Hu_up`, on met `knocked_out = True` et on peut considérer que la trajectoire ne contribuera plus au payoff.\n"
                            "- **Étape 4 – Calcul du payoff terminal** : à la maturité, pour les trajectoires qui ne sont pas en knock‑out (`knocked_out = False`), on calcule le payoff européen standard `max(±(S_T-K_common), 0)`. Pour les trajectoires en knock‑out, le payoff est `0`.\n"
                            "- **Étape 5 – Actualisation et moyenne** : on actualise tous les payoffs par `exp(-r_common T_common)` puis on moyenne sur toutes les trajectoires.\n"
                            "- **Étape 6 – Construction de la heatmap barrière** : en répétant ces étapes pour différentes valeurs de `S0_common` ou `Hu`, on peut cartographier l’impact de la position de la barrière sur le prix, et visualiser le compromis entre protection et coût de la prime."
                        ),
                    )
                    render_inputs_explainer(
                        "🔧 Paramètres utilisés – Up-and-out",
                        (
                            "- **\"S0 (spot)\"** : niveau de départ du sous‑jacent pour toutes les trajectoires simulées.\n"
                            "- **\"K (strike)\"** : strike de l’option barrière (call ou put) utilisée pour le payoff si la barrière n’est jamais touchée.\n"
                            "- **\"T (maturité, années)\"** : durée de vie de l’option, donc horizon de simulation.\n"
                            "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : utilisés pour définir le drift neutre au risque et actualiser les payoffs.\n"
                            "- **\"Volatilité σ\"** : volatilité constante supposée dans les trajectoires Monte Carlo.\n"
                            "- **\"Call / Put\"** : choix du type d’option (call ou put) sur lequel la barrière s’applique.\n"
                            "- **\"Barrière haute Hu\"** : niveau de prix au‑dessus du spot à partir duquel le knock‑out se déclenche.\n"
                            "- **\"Trajectoires Monte Carlo\"** : nombre de chemins simulés pour estimer le prix.\n"
                            "- **\"Pas de temps MC\"** : nombre de pas de temps par trajectoire, qui conditionne la finesse de la détection de la barrière."
                        ),
                    )
                    cpflag_barrier_up = option_label
                    cpflag_barrier_up_char = option_char
                    st.caption("Type fixé par l’onglet Call / Put en haut de page.")
                    Hu_up = st.slider(
                        "Barrière haute Hu",
                        min_value=float(max(S0_common * 1.0, 0.01)),
                        max_value=float(max(S0_common * 3.0, S0_common + 1.0)),
                        value=float(max(110.0, S0_common * 1.1)),
                        step=float(max(S0_common * 0.01, 0.1)),
                        key=_k("Hu_up"),
                        help="Curseur pour fixer la barrière haute du scénario Up.",
                    )
                    n_paths_up = st.number_input(
                        "Trajectoires Monte Carlo",
                        value=1000,
                        min_value=500,
                        step=500,
                        key=_k("n_paths_barrier_up"),
                        help="Nombre de trajectoires simulées pour la barrière Up-and-out.",
                    )
                    n_steps_up = st.number_input(
                        "Pas de temps MC",
                        value=200,
                        min_value=10,
                        key=_k("n_steps_barrier_up"),
                        help="Nombre de pas de temps pour suivre le franchissement de la barrière.",
                    )

                    ctx_up_out = st.session_state.get(_k("saved_barrier_up_out"))

                    if st.button("Calculer (Up-and-out)", key=_k("btn_barrier_up")):
                        progress = st.progress(0)
                        with st.spinner("Simulation Monte Carlo en cours..."):
                            price = _barrier_monte_carlo_price(
                                option_type=cpflag_barrier_up_char,
                                barrier_type="up",
                                S0=S0_common,
                                K=K_common,
                                barrier=Hu_up,
                                T=T_common,
                                r=r_common,
                                dividend=d_common,
                                sigma=sigma_common,
                                n_paths=int(n_paths_up),
                                n_steps=int(n_steps_up),
                            )
                            progress.progress(100)
                        st.write(f"**Prix Monte Carlo barrière**: {price:.6f}")
                        progress.empty()
                        ctx_up_out = {
                            "price": price,
                            "misc": {
                                "barrier_type": "up",
                                "knock": "out",
                                "barrier_level": Hu_up,
                                "n_paths": int(n_paths_up),
                                "n_steps": int(n_steps_up),
                            },
                        }
                        st.session_state[_k("saved_barrier_up_out")] = ctx_up_out

                    if ctx_up_out:
                        render_add_to_dashboard_button(
                            product_label="Acheter barrière up-and-out",
                            option_char=option_char,
                            price_value=ctx_up_out["price"],
                            strike=K_common,
                            maturity=T_common,
                            key_prefix=_k("save_barrier_up_out"),
                            spot=S0_common,
                            misc=ctx_up_out["misc"],
                        )

                    st.caption(f"Rappel : S0 = {S0_common:.4f}, Hu = {Hu_up:.4f}")

                with tab_barrier_down_out:
                    st.subheader("Down-and-out")
                    render_method_explainer(
                        "⬇️ Méthode Monte Carlo – Down-and-out",
                        (
                            "- **Étape 1 – Positionnement de la barrière basse** : on choisit une barrière `Hd` située en dessous du spot `S0_common`. L’option disparaît si `S_t` tombe à ou sous ce niveau avant la maturité.\n"
                            "- **Étape 2 – Simulation des trajectoires** : on simule de nombreuses trajectoires `S_t` sous la mesure neutre au risque jusqu’à `T_common`, en `n_steps_down` pas de temps.\n"
                            "- **Étape 3 – Suivi du knock‑out** : pour chaque trajectoire, on surveille `S_t`. Dès que `S_t ≤ Hd_down`, on enregistre un état `knocked_out = True`.\n"
                            "- **Étape 4 – Payoff terminal** : à l’échéance, si `knocked_out = False`, on calcule le payoff européen standard (call ou put selon `cpflag_barrier_down`). Si `knocked_out = True`, le payoff est nul.\n"
                            "- **Étape 5 – Actualisation et moyennage** : on actualise les payoffs et on en prend la moyenne sur toutes les trajectoires pour obtenir le prix Monte Carlo.\n"
                            "- **Étape 6 – Étude de sensibilité** : la répétition de ce calcul pour différents `Hd` et `T` permet d’analyser la probabilité de survie de l’option et l’amplitude de la réduction de prime liée à la barrière."
                        ),
                    )
                    render_inputs_explainer(
                        "🔧 Paramètres utilisés – Down-and-out",
                        (
                            "- **\"S0 (spot)\"** : valeur initiale utilisée pour les trajectoires.\n"
                            "- **\"K (strike)\"** : strike de l’option à barrière.\n"
                            "- **\"T (maturité, années)\"** : horizon temporel de l’option.\n"
                            "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : interviennent dans le drift neutre au risque et l’actualisation des payoffs.\n"
                            "- **\"Volatilité σ\"** : volatilité constante supposée dans les simulations.\n"
                            "- **\"Call / Put\"** : sélection du type d’option (call ou put).\n"
                            "- **\"Barrière basse Hd\"** : niveau de prix en dessous du spot à partir duquel le knock‑out est activé.\n"
                            "- **\"Trajectoires Monte Carlo\"** : nombre de chemins simulés.\n"
                            "- **\"Pas de temps MC\"** : nombre de pas de simulation par trajectoire."
                        ),
                    )
                    cpflag_barrier_down = option_label
                    cpflag_barrier_down_char = option_char
                    st.caption("Type fixé par l’onglet Call / Put en haut de page.")
                    Hd_down = st.slider(
                        "Barrière basse Hd",
                        min_value=float(max(0.01, S0_common * 0.2)),
                        max_value=float(max(S0_common * 0.99, 0.1)),
                        value=float(max(1.0, S0_common * 0.8)),
                        step=float(max(S0_common * 0.01, 0.05)),
                        key=_k("Hd_down"),
                        help="Curseur pour régler la barrière basse du scénario Down.",
                    )
                    n_paths_down = st.number_input(
                        "Trajectoires Monte Carlo",
                        value=1000,
                        min_value=500,
                        step=500,
                        key=_k("n_paths_barrier_down"),
                        help="Nombre de trajectoires simulées pour la barrière Down-and-out.",
                    )
                    n_steps_down = st.number_input(
                        "Pas de temps MC",
                        value=200,
                        min_value=10,
                        key=_k("n_steps_barrier_down"),
                        help="Nombre de pas de temps pour suivre la barrière.",
                    )

                    ctx_down_out = st.session_state.get(_k("saved_barrier_down_out"))

                    if st.button("Calculer (Down-and-out)", key=_k("btn_barrier_down")):
                        progress = st.progress(0)
                        with st.spinner("Simulation Monte Carlo en cours..."):
                            price = _barrier_monte_carlo_price(
                                option_type=cpflag_barrier_down_char,
                                barrier_type="down",
                                S0=S0_common,
                                K=K_common,
                                barrier=Hd_down,
                                T=T_common,
                                r=r_common,
                                dividend=d_common,
                                sigma=sigma_common,
                                n_paths=int(n_paths_down),
                                n_steps=int(n_steps_down),
                            )
                            progress.progress(100)
                        st.write(f"**Prix Monte Carlo barrière**: {price:.6f}")
                        progress.empty()
                        ctx_down_out = {
                            "price": price,
                            "misc": {
                                "barrier_type": "down",
                                "knock": "out",
                                "barrier_level": Hd_down,
                                "n_paths": int(n_paths_down),
                                "n_steps": int(n_steps_down),
                            },
                        }
                        st.session_state[_k("saved_barrier_down_out")] = ctx_down_out
                    if ctx_down_out:
                        render_add_to_dashboard_button(
                            product_label="Acheter barrière down-and-out",
                            option_char=option_char,
                            price_value=ctx_down_out["price"],
                            strike=K_common,
                            maturity=T_common,
                            key_prefix=_k("save_barrier_down_out"),
                            spot=S0_common,
                            misc=ctx_down_out["misc"],
                        )

                    st.caption(f"Rappel : S0 = {S0_common:.4f}, Hd = {Hd_down:.4f}")

                with tab_barrier_up_in:
                    st.subheader("Up-and-in")
                    render_method_explainer(
                        "⬆️ Méthode Monte Carlo – Up-and-in",
                        (
                            "- **Étape 1 – Définition de la condition de knock‑in** : l’option n’a de valeur que si, à un moment entre `0` et `T_common`, le sous‑jacent a franchi la barrière haute `Hu`.\n"
                            "- **Étape 2 – Simulation des trajectoires** : on simule un grand nombre de trajectoires `S_t` sous la mesure neutre au risque, sur `n_steps_up_in` pas de temps.\n"
                            "- **Étape 3 – Suivi du knock‑in** : pour chaque trajectoire, on initialise un drapeau `knocked_in = False`. À chaque pas, si `S_t ≥ Hu_up_in`, on met `knocked_in = True`.\n"
                            "- **Étape 4 – Évaluation à maturité** : à `T_common`, si `knocked_in = True`, on calcule le payoff européen standard (call ou put). Si `knocked_in = False`, le payoff est nul, car la barrière n’a jamais été touchée.\n"
                            "- **Étape 5 – Actualisation et moyenne** : on actualise les payoffs et on en prend la moyenne pour obtenir le prix de l’option Up‑and‑in.\n"
                            "- **Étape 6 – Lien avec l’Up‑and‑out** : théoriquement, pour un même niveau de barrière, la somme des prix Up‑and‑in et Up‑and‑out (avec même type d’option) s’approche du prix de l’option vanilla, ce qui fournit un contrôle de cohérence."
                        ),
                    )
                    render_inputs_explainer(
                        "🔧 Paramètres utilisés – Up-and-in",
                        (
                            "- `S0_common` : spot initial.\n"
                            "- `K_common` : strike de l’option conditionnelle.\n"
                            "- `T_common` : maturité de l’option.\n"
                            "- `r_common` : taux sans risque.\n"
                            "- `d_common` : dividende continu.\n"
                            "- `sigma_common` : volatilité utilisée pour les simulations.\n"
                            "- `cpflag_barrier_up_in` : type d’option (call ou put) pour le scénario Up‑and‑in.\n"
                            "- `Hu_up_in` : niveau de barrière haute déclenchant le knock‑in.\n"
                            "- `n_paths_up_in` : nombre de trajectoires Monte Carlo.\n"
                            "- `n_steps_up_in` : nombre de pas de temps par trajectoire.\n"
                            "- `knock_in` : paramètre logique interne positionné à `True` pour spécifier la nature knock‑in du produit.\n"
                            "- Variables internes : drapeau de knock‑in par trajectoire, facteur d’actualisation, générateur pseudo‑aléatoire."
                        ),
                    )
                    cpflag_barrier_up_in = option_label
                    cpflag_barrier_up_in_char = option_char
                    st.caption("Type fixé par l’onglet Call / Put en haut de page.")
                    Hu_up_in = st.slider(
                        "Barrière haute Hu (Up-in)",
                        min_value=float(max(S0_common * 1.0, 0.01)),
                        max_value=float(max(S0_common * 3.0, S0_common + 1.0)),
                        value=float(max(110.0, S0_common * 1.1)),
                        step=float(max(S0_common * 0.01, 0.1)),
                        key=_k("Hu_up_in"),
                        help="Curseur pour positionner la barrière haute du scénario Up-in.",
                    )
                    n_paths_up_in = st.number_input(
                        "Trajectoires Monte Carlo (Up-in)",
                        value=1000,
                        min_value=500,
                        step=500,
                        key=_k("n_paths_barrier_up_in"),
                        help="Nombre de trajectoires simulées pour l’Up-and-in.",
                    )
                    n_steps_up_in = st.number_input(
                        "Pas de temps MC (Up-in)",
                        value=200,
                        min_value=10,
                        key=_k("n_steps_barrier_up_in"),
                        help="Nombre de pas de temps par trajectoire pour l’Up-in.",
                    )

                    ctx_up_in = st.session_state.get(_k("saved_barrier_up_in"))

                    if st.button("Calculer (Up-and-in)", key=_k("btn_barrier_up_in")):
                        progress = st.progress(0)
                        with st.spinner("Monte Carlo knock-in (Up)..."):
                            price = _barrier_monte_carlo_price(
                                option_type=cpflag_barrier_up_in_char,
                                barrier_type="up",
                                S0=S0_common,
                                K=K_common,
                                barrier=Hu_up_in,
                                T=T_common,
                                r=r_common,
                                dividend=d_common,
                                sigma=sigma_common,
                                n_paths=int(n_paths_up_in),
                                n_steps=int(n_steps_up_in),
                                knock_in=True,
                            )
                            progress.progress(100)
                        st.write(f"**Prix Monte Carlo knock-in**: {price:.6f}")
                        progress.empty()
                        ctx_up_in = {
                            "price": price,
                            "misc": {
                                "barrier_type": "up",
                                "knock": "in",
                                "barrier_level": Hu_up_in,
                                "n_paths": int(n_paths_up_in),
                                "n_steps": int(n_steps_up_in),
                            },
                        }
                        st.session_state[_k("saved_barrier_up_in")] = ctx_up_in
                    if ctx_up_in:
                        render_add_to_dashboard_button(
                            product_label="Acheter barrière up-and-in",
                            option_char=option_char,
                            price_value=ctx_up_in["price"],
                            strike=K_common,
                            maturity=T_common,
                            key_prefix=_k("save_barrier_up_in"),
                            spot=S0_common,
                            misc=ctx_up_in["misc"],
                        )

                    st.caption(f"Rappel : S0 = {S0_common:.4f}, Hu = {Hu_up_in:.4f}")

                with tab_barrier_down_in:
                    st.subheader("Down-and-in")
                    render_method_explainer(
                        "⬇️ Méthode Monte Carlo – Down-and-in",
                        (
                            "- **Étape 1 – Condition de knock‑in** : l’option ne vaut quelque chose que si la barrière basse `Hd` a été touchée ou cassée au moins une fois avant `T_common`.\n"
                            "- **Étape 2 – Simulation** : on simule des trajectoires du sous‑jacent et on surveille `S_t` à chaque pas.\n"
                            "- **Étape 3 – Suivi du drapeau** : pour chaque trajectoire, on initialise `knocked_in = False`. Dès qu’un `S_t ≤ Hd_down_in` est observé, on met `knocked_in = True`.\n"
                            "- **Étape 4 – Payoff terminal** : en fin de trajectoire, si `knocked_in = True`, on évalue le payoff européen (call ou put) ; sinon, le payoff est nul.\n"
                            "- **Étape 5 – Actualisation et agrégation** : les payoffs sont actualisés, puis moyennés sur toutes les trajectoires pour obtenir le prix.\n"
                            "- **Étape 6 – Sensibilité au niveau de barrière** : plus `Hd` est éloignée sous `S0_common`, moins la barrière a de chances d’être touchée et plus la prime du produit baisse, ce qui se visualise directement dans les résultats numériquement obtenus."
                        ),
                    )
                    render_inputs_explainer(
                        "🔧 Paramètres utilisés – Down-and-in",
                        (
                            "- **\"S0 (spot)\"** : spot de départ des trajectoires.\n"
                            "- **\"K (strike)\"** : strike de l’option Down‑and‑in.\n"
                            "- **\"T (maturité, années)\"** : horizon de l’option.\n"
                            "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : paramètres de taux utilisés dans la simulation et l’actualisation.\n"
                            "- **\"Volatilité σ\"** : volatilité utilisée pour la dynamique Monte Carlo.\n"
                            "- **\"Call / Put\"** : choix du type d’option.\n"
                            "- **\"Barrière basse Hd (Down-in)\"** : niveau de prix sous lequel la barrière est considérée comme touchée.\n"
                            "- **\"Trajectoires Monte Carlo (Down-in)\"** : nombre de trajectoires simulées.\n"
                            "- **\"Pas de temps MC (Down-in)\"** : nombre de pas de temps par trajectoire."
                        ),
                    )
                    cpflag_barrier_down_in = option_label
                    cpflag_barrier_down_in_char = option_char
                    st.caption("Type fixé par l’onglet Call / Put en haut de page.")
                    Hd_down_in = st.slider(
                        "Barrière basse Hd (Down-in)",
                        min_value=float(max(0.01, S0_common * 0.2)),
                        max_value=float(max(S0_common * 0.99, 0.1)),
                        value=float(max(1.0, S0_common * 0.8)),
                        step=float(max(S0_common * 0.01, 0.05)),
                        key=_k("Hd_down_in"),
                        help="Curseur pour fixer la barrière basse pour l’Up/Down-in.",
                    )
                    n_paths_down_in = st.number_input(
                        "Trajectoires Monte Carlo (Down-in)",
                        value=1000,
                        min_value=500,
                        step=500,
                        key=_k("n_paths_barrier_down_in"),
                    )
                    n_steps_down_in = st.number_input(
                        "Pas de temps MC (Down-in)",
                        value=200,
                        min_value=10,
                        key=_k("n_steps_barrier_down_in"),
                    )

                    ctx_down_in = st.session_state.get(_k("saved_barrier_down_in"))

                    if st.button("Calculer (Down-and-in)", key=_k("btn_barrier_down_in")):
                        progress = st.progress(0)
                        with st.spinner("Monte Carlo knock-in (Down)..."):
                            price = _barrier_monte_carlo_price(
                                option_type=cpflag_barrier_down_in_char,
                                barrier_type="down",
                                S0=S0_common,
                                K=K_common,
                                barrier=Hd_down_in,
                                T=T_common,
                                r=r_common,
                                dividend=d_common,
                                sigma=sigma_common,
                                n_paths=int(n_paths_down_in),
                                n_steps=int(n_steps_down_in),
                                knock_in=True,
                            )
                            progress.progress(100)
                        st.write(f"**Prix Monte Carlo knock-in**: {price:.6f}")
                        progress.empty()
                        ctx_down_in = {
                            "price": price,
                            "misc": {
                                "barrier_type": "down",
                                "knock": "in",
                                "barrier_level": Hd_down_in,
                                "n_paths": int(n_paths_down_in),
                                "n_steps": int(n_steps_down_in),
                            },
                        }
                        st.session_state[_k("saved_barrier_down_in")] = ctx_down_in
                    if ctx_down_in:
                        render_add_to_dashboard_button(
                            product_label="Acheter barrière down-and-in",
                            option_char=option_char,
                            price_value=ctx_down_in["price"],
                            strike=K_common,
                            maturity=T_common,
                            key_prefix=_k("save_barrier_down_in"),
                            spot=S0_common,
                            misc=ctx_down_in["misc"],
                        )


        with tab_bermudan:
            st.header("Option bermudéenne")
            _render_option_text(f"bermuda ({option_label})", "bermuda_payoff")
            cpflag_bmd = option_label
            cpflag_bmd_char = option_char
            st.caption("Type fixé par l’onglet Call / Put en haut de page.")

            bmd_run_flag = st.session_state.get(_k("run_bmd_done"), False)
            if not bmd_run_flag:
                if st.button("🚀 Lancer le pricing Bermuda", key=_k("run_all_bmd"), type="primary"):
                    st.session_state[_k("run_bmd_done")] = True
                    st.rerun()
                st.info("Clique sur le bouton pour afficher les dropdowns et lancer le pricing Bermuda.")

            if bmd_run_flag:
                with st.expander("📊 Graph payoff (Bermuda)", expanded=False):
                    xs = np.linspace(max(0.1, K_common * 0.5), K_common * 1.5, 120)
                    ys = [max(x - K_common, 0.0) if option_char == "c" else max(K_common - x, 0.0) for x in xs]
                    fig_pay = _payoff_plot(xs, ys, f"Payoff bermuda ({cpflag_bmd})", strike_lines=[K_common])
                    st.plotly_chart(fig_pay, width="stretch")

                n_ex_dates_bmd = st.slider(
                    "Nombre de dates d'exercice Bermude",
                    min_value=2,
                    max_value=30,
                    value=6,
                    step=1,
                    help="Les dates sont réparties uniformément sur la grille PDE (incluant l'échéance).",
                    key=_k("n_ex_dates_bmd"),
                )

                render_method_explainer(
                    "🧮 Méthode PDE Crank–Nicolson pour options bermudéennes",
                    (
                        "- **Étape 1 – Formulation PDE** : on écrit l’équation de Black–Scholes pour le prix `V(t, S)` en fonction du temps et du spot, en supposant volatilité constante `σ_common`, taux `r_common` et dividende `d_common`.\n"
                        "- **Étape 2 – Changement de variable en log‑prix** : pour des raisons numériques, on travaille en log‑spot `x = ln(S/S0)` et on construit une grille spatiale régulière en `x` centrée autour de `S0_common`.\n"
                        "- **Étape 3 – Discrétisation Crank–Nicolson** : la PDE est discrétisée dans le temps et l’espace en combinant une approche implicite et explicite (50 %–50 %). Cela conduit à des systèmes linéaires tridiagonaux à résoudre à chaque pas de temps.\n"
                        "- **Étape 4 – Condition terminale** : à la maturité `T_common`, on initialise `V(T, S)` au payoff européen standard (call ou put) pour toutes les valeurs de `S` sur la grille.\n"
                        "- **Étape 5 – Intégration temporelle backward** : on remonte le temps pas à pas en résolvant, à chaque pas, un système linéaire obtenu à partir des matrices `A` et `B` du schéma Crank–Nicolson. On applique en parallèle les conditions aux bornes (comportement pour `S → 0` et `S → +∞`).\n"
                        "- **Étape 6 – Traitement des dates Bermudes** : à chaque date d’exercice autorisée, on remplace la valeur obtenue par la PDE par `max(V(t, S), payoff(S))`, de façon à imposer la possibilité d’exercice anticipé discret.\n"
                        "- **Étape 7 – Lecture de la solution** : une fois revenue au temps initial, on lit la valeur de `V(0, S0_common)` sur la grille pour obtenir le prix. Les grecs `Delta`, `Gamma` et `Theta` sont ensuite calculés par différences finies à partir des valeurs de la grille dans un voisinage de `S0_common`."
                    ),
                )
                render_inputs_explainer(
                    "🔧 Paramètres utilisés – Bermuda (PDE)",
                    (
                        "- **\"S0 (spot)\"** : point de départ sur l’axe des prix pour lequel on lit le résultat de la PDE.\n"
                        "- **\"K (strike)\"** : strike de l’option bermudéenne.\n"
                        "- **\"T (maturité, années)\"** : échéance finale de l’option.\n"
                        "- **\"Volatilité σ\"** : volatilité constante utilisée dans l’équation de Black–Scholes.\n"
                        "- **\"Taux sans risque r\"** et **\"Dividende continu d\"** : paramètres de taux du sous‑jacent.\n"
                        "- **\"Call / Put (bermuda)\"** : choix du type d’option.\n"
                        "- **\"Nombre de dates d'exercice Bermude\"** : nombre de dates intermédiaires où l’exercice anticipé est autorisé (en plus de l’échéance)."
                    ),
                )

                with st.expander(
                    f"📈 Prix Bermuda (PDE) – S0={S0_common:.2f}, K={K_common:.2f}, T={T_common:.2f}",
                    expanded=False,
                ):
                    with st.spinner("Calcul PDE Bermuda (Crank–Nicolson)..."):
                        model_bmd = CrankNicolsonBS(
                            Typeflag="Bmd",
                            cpflag=cpflag_bmd_char,
                            S0=S0_common,
                            K=K_common,
                            T=T_common,
                            vol=sigma_common,
                            r=r_common,
                            d=d_common,
                            n_exercise_dates=int(n_ex_dates_bmd),
                        )
                        price_bmd, delta_bmd, gamma_bmd, theta_bmd = model_bmd.CN_option_info()
                    st.write(f"**Prix**: {price_bmd:.4f}")
                    st.write(f"**Delta**: {delta_bmd:.4f}")
                    st.write(f"**Gamma**: {gamma_bmd:.4f}")
                    st.write(f"**Theta**: {theta_bmd:.4f}")
                    render_add_to_dashboard_button(
                        product_label="Bermudan (PDE)",
                        option_char=option_char,
                        price_value=price_bmd,
                        strike=K_common,
                        maturity=T_common,
                        key_prefix=_k("save_bermudan_pde"),
                        spot=S0_common,
                        misc={"n_ex_dates": int(n_ex_dates_bmd)},
                    )
        with tab_grp_basket:
            st.header("Options basket")
            _render_option_text("Basket option", "basket_graph")
            basket_run_flag = st.session_state.get(_k("run_basket_done"), False)
            if not basket_run_flag:
                if st.button("🚀 Lancer le module Basket", key=_k("run_basket_btn"), type="primary"):
                    st.session_state[_k("run_basket_done")] = True
                    basket_run_flag = True
                    st.rerun()
            if basket_run_flag:
                render_general_definition_explainer(
                    "🧺 Comprendre les options basket",
                    (
                        "- **Définition** : une option basket porte sur un panier de plusieurs sous‑jacents (actions, indices, etc.), typiquement via une combinaison pondérée de leurs prix.\n"
                        "- **Mécanisme** : le payoff dépend de la valeur de ce panier (par exemple une moyenne pondérée des spots) à l’échéance ou selon une trajectoire donnée.\n"
                        "- **Intérêt** : ces produits permettent de mutualiser le risque entre plusieurs actifs et de construire des vues relatives (sur‑/sous‑performance de certains composants du panier).\n"
                        "- **Enjeux de modélisation** : la corrélation entre les sous‑jacents et la structure de la volatilité jouent un rôle central dans la forme de la distribution du panier.\n"
                        "- **Objectif de cet onglet** : explorer, à travers une surface de prix et éventuellement une calibration, l’impact des paramètres de marché et des pondérations sur le prix du basket."
                    ),
                )
                render_method_explainer(
                    "🧮 Méthode utilisée dans le module Basket",
                    (
                        "- **Étape 1 – Chargement des historiques** : on charge les séries de prix de clôture des actifs du panier (ticker par ticker) à partir de fichiers CSV, en s’assurant d’avoir une période historique commune.\n"
                        "- **Étape 2 – Construction du dataset** : à partir de ces séries, on construit un jeu de données où chaque ligne correspond à un scénario de marché (niveaux de prix, volatilités implicites, corrélations, strike, maturité, etc.) et à un prix d’option panier associé (label).\n"
                        "- **Étape 3 – Séparation train / test** : le dataset est découpé selon `split_ratio` en un ensemble d’entraînement et un ensemble de test, afin de pouvoir évaluer la capacité du modèle à généraliser.\n"
                        "- **Étape 4 – Entraînement du réseau de neurones** : un modèle `build_model_nn` est instancié avec une architecture adaptée (couches denses, activations non linéaires). On l’entraîne pendant `epochs` itérations pour minimiser une fonction de perte de type MSE entre prix prédits et prix \"théoriques\" (issus de BSM multi‑actifs ou Monte Carlo).\n"
                        "- **Étape 5 – Suivi de l’apprentissage** : pendant l’entraînement, on suit l’évolution de la perte sur le jeu d’entraînement et de validation (MSE train / val) pour détecter surapprentissage ou sous‑apprentissage.\n"
                        "- **Étape 6 – Construction des heatmaps de prix** : une fois le modèle entraîné, on le met en production sur une grille de paramètres (par exemple `S` et `K` autour de valeurs communes) pour produire une heatmap des prix d’option basket.\n"
                        "- **Étape 7 – Construction de la surface de volatilité implicite** : en inversant éventuellement les prix du modèle sur un ensemble de paramètres, on peut reconstruire une surface de volatilité implicite associée au panier et la comparer aux données de marché.\n"
                        "- **Étape 8 – Analyse des résultats** : les heatmaps et les courbes MSE permettent de juger de la qualité de l’approximation et de l’intérêt du modèle pour un pricing rapide en temps réel."
                    ),
                )
                render_inputs_explainer(
                    "🔧 Paramètres utilisés – Basket",
                    (
                        "- **\"S0 (spot)\"** : niveau de spot de référence utilisé pour centrer certaines grilles de prix du panier.\n"
                        "- **\"K (strike)\"** : strike de référence du basket, autour duquel on définit les domaines de strikes.\n"
                        "- **\"T (maturité, années)\"** : maturité de référence utilisée pour les surfaces de prix ou de volatilité.\n"
                        "- **\"Taux sans risque r\"** : taux utilisé pour actualiser les flux dans les modèles internes.\n"
                        "- **Sélection des actifs du panier** : zone de texte / boutons permettant de choisir les tickers qui composeront le basket.\n"
                        "- **\"Train ratio\"** : pourcentage du dataset historique utilisé pour l’apprentissage (le reste servant au test).\n"
                        "- **\"Epochs d'entraînement\"** : nombre de passes sur le dataset lors de l’entraînement du réseau de neurones."
                    ),
                )
                ui_basket_surface(
                    spot_common=common_spot_value,
                    maturity_common=common_maturity_value,
                    rate_common=common_rate_value,
                    strike_common=common_strike_value,
                    key_prefix=f"basket_{option_label.lower()}",
                )


        with tab_asian:
            _render_option_text("Asian (payoff terminal)", "asian_graph")
            run_flag = st.session_state.get(_k("run_path_asian_done"), False)
            if not run_flag:
                if st.button("🚀 Lancer tous les pricings Asian", key=_k("run_path_asian_btn"), type="primary"):
                    st.session_state[_k("run_path_asian_done")] = True
                    st.rerun()
            else:
                ui_asian_options(
                    spot_default=common_spot_value,
                    sigma_common=common_sigma_value,
                    maturity_common=common_maturity_value,
                    strike_common=common_strike_value,
                    rate_common=common_rate_value,
                    key_prefix=_k("asian"),
                    option_char=option_char,
                )

        with tab_iron_condor:
            _render_payoff_dropdown(
                "Iron Condor",
                "Crédit maximum au centre (entre strikes courts), perte plafonnée entre les ailes longues.",
                lambda s, K, K2: (
                    (lambda k1, k2, k3, k4: max(k1 - s, 0.0) - max(k2 - s, 0.0) - max(s - k3, 0.0) + max(s - k4, 0.0))(
                        K - max(abs(K2 - K), max(0.5, K * 0.05)) * 1.5,
                        K - max(abs(K2 - K), max(0.5, K * 0.05)) * 0.5,
                        K + max(abs(K2 - K), max(0.5, K * 0.05)) * 0.5,
                        K + max(abs(K2 - K), max(0.5, K * 0.05)) * 1.5,
                    )
                ),
                strike_lines_override=lambda K, K2: (
                    lambda k1, k2, k3, k4: [k1, k2, k3, k4]
                )(
                    K - max(abs(K2 - K), max(0.5, K * 0.05)) * 1.5,
                    K - max(abs(K2 - K), max(0.5, K * 0.05)) * 0.5,
                    K + max(abs(K2 - K), max(0.5, K * 0.05)) * 0.5,
                    K + max(abs(K2 - K), max(0.5, K * 0.05)) * 1.5,
                ),
            )
            _flag_ic = st.session_state.get(_k("run_iron_condor_done"), False)
            if not _flag_ic:
                if st.button("🚀 Lancer le pricing Iron Condor", key=_k("run_iron_condor_btn"), type="primary"):
                    st.session_state[_k("run_iron_condor_done")] = True
                    _flag_ic = True
            if _flag_ic:
                _render_structure_panel("Iron Condor")

        with tab_digital:
            _render_option_text("Digital (cash-or-nothing)", "digital_graph")
            _flag_dig = st.session_state.get(_k("run_digital_done"), False)
            if not _flag_dig:
                if st.button("🚀 Lancer le pricing Digital", key=_k("run_digital_btn"), type="primary"):
                    st.session_state[_k("run_digital_done")] = True
                    _flag_dig = True
            if _flag_dig:
                _render_structure_panel("Digital (cash-or-nothing)")

        with tab_asset_on:
            _render_option_text("Asset-or-nothing", "asset_on_graph")
            _flag_asset = st.session_state.get(_k("run_asset_on_done"), False)
            if not _flag_asset:
                if st.button("🚀 Lancer le pricing Asset-or-nothing", key=_k("run_asset_on_btn"), type="primary"):
                    st.session_state[_k("run_asset_on_done")] = True
                    _flag_asset = True
            if _flag_asset:
                _render_structure_panel("Asset-or-nothing")

        with tab_forward_start:
            _render_option_text("Forward-start (approx payoff)", "forward_start_graph")
            run_flag = st.session_state.get(_k("run_path_forward_done"), False)
            if not run_flag:
                if st.button("🚀 Lancer tous les pricings Forward-start", key=_k("run_path_forward_btn"), type="primary"):
                    st.session_state[_k("run_path_forward_done")] = True
                    st.rerun()
            else:
                _render_structure_panel("Forward-start option")

        with tab_chooser:
            _render_option_text("Chooser option", "chooser_graph")
            _flag_chooser = st.session_state.get(_k("run_chooser_done"), False)
            if not _flag_chooser:
                if st.button("🚀 Lancer le pricing Chooser", key=_k("run_chooser_btn"), type="primary"):
                    st.session_state[_k("run_chooser_done")] = True
                    _flag_chooser = True
            if _flag_chooser:
                _render_structure_panel("Chooser option")

        with tab_straddle:
            _render_payoff_dropdown(
                "Straddle",
                "Zone de gain : au-delà des strikes K±, la courbe de payoff devient positive (symétrique Call+Put).",
                lambda s, K, K2: max(s - K, 0.0) + max(K - s, 0.0),
                strike2_factor=1.0,
            )
            _flag = st.session_state.get(_k("run_straddle_done"), False)
            if not _flag:
                if st.button("🚀 Lancer le pricing Straddle", key=_k("run_straddle_btn"), type="primary"):
                    st.session_state[_k("run_straddle_done")] = True
                    _flag = True
            if _flag:
                _render_structure_panel("Straddle")

        with tab_strangle:
            _render_payoff_dropdown(
                "Strangle",
                "Zone de gain : en dehors des strikes éloignés (put bas, call haut), payoff devient positif.",
                lambda s, K, K2: max(K - s, 0.0) + max(s - K2, 0.0),
                strike_lines_override=lambda K, K2: [K, K2],
            )
            _flag = st.session_state.get(_k("run_strangle_done"), False)
            if not _flag:
                if st.button("🚀 Lancer le pricing Strangle", key=_k("run_strangle_btn"), type="primary"):
                    st.session_state[_k("run_strangle_done")] = True
                    _flag = True
            if _flag:
                _render_structure_panel("Strangle")

        with tab_call_spread:
            _render_payoff_dropdown(
                "Call spread",
                "Zone de gain : entre K (long call) et K2 (short call), payoff positif plafonné après K2.",
                lambda s, K, K2: max(s - K, 0.0) - max(s - K2, 0.0),
            )
            _flag = st.session_state.get(_k("run_call_spread_done"), False)
            if not _flag:
                if st.button("🚀 Lancer le pricing Call spread", key=_k("run_call_spread_btn"), type="primary"):
                    st.session_state[_k("run_call_spread_done")] = True
                    _flag = True
            if _flag:
                _render_structure_panel("Call spread")

        with tab_put_spread:
            _render_payoff_dropdown(
                "Put spread",
                "Zone de gain : entre Kbas (put short) et Khaut (put long). Payoff plafonné au-dessus de Khaut.",
                lambda s, K, K2: (
                    max(max(K, K2) - s, 0.0) - max(min(K, K2) - s, 0.0)
                ),
                strike2_factor=0.95,
            )
            _flag = st.session_state.get(_k("run_put_spread_done"), False)
            if not _flag:
                if st.button("🚀 Lancer le pricing Put spread", key=_k("run_put_spread_btn"), type="primary"):
                    st.session_state[_k("run_put_spread_done")] = True
                    _flag = True
            if _flag:
                _render_structure_panel("Put spread")

        with tab_butterfly:
            _render_payoff_dropdown(
                "Butterfly",
                "Zone de gain : centrée autour de K2 (strikes courts), payoff en forme de tente, perte en dehors.",
                lambda s, K, K2: max(s - K, 0.0) - 2 * max(s - K2, 0.0) + max(s - (K2 + (K2 - K)), 0.0),
            )
            _flag = st.session_state.get(_k("run_butterfly_done"), False)
            if not _flag:
                if st.button("🚀 Lancer le pricing Butterfly", key=_k("run_butterfly_btn"), type="primary"):
                    st.session_state[_k("run_butterfly_done")] = True
                    _flag = True
            if _flag:
                _render_structure_panel("Butterfly")

        with tab_condor:
            _render_payoff_dropdown(
                "Condor",
                "Zone de gain : plateau central entre strikes courts, pertes en dehors des ailes.",
                lambda s, K, K2: (
                    max(s - (K * 0.97), 0.0)
                    - max(s - K, 0.0)
                    - max(s - K2, 0.0)
                    + max(s - (K2 + (K - K * 0.97)), 0.0)
                ),
                strike_lines_override=lambda K, K2: [K, K2],
            )
            _flag = st.session_state.get(_k("run_condor_done"), False)
            if not _flag:
                if st.button("🚀 Lancer le pricing Condor", key=_k("run_condor_btn"), type="primary"):
                    st.session_state[_k("run_condor_done")] = True
                    _flag = True
            if _flag:
                _render_structure_panel("Condor")

        with tab_iron_bfly:
            _render_payoff_dropdown(
                "Iron Butterfly",
                "Straddle vendu au centre, ailes longues protègent en dehors. Payoff en tente inversée (crédit maximal autour de K).",
                lambda s, K, K2: (
                    (lambda w: max((K - w) - s, 0.0) - max(K - s, 0.0) - max(s - K, 0.0) + max(s - (K + w), 0.0))(
                        max(abs(K2 - K), max(0.5, K * 0.05))
                    )
                ),
                strike2_factor=1.0,
                strike_lines_override=lambda K, K2: [
                    K - max(abs(K2 - K), max(0.5, K * 0.05)),
                    K,
                    K + max(abs(K2 - K), max(0.5, K * 0.05)),
                ],
            )
            _flag_ib = st.session_state.get(_k("run_iron_bfly_done"), False)
            if not _flag_ib:
                if st.button("🚀 Lancer le pricing Iron Butterfly", key=_k("run_iron_bfly_btn"), type="primary"):
                    st.session_state[_k("run_iron_bfly_done")] = True
                    _flag_ib = True
            if _flag_ib:
                _render_structure_panel("Iron Butterfly")

        with tab_calendar:
            _render_option_text("Calendar spread", "calendar_graph")
            _flag_cal = st.session_state.get(_k("run_calendar_done"), False)
            if not _flag_cal:
                if st.button("🚀 Lancer le pricing Calendar spread", key=_k("run_calendar_btn"), type="primary"):
                    st.session_state[_k("run_calendar_done")] = True
                    _flag_cal = True
            if _flag_cal:
                _render_structure_panel("Calendar spread")

        with tab_diagonal:
            _render_option_text("Diagonal spread", "diagonal_graph")
            _flag_diag = st.session_state.get(_k("run_diagonal_done"), False)
            if not _flag_diag:
                if st.button("🚀 Lancer le pricing Diagonal spread", key=_k("run_diagonal_btn"), type="primary"):
                    st.session_state[_k("run_diagonal_done")] = True
                    _flag_diag = True
            if _flag_diag:
                _render_structure_panel("Diagonal spread")

        with tab_binary_barrier:
            _render_option_text("Binary barrière (digital)", "binary_barrier_graph")
            barrier_run_flag_local = st.session_state.get(_k("run_binary_barrier_done"), False)
            if not barrier_run_flag_local:
                if st.button("🚀 Lancer le pricing Binary barrière", key=_k("run_binary_barrier_btn"), type="primary"):
                    st.session_state[_k("run_binary_barrier_done")] = True
                    barrier_run_flag_local = True
            if barrier_run_flag_local:
                _render_structure_panel("Binary barrier (digital)")

        with tab_asian_geo:
            _render_option_text("Asian géométrique (payoff terminal)", "asian_geo_graph")
            run_flag = st.session_state.get(_k("run_path_asian_geo_done"), False)
            if not run_flag:
                if st.button("🚀 Lancer tous les pricings Asian géométrique", key=_k("run_path_asian_geo_btn"), type="primary"):
                    st.session_state[_k("run_path_asian_geo_done")] = True
                    st.rerun()
            else:
                _render_structure_panel("Asian géométrique")

        with tab_lookback_fixed:
            _render_option_text("Lookback fixed (approx payoff)", "lookback_fixed_graph")
            run_flag = st.session_state.get(_k("run_path_lookback_fixed_done"), False)
            if not run_flag:
                if st.button("🚀 Lancer tous les pricings Lookback fixed", key=_k("run_path_lookback_fixed_btn"), type="primary"):
                    st.session_state[_k("run_path_lookback_fixed_done")] = True
                    st.rerun()
            else:
                _render_structure_panel("Lookback fixed (MC)")

        with tab_cliquet:
            _render_option_text("Cliquet / Ratchet (approx payoff)", "cliquet_graph")
            run_flag = st.session_state.get(_k("run_path_cliquet_done"), False)
            if not run_flag:
                if st.button("🚀 Lancer tous les pricings Cliquet / Ratchet", key=_k("run_path_cliquet_btn"), type="primary"):
                    st.session_state[_k("run_path_cliquet_done")] = True
                    st.rerun()
            else:
                _render_structure_panel("Cliquet / Ratchet (MC)")

        with tab_quanto:
            _render_option_text("Quanto option", "quanto_graph")
            _flag_quanto = st.session_state.get(_k("run_quanto_done"), False)
            if not _flag_quanto:
                if st.button("🚀 Lancer le pricing Quanto", key=_k("run_quanto_btn"), type="primary"):
                    st.session_state[_k("run_quanto_done")] = True
                    _flag_quanto = True
            if _flag_quanto:
                _render_structure_panel("Quanto option")

        with tab_rainbow:
            _render_option_text("Rainbow option", "rainbow_graph")
            _flag_rainbow = st.session_state.get(_k("run_rainbow_done"), False)
            if not _flag_rainbow:
                if st.button("🚀 Lancer le pricing Rainbow", key=_k("run_rainbow_btn"), type="primary"):
                    st.session_state[_k("run_rainbow_done")] = True
                    _flag_rainbow = True
            if _flag_rainbow:
                _render_structure_panel("Rainbow option")

    tab_call, tab_put = st.tabs(["Call", "Put"])
    for _label, _tab in (("Call", tab_call), ("Put", tab_put)):
        _char = "c" if _label == "Call" else "p"
        with _tab:
            render_option_tabs_for_type(_label, _char)

# Alpaca API Setup
key = "PKRQ4GPVDAPCYIH6QGR4HI5USK"
secret_key = "3mENa9jXaLhESSekQzvz4cRh758awvBppB7Dfs9o1LJw"
BASE_URL = "https://paper-api.alpaca.markets/"
api = tradeapi.REST(key, secret_key, BASE_URL, api_version="v2")

# Page config
st.set_page_config(page_title="AI Trading Bot", page_icon="📈", layout="wide", initial_sidebar_state="collapsed")

# Helper functions
@st.cache_data(ttl=10)
def fetch_portfolio():
    """
    Fetch live equity positions from Alpaca and project them into a UI-friendly list.

    Returns:
        list[dict]: Each entry contains Symbol, Quantity, prices, P/L, and side.
    Side effects:
        - Network call to Alpaca REST API.
        - Displays Streamlit error on failure.
    """
    try:
        positions = api.list_positions()
        portfolio = []
        for pos in positions:
            portfolio.append({
                'Symbol': pos.symbol,
                'Quantity': pos.qty,
                'Entry Price': f"${float(pos.avg_entry_price):.2f}",
                'Current Price': f"${float(pos.current_price):.2f}",
                'Unrealized P/L': f"${float(pos.unrealized_pl):.2f}",
                'Side': 'buy'
            })
        return portfolio
    except Exception as e:
        st.error(f"Error fetching portfolio: {e}")
        return []

@st.cache_data(ttl=10)
def fetch_open_orders():
    """
    Fetch open orders from Alpaca for display.

    Returns:
        list[dict]: Orders with symbol, qty, limit price (or Market), and side.
    Side effects:
        - Network call to Alpaca REST API.
        - Displays Streamlit error on failure.
    """
    try:
        orders = api.list_orders(status='open')
        open_orders = []
        for order in orders:
            open_orders.append({
                'Symbol': order.symbol,
                'Quantity': order.qty,
                'Limit Price': f"${float(order.limit_price):.2f}" if order.limit_price else "Market",
                'Side': order.side
            })
        return open_orders
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return []

def get_data(symbol):
    """
    Retrieve a spot price for a ticker using Alpaca first, then yfinance as fallback.

    Args:
        symbol (str): Ticker symbol to quote.
    Returns:
        dict: {"price": float} with -1 if no source returned a valid price.
    Side effects:
        - Network calls to Alpaca and yfinance.
    """
    symbol = (symbol or "").strip().upper()
    # 1) Alpaca live trade
    try:
        barset = api.get_latest_trade(symbol)
        return {"price": float(barset.price)}
    except Exception:
        pass
    # 2) yfinance fallback (last close)
    try:
        hist = yf.Ticker(symbol).history(period="5d", interval="1d")
        if not hist.empty and "Close" in hist.columns:
            return {"price": float(hist["Close"].iloc[-1])}
    except Exception:
        pass
    return {"price": -1}

def load_equities():
    """Load configured trading systems (equities) from disk; returns {} on failure."""
    try:
        with open(DATA_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_equities(equities):
    """Persist trading systems (equities) to disk."""
    with open(DATA_FILE, 'w') as f:
        json.dump(equities, f, indent=2)

def load_portfolio():
    """Load the spot portfolio snapshot from disk; returns {} on failure."""
    try:
        with open(PORTFOLIO_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def save_portfolio(portfolio):
    """Persist the spot portfolio snapshot to disk."""
    with open(PORTFOLIO_FILE, 'w') as f:
        json.dump(portfolio, f, indent=2)


def load_sell_systems():
    """Load automated sell systems configuration from disk; returns {} on failure."""
    try:
        with open(SELL_SYSTEMS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_sell_systems(sell_systems):
    """Persist automated sell systems configuration to disk."""
    with open(SELL_SYSTEMS_FILE, 'w') as f:
        json.dump(sell_systems, f, indent=2)


def load_options_book():
    """
    Load the unified options book from disk, preferring the current file and
    falling back to the legacy file if needed.

    Returns:
        dict: Map of option_id -> option entry; {} if nothing is found/parseable.
    """
    for path in (OPTIONS_BOOK_FILE, OPTIONS_BOOK_FILE_LEGACY):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return {}


def load_options_book_legacy_only() -> dict:
    """Direct read of the legacy options_book.json (used for display/debug)."""
    try:
        with open(OPTIONS_BOOK_FILE_LEGACY, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_options_book(book):
    """Persist the unified options book (and legacy mirror) to disk."""
    try:
        OPTIONS_BOOK_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(OPTIONS_BOOK_FILE, 'w') as f:
            json.dump(book, f, indent=2)
        if OPTIONS_BOOK_FILE_LEGACY != OPTIONS_BOOK_FILE:
            try:
                with open(OPTIONS_BOOK_FILE_LEGACY, 'w') as f_legacy:
                    json.dump(book, f_legacy, indent=2)
            except Exception:
                # Legacy mirror is best-effort; ignore failures there.
                pass
    except Exception as exc:
        # Surface file-system issues (permissions, etc.) in the UI.
        try:
            st.error(f"Erreur lors de l'écriture du fichier options : {exc}")
        except Exception:
            pass
        raise


def _split_options_book(book: dict):
    """
    Partition the unified book into open vs expired slices.

    Args:
        book (dict): Full options book keyed by option id.
    Returns:
        tuple(dict, dict): (active, expired) maps.
    """
    active = {k: v for k, v in book.items() if v.get("status", "open") == "open"}
    expired = {k: v for k, v in book.items() if v.get("status") == "expired"}
    return active, expired


def load_options_portfolio():
    """Return only active option positions from the unified book."""
    active, _ = _split_options_book(load_options_book())
    return active


def save_options_portfolio(options_portfolio):
    """
    Persist active options merged with any expired entries back to disk.

    Args:
        options_portfolio (dict): Active/open options keyed by id.
    Returns:
        dict: Full merged book (active + expired) written to disk.
    """
    book = load_options_book()
    _, expired = _split_options_book(book)
    merged = dict(expired)
    for option_id, entry in options_portfolio.items():
        entry_copy = dict(entry)
        entry_copy.setdefault("status", "open")
        merged[option_id] = entry_copy
    save_options_book(merged)
    return merged


def load_expired_options():
    """Return expired/closed options from the unified book (migration safe)."""
    migrate_legacy_expired_options()
    _, expired = _split_options_book(load_options_book())
    return expired


def save_expired_options(expired_options):
    """
    Persist expired options merged with active ones, marking each entry as expired.

    Args:
        expired_options (dict): Expired option entries keyed by id.
    Returns:
        dict: Full merged book (active + expired) written to disk.
    """
    active = load_options_portfolio()
    book = dict(active)
    for option_id, entry in expired_options.items():
        entry_copy = dict(entry)
        entry_copy["status"] = "expired"
        book[option_id] = entry_copy
    save_options_book(book)
    return book


def migrate_legacy_expired_options():
    """Move legacy expired_options.json entries into the unified book schema."""
    if not LEGACY_EXPIRED_FILE.exists():
        return
    try:
        with open(LEGACY_EXPIRED_FILE, "r") as f:
            legacy = json.load(f)
    except Exception:
        return
    if not isinstance(legacy, dict) or not legacy:
        try:
            LEGACY_EXPIRED_FILE.unlink(missing_ok=True)
        except Exception:
            pass
        return
    book = load_options_book()
    changed = False
    for option_id, entry in legacy.items():
        entry_copy = dict(entry)
        entry_copy["status"] = "expired"
        entry_copy.setdefault("option_type", entry_copy.get("type", ""))
        entry_copy.setdefault("product", entry_copy.get("product_type", entry_copy.get("structure", "vanilla")))
        entry_copy.setdefault("side", entry_copy.get("side", "long"))
        entry_copy.setdefault("quantity", entry_copy.get("quantity", entry_copy.get("qty", 0)))
        entry_copy.setdefault("avg_price", entry_copy.get("avg_price", entry_copy.get("T_0_price", 0.0)))
        book[option_id] = entry_copy
        changed = True
    if changed:
        save_options_book(book)
    try:
        LEGACY_EXPIRED_FILE.unlink(missing_ok=True)
    except Exception:
        pass


def _compute_leg_payoff(leg: dict, spot: float) -> float:
    """
    Compute payoff contribution for a single leg (call/put, long/short).

    Args:
        leg: Leg definition containing option_type, strike, qty, side.
        spot: Underlying level at evaluation.
    Returns:
        Signed payoff scaled by quantity.
    """
    option_type = (leg.get("option_type") or leg.get("type") or "call").lower()
    strike = float(leg.get("strike", 0.0) or 0.0)
    qty = float(leg.get("qty", 1.0) or 1.0)
    payoff = max(spot - strike, 0.0) if option_type == "call" else max(strike - spot, 0.0)
    if (leg.get("side") or "long").lower() == "short":
        payoff *= -1.0
    return payoff * qty


def compute_option_payoff(option: dict, spot: float) -> float:
    """
    Evaluate payoff for a stored option/structure at a given spot.

    Supports multi-leg structures, Asian, digital, barrier, straddle/strangle,
    vanilla spreads, and vanilla fallback.

    Args:
        option: Stored option dict (dashboard schema).
        spot: Underlying level to evaluate payoff.
    Returns:
        float: Payoff (positive for long exposure, negative if encoded in legs).
    """
    legs = option.get("legs") or []
    if legs:
        return sum(_compute_leg_payoff(leg, spot) for leg in legs)

    product = (option.get("product_type") or option.get("product") or option.get("structure") or option.get("type") or "").lower()
    option_type_raw = (
        option.get("option_type")
        or option.get("cpflag")
        or option.get("cp_flag")
        or option.get("cp")
        or ""
    )
    if not option_type_raw:
        t_val = str(option.get("type", "")).lower()
        if t_val in {"call", "put"}:
            option_type_raw = t_val
    option_type = str(option_type_raw).lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", 0.0) or 0.0) or 0.0)
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    closing_prices = misc.get("closing_prices") if isinstance(misc, dict) else None

    # Path-dependent: Asian arithmetic / geometric via provided path
    if "asian" in product and closing_prices:
        if "geom" in product:
            vals = [v for v in closing_prices if v and v > 0]
            if not vals:
                avg_level = spot
            else:
                avg_level = float(np.exp(np.mean(np.log(vals))))
        else:
            avg_level = float(np.mean(closing_prices))
        if option_type == "put":
            return max(strike - avg_level, 0.0)
        return max(avg_level - strike, 0.0)

    # Digital (cash-or-nothing) payoff
    if "digital" in product:
        payout = float(misc.get("payout", 1.0) or 1.0)
        if option_type == "put":
            return payout if spot < strike else 0.0
        return payout if spot > strike else 0.0

    # Barrier (simple in/out check using closing_prices if provided)
    if "barrier" in product:
        barrier = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
        barrier_type = str(misc.get("barrier_type", "up")).lower()
        knock = str(misc.get("knock", misc.get("direction", "out"))).lower()
        path_vals = closing_prices or [spot]
        hit = False
        for p in path_vals:
            if barrier_type == "up" and p >= barrier:
                hit = True
                break
            if barrier_type == "down" and p <= barrier:
                hit = True
                break
        if knock == "out" and hit:
            return 0.0
        if knock == "in" and not hit:
            return 0.0
        # otherwise vanilla payoff

    if product in {"straddle"}:
        return max(spot - strike, 0.0) + max(strike - spot, 0.0)
    if product in {"strangle"}:
        k_put = min(strike, strike2 or strike)
        k_call = max(strike, strike2 or strike)
        return max(k_put - spot, 0.0) + max(spot - k_call, 0.0)
    if product in {"call_spread", "bull_call_spread", "callspread"}:
        k1 = strike
        k2 = strike2 or strike
        return max(spot - k1, 0.0) - max(spot - k2, 0.0)
    if product in {"put_spread", "bear_put_spread", "putspread"}:
        k_long = strike
        k_short = strike2 or strike
        return max(k_long - spot, 0.0) - max(k_short - spot, 0.0)

    # Vanilla fallback
    if option_type == "put" or product == "put":
        return max(strike - spot, 0.0)
    return max(spot - strike, 0.0)


def compute_option_pnl(option: dict, spot_at_event: float, mark_price: float | None = None) -> dict:
    """
    Compute PnL metadata for an option position given an event spot or mark price.

    Args:
        option: Stored option dict with avg_price/side/quantity.
        spot_at_event: Underlying level at expiration/event.
        mark_price: Optional mark-to-market override.
    Returns:
        dict: payoff_per_unit, pnl_per_unit, pnl_total.
    """
    premium = float(option.get("avg_price", 0.0) or 0.0)
    quantity = float(option.get("quantity", 1) or 0.0)
    side = (option.get("side") or "long").lower()
    payoff_value = mark_price if mark_price is not None else compute_option_payoff(option, spot_at_event)

    if side == "long":
        pnl_per_unit = payoff_value - premium
    else:
        pnl_per_unit = premium - payoff_value

    return {
        "payoff_per_unit": payoff_value,
        "pnl_per_unit": pnl_per_unit,
        "pnl_total": pnl_per_unit * quantity,
    }


def describe_expired_option_payoff(option: dict, spot_at_event: float | None = None) -> dict:
    """Short metadata for the payoff computation of an expired option."""
    product_raw = (
        option.get("product_type")
        or option.get("product")
        or option.get("structure")
        or option.get("type")
        or ""
    )
    product = str(product_raw).lower()
    option_type_raw = (
        option.get("option_type")
        or option.get("cpflag")
        or option.get("cp")
        or ""
    )
    option_type = str(option_type_raw or "call").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    strike2 = float(option.get("strike2", option.get("strike_upper", 0.0) or 0.0) or 0.0)
    quantity = float(option.get("quantity", 0.0) or 0.0)
    side = (option.get("side") or "long").lower()
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    closing_prices = misc.get("closing_prices") if isinstance(misc, dict) else None
    barrier_level = float(misc.get("barrier", misc.get("barrier_level", 0.0)) or 0.0)
    barrier_type = str(misc.get("barrier_type", "up")).lower() if isinstance(misc, dict) else ""
    knock = str(misc.get("knock", misc.get("direction", "out"))).lower() if isinstance(misc, dict) else ""
    payout = float(misc.get("payout", 1.0)) if isinstance(misc, dict) else 1.0
    legs = option.get("legs") or []

    spot_close = float(
        spot_at_event
        if spot_at_event is not None
        else option.get("underlying_close", option.get("S_T", 0.0))
        or 0.0
    )

    method_label = "Vanilla"
    description = "Payoff européen: call=max(S-K,0) ; put=max(K-S,0)."
    params: dict = {"strike": strike, "option_type": option_type, "underlying_close": spot_close}

    prod_low = product.lower()
    if legs:
        method_label = "Structure multi-jambes"
        description = "Somme des payoffs jambe par jambe avec signe long/short et quantités propres à chaque jambe."
        params["legs"] = legs
    elif "asian" in prod_low:
        if closing_prices:
            params["closing_prices"] = closing_prices
        method_label = "Asiatique"
        if "geom" in prod_low:
            description = "Moyenne géométrique des prix observés, puis payoff vanilla sur cette moyenne."
            params["average"] = "geometric"
        else:
            description = "Moyenne arithmétique des prix observés, puis payoff vanilla sur cette moyenne."
            params["average"] = "arithmetic"
        params["strike"] = strike
    elif "digital" in prod_low:
        method_label = "Digital (cash-or-nothing)"
        description = "Paiement fixe si la condition sur S_T est vraie, 0 sinon."
        params.update({"strike": strike, "payout": payout, "option_type": option_type})
    elif "barrier" in prod_low:
        method_label = "Barrière"
        description = "Vérifie le franchissement de la barrière (knock in/out) avant d'appliquer le payoff vanilla."
        params.update(
            {
                "strike": strike,
                "barrier": barrier_level,
                "barrier_type": barrier_type,
                "direction": knock,
                "closing_prices": closing_prices or [],
            }
        )
    elif "straddle" in prod_low:
        method_label = "Straddle"
        description = "Somme call+put au même strike: max(S-K,0)+max(K-S,0)."
        params["strike"] = strike
    elif "strangle" in prod_low:
        method_label = "Strangle"
        description = "Put strike bas + Call strike haut: max(K_put-S,0)+max(S-K_call,0)."
        params.update({"strike_put": strike, "strike_call": strike2})
    elif any(k in prod_low for k in ["spread", "butterfly", "condor", "iron butterfly", "iron condor", "diagonal", "calendar"]):
        method_label = "Combinaison de spreads"
        description = "Combinaison linéaire de payoffs vanilla (long/short) sur différents strikes/échéances."
        params.update({"strike": strike, "strike2": strike2, "legs": legs or []})

    payoff_unit = option.get("payoff_per_unit")
    if payoff_unit is None:
        payoff_unit = compute_option_payoff(option, spot_close)
    side_mult = -1.0 if side == "short" else 1.0
    payoff_signed_unit = payoff_unit * side_mult
    payoff_total = payoff_signed_unit * quantity

    return {
        "function": "compute_option_payoff",
        "method_label": method_label,
        "description": description,
        "params": params,
        "underlying_close": spot_close,
        "quantity": quantity,
        "side": side,
        "payoff_unit": payoff_unit,
        "payoff_signed_unit": payoff_signed_unit,
        "payoff_total": payoff_total,
    }


def mark_option_market_value(option: dict, chain_entry: dict | None = None) -> tuple[float, float | None, float, float, str]:
    """
    Derive a model mark for an option position.

    Resolution order:
    - If chain_entry provided (CBOE row), use its spot/IV/T.
    - If misc contains Heston params, price via Carr-Madan; else BSM; else intrinsic payoff.

    Args:
        option: Stored option dict (dashboard schema).
        chain_entry: Optional CBOE chain row with spot/iv/T hints.
    Returns:
        tuple: (spot, T_years, sigma_used, mark_price, method_label).
    """
    underlying = option.get("underlying")
    spot_data = get_data(underlying) if underlying else {"price": 0}
    spot = float(spot_data.get("price", 0.0) or 0.0)
    # Fallback to provided S0 if live spot not available
    if spot <= 0:
        spot = float(option.get("S0", 0.0) or 0.0)

    sigma = float(option.get("sigma", 0.2) or 0.2)
    option_type = (option.get("option_type") or option.get("type") or "").lower()
    strike = float(option.get("strike", 0.0) or 0.0)
    q_val = 0.0

    T_years = None
    try:
        expiry_date = datetime.date.fromisoformat(option.get("expiration"))
        days = (expiry_date - datetime.date.today()).days
        T_years = max(days, 0) / 365.0
    except Exception:
        T_years = None

    try:
        q_val = float(get_q(underlying) or 0.0) if underlying else 0.0
    except Exception:
        q_val = 0.0

    if chain_entry:
        spot = float(chain_entry.get("spot", spot) or spot)
        sigma = float(chain_entry.get("iv", sigma) or sigma)
        T_years = float(chain_entry.get("T", T_years) or (T_years or 0.0))

    mark_price = None
    method = "intrinsic"
    # Heston Carr–Madan if params are provided in misc
    misc = option.get("misc") if isinstance(option.get("misc"), dict) else {}
    heston_misc = misc.get("heston_params") if isinstance(misc, dict) else None
    # fallback maturity from stored field if expiration missing
    if T_years is None:
        try:
            T_years = float(option.get("maturity_years") or 0.0)
        except Exception:
            T_years = None
    if (
        heston_misc
        and spot > 0
        and strike > 0
        and T_years is not None
        and T_years > 0
        and option_type in {"call", "put"}
    ):
        try:
            params = HestonParams(
                torch.tensor(float(heston_misc.get("kappa")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("theta")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("eta")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("rho")), device=HES_DEVICE),
                torch.tensor(float(heston_misc.get("v0")), device=HES_DEVICE),
            )
            r_val = float(get_r(T_years) or 0.0)
            mark_price = _carr_madan_price(
                S0=spot,
                K=strike,
                T=T_years,
                r=r_val,
                q=q_val,
                opt_char="c" if option_type == "call" else "p",
                params=params,
            )
            method = "Heston CM (stored params)"
        except Exception:
            mark_price = None

    # BSM fallback
    if mark_price is None and spot > 0 and strike > 0 and T_years is not None and option_type in {"call", "put"}:
        try:
            r_val = float(get_r(T_years) or 0.0)
        except Exception:
            r_val = 0.0
        mark_price = black_scholes_price(
            S=spot,
            K=strike,
            T=T_years,
            r=r_val,
            sigma=max(sigma, 1e-6),
            option_type=option_type,
            q=q_val,
        )
        method = "BSM (CBOE IV)" if chain_entry else "BSM"

    if mark_price is None and spot > 0:
        mark_price = compute_option_payoff(option, spot)
        method = "payoff"

    if mark_price is None:
        mark_price = 0.0

    return spot, T_years, sigma, float(mark_price), method


def add_option_to_dashboard(record: dict) -> str:
    """
    Normalize and persist an option/structure into the unified options book.

    Args:
        record: Raw option payload (pricing UI or chain selection).
    Returns:
        str: Assigned option id.
    Side effects:
        - Writes to options_portfolio.json (and legacy mirror).
    """
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    book = load_options_book()
    option_id = record.get("id") or record.get("contract_symbol") or record.get("label") or f"opt_{int(time.time() * 1000)}"
    entry = dict(record)
    entry["id"] = option_id
    entry.setdefault("status", "open")
    entry.setdefault("created_at", now)
    entry["last_updated"] = now

    def _f(val):
        return float(val) if val not in (None, "") else None

    # Canonical fields for storage (uniform schema)
    entry["underlying"] = entry.get("underlying") or entry.get("ticker") or entry.get("symbol") or ""
    product_val = entry.get("product") or entry.get("product_type") or entry.get("structure") or "vanilla"
    entry["product"] = product_val
    entry["product_type"] = product_val
    entry["type"] = entry.get("type") or "vanilla"
    entry["option_type"] = (
        entry.get("option_type")
        or entry.get("cpflag")
        or entry.get("cp")
        or ("call" if str(product_val).lower().startswith("call") else "put")
    )
    entry["side"] = entry.get("side") or "long"
    entry["strike"] = _f(entry.get("strike"))
    entry["strike2"] = _f(entry.get("strike2"))
    entry["expiration"] = entry.get("expiration") or None
    entry["quantity"] = _f(entry.get("quantity"))
    entry["S0"] = _f(entry.get("S0"))
    entry["legs"] = entry.get("legs")

    t0_price = entry.get("T_0_price")
    if t0_price is None:
        t0_price = entry.get("avg_price", entry.get("price"))
    entry["T_0_price"] = _f(t0_price)
    entry["avg_price"] = _f(entry["T_0_price"])

    misc_val = entry.get("misc")
    if not isinstance(misc_val, dict):
        misc_val = None
    # Normalization for misc keys
    if isinstance(misc_val, dict) and "product" in entry and "strangle" in str(entry["product"]).lower():
        # rename legacy keys
        if "k_call" in misc_val:
            misc_val["strike_call"] = misc_val.pop("k_call")
        if "k_put" in misc_val:
            misc_val["strike_put"] = misc_val.pop("k_put")
    if isinstance(misc_val, dict) and "product" in entry and "barrier" in str(entry["product"]).lower():
        # ensure knock/direction are both populated for barrier structures
        if "knock" not in misc_val and "direction" in misc_val:
            misc_val["knock"] = misc_val["direction"]
        if "direction" not in misc_val and "knock" in misc_val:
            misc_val["direction"] = misc_val["knock"]
    entry["misc"] = misc_val

    book[option_id] = entry
    save_options_book(book)
    return option_id


def load_custom_options():
    """Load custom options book (pricing-saved) from disk; returns {} on failure."""
    try:
        with open(CUSTOM_OPTIONS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_custom_options(custom_options):
    """Persist custom options to disk."""
    with open(CUSTOM_OPTIONS_FILE, 'w') as f:
        json.dump(custom_options, f, indent=2)


def load_forwards():
    """Load forward positions from disk; returns {} on failure."""
    try:
        with open(FORWARDS_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_forwards(forwards):
    """Persist forward positions to disk."""
    with open(FORWARDS_FILE, 'w') as f:
        json.dump(forwards, f, indent=2)


def floor_3(v: float) -> float:
    """
    Floor a numeric value to 3 decimal places (no rounding).
    Example: 1.999999 -> 1.999, 1.994 -> 1.994
    """
    v = float(v or 0.0)
    return math.floor(v * 1000.0) / 1000.0

def buy_asset(symbol, quantity, price):
    """
    Update spot portfolio for a buy (or reducing a short).

    Args:
        symbol: Ticker symbol.
        quantity: Units to add.
        price: Execution price.
    Returns:
        dict | None: Updated position or None if closed.
    """
    portfolio = load_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = portfolio.get(symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        side = position.get('side', 'long')

        if side == 'long':
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            portfolio[symbol] = {
                'quantity': new_qty,
                'avg_price': round(new_avg, 2),
                'side': 'long',
                'last_updated': now
            }
        else:
            if quantity < old_qty:
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': 'short',
                    'last_updated': now
                }
            elif quantity == old_qty:
                del portfolio[symbol]
            else:
                new_long_qty = quantity - old_qty
                portfolio[symbol] = {
                    'quantity': new_long_qty,
                    'avg_price': round(price, 2),
                    'side': 'long',
                    'last_updated': now
                }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': round(price, 2),
            'side': 'long',
            'last_updated': now
        }
    save_portfolio(portfolio)
    return portfolio.get(symbol)


def sell_asset(symbol, quantity, price):
    """
    Update spot portfolio for a sell/short action.

    Args:
        symbol: Ticker symbol.
        quantity: Units to sell.
        price: Execution price.
    Returns:
        bool: True if applied.
    """
    portfolio = load_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = portfolio.get(symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        side = position.get('side', 'long')

        if side == 'long':
            if quantity < old_qty:
                new_qty = old_qty - quantity
                portfolio[symbol] = {
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'side': 'long',
                    'last_updated': now
                }
            elif quantity == old_qty:
                del portfolio[symbol]
            else:
                new_short_qty = quantity - old_qty
                portfolio[symbol] = {
                    'quantity': new_short_qty,
                    'avg_price': round(price, 2),
                    'side': 'short',
                    'last_updated': now
                }
        else:
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            portfolio[symbol] = {
                'quantity': new_qty,
                'avg_price': round(new_avg, 2),
                'side': 'short',
                'last_updated': now
            }
    else:
        portfolio[symbol] = {
            'quantity': quantity,
            'avg_price': round(price, 2),
            'side': 'short',
            'last_updated': now
        }

    save_portfolio(portfolio)
    return True


def process_sell_systems():
    """
    Run automated sell systems against current market data.

    - Loads configured systems and current portfolio.
    - Triggers sells when price <= configured levels.
    - Persists level state and portfolio updates.
    Side effects: network price fetches, JSON writes, Streamlit messages.
    """
    sell_systems = load_sell_systems()
    if not sell_systems:
        st.info("No sell systems configured.")
        return

    portfolio = load_portfolio()
    any_executed = False

    for symbol, config in sell_systems.items():
        if config.get("status") != "On":
            continue
        if symbol not in portfolio:
            continue

        current_price_data = get_data(symbol)
        current_price = current_price_data['price']
        if current_price <= 0:
            continue

        levels = config.get("levels", {})
        for level_key, level in levels.items():
            if level.get("triggered"):
                continue

            trigger_price = level.get("price")
            level_qty = int(level.get("quantity", 0))
            if trigger_price is None or level_qty <= 0:
                continue

            current_position_qty = portfolio.get(symbol, {}).get("quantity", 0)
            if current_position_qty <= 0:
                break

            if current_price <= trigger_price:
                qty_to_sell = min(level_qty, current_position_qty)
                if qty_to_sell <= 0:
                    continue

                if sell_asset(symbol, qty_to_sell, current_price):
                    level["triggered"] = True
                    any_executed = True
                    st.success(
                        f"Auto-sell executed for {symbol}: sold {qty_to_sell} units "
                        f"around ${current_price:.2f} (level {level_key})"
                    )
                    portfolio = load_portfolio()

        config["levels"] = levels

    if any_executed:
        save_sell_systems(sell_systems)
    else:
        st.info("No sell levels were triggered based on current market prices.")


def trade_option_contract(
    contract_symbol,
    underlying_symbol,
    option_type,
    strike,
    expiration,
    side,
    quantity,
    price,
    spot_at_trade=None,
):
    """
    Update options portfolio positions with a trade (buy/sell/close).

    Args:
        contract_symbol: OCC or custom id.
        underlying_symbol: Underlying ticker.
        option_type: "call" or "put".
        strike: Strike price.
        expiration: Expiration ISO date.
        side: "long" or "short".
        quantity: Units traded.
        price: Execution price.
        spot_at_trade: Optional underlying level at trade time.
    Returns:
        dict | None: Updated position or None if closed.
    """
    options_portfolio = load_options_portfolio()
    now = time.strftime('%Y-%m-%d %H:%M:%S')

    position = options_portfolio.get(contract_symbol)

    if position:
        old_qty = position['quantity']
        old_avg = position['avg_price']
        old_side = position.get('side', 'long')
        s0 = position.get('S0')

        if old_side == side:
            new_qty = old_qty + quantity
            new_avg = ((old_qty * old_avg) + (quantity * price)) / new_qty
            options_portfolio[contract_symbol] = {
                'underlying': underlying_symbol,
                'type': option_type,
                'strike': strike,
                'expiration': expiration,
                'quantity': new_qty,
                'avg_price': floor_3(new_avg),
                'S0': s0 if s0 is not None else (floor_3(spot_at_trade) if spot_at_trade is not None else None),
                'side': side,
                'last_updated': now,
            }
        else:
            if quantity < old_qty:
                new_qty = old_qty - quantity
                options_portfolio[contract_symbol] = {
                    'underlying': underlying_symbol,
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'quantity': new_qty,
                    'avg_price': old_avg,
                    'S0': s0,
                    'side': old_side,
                    'last_updated': now,
                }
            elif quantity == old_qty:
                del options_portfolio[contract_symbol]
            else:
                new_qty = quantity - old_qty
                options_portfolio[contract_symbol] = {
                    'underlying': underlying_symbol,
                    'type': option_type,
                    'strike': strike,
                    'expiration': expiration,
                    'quantity': new_qty,
                    'avg_price': floor_3(price),
                    'S0': s0 if s0 is not None else (floor_3(spot_at_trade) if spot_at_trade is not None else None),
                    'side': side,
                    'last_updated': now,
                }
    else:
        options_portfolio[contract_symbol] = {
            'underlying': underlying_symbol,
            'type': option_type,
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'avg_price': floor_3(price),
            'S0': floor_3(spot_at_trade) if spot_at_trade is not None else None,
            'side': side,
            'last_updated': now,
        }

    save_options_portfolio(options_portfolio)
    return options_portfolio.get(contract_symbol)


def fetch_options_chain(symbol):
    """
    Retrieve delayed CBOE option chain for a given ticker.

    Args:
        symbol: Underlying ticker symbol.
    Returns:
        list[dict]: Chain rows with symbol, strike, expiration, T, price, iv.
    Side effects:
        - Network I/O to cdn.cboe.com.
        - Streamlit error message on failure.
    """
    try:
        url = f"https://cdn.cboe.com/api/global/delayed_quotes/options/{symbol.upper()}.json"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        data_block = payload.get("data", {})
        options = data_block.get("options", [])
        raw_spot = data_block.get("current_price", 0.0) or 0.0
        spot = floor_3(raw_spot)

        today = datetime.date.today()
        chain = []

        for c in options:
            opt_symbol = c.get("option")
            if not opt_symbol or len(opt_symbol) < 15:
                continue

            # OCC symbology: root + YYMMDD + C/P + 8-digit strike
            date_code = opt_symbol[-15:-9]  # YYMMDD
            cp_flag = opt_symbol[-9]        # C or P
            strike_code = opt_symbol[-8:]   # 8 digits

            try:
                year = 2000 + int(date_code[0:2])
                month = int(date_code[2:4])
                day = int(date_code[4:6])
                expiration = datetime.date(year, month, day)
            except Exception:
                continue

            try:
                strike = int(strike_code) / 1000.0
                strike = floor_3(strike)
            except Exception:
                continue

            days_to_expiry = (expiration - today).days
            T = max(days_to_expiry, 0) / 365.0

            bid = c.get("bid", 0.0) or 0.0
            ask = c.get("ask", 0.0) or 0.0
            last = c.get("last_trade_price", 0.0) or 0.0

            if bid > 0 and ask > 0:
                price = (bid + ask) / 2.0
            elif last > 0:
                price = last
            else:
                price = max(bid, ask)

            price = floor_3(price)

            iv = c.get("iv", 0.0) or 0.0

            chain.append(
                {
                    "symbol": opt_symbol,
                    "underlying": data_block.get("symbol", symbol.upper()),
                    "spot": spot,
                    "type": "call" if cp_flag.upper() == "C" else "put",
                    "strike": strike,
                    "expiration": expiration.isoformat(),
                    "T": T,
                    "price": price,
                    "iv": iv,
                }
            )

        return chain
    except Exception as e:
        st.error(f"Error fetching options from CBOE for {symbol}: {e}")
        return []


def norm_cdf(x: float) -> float:
    """Cumulative distribution function for the standard normal."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def black_scholes_price(S: float, K: float, T: float, r: float, sigma: float, option_type: str, q: float = 0.0) -> float:
    """
    Black-Scholes price for a European option.

    Args:
        S: Spot price.
        K: Strike price.
        T: Time to maturity in years.
        r: Risk-free rate.
        sigma: Volatility.
        option_type: "call" or "put".
        q: Continuous dividend yield.
    Returns:
        float: BSM theoretical price; falls back to intrinsic if inputs degenerate.
    """
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        if option_type == "call":
            return max(S - K, 0.0)
        else:
            return max(K - S, 0.0)

    r_eff = float(r) - float(q or 0.0)
    d1 = (math.log(S / K) + (r_eff + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    if option_type == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def get_underlying_close_on_date(symbol: str, date: datetime.date) -> float:
    """
    Try to get the underlying daily close on a given date using Alpaca data.
    Falls back to current price if historical data is unavailable.
    """
    symbol = (symbol or "").strip().upper()
    if not symbol:
        return 0.0

    # Primary: yfinance daily close for the specific date
    try:
        start = date.isoformat()
        end = (date + datetime.timedelta(days=1)).isoformat()
        hist = yf.Ticker(symbol).history(start=start, end=end, interval="1d")
        if not hist.empty and "Close" in hist.columns:
            return float(hist["Close"].iloc[-1] or 0.0)
    except Exception:
        pass

    # Fallback: latest available price (API or yfinance) to avoid noisy warnings
    price_data = get_data(symbol)
    return float(price_data.get("price", 0.0) or 0.0)


def process_expired_options():
    """
    Mark expired options inside the unified book, computing payoff and realized PnL
    using the underlying close on expiration.
    """
    book = load_options_book()
    if not book:
        return

    today = datetime.date.today()

    changed = False
    for option_id, pos in list(book.items()):
        exp_str = pos.get("expiration")
        if not exp_str:
            continue

        try:
            exp_date = datetime.date.fromisoformat(exp_str)
        except Exception:
            continue

        if exp_date > today or pos.get("status", "open") != "open":
            continue

        underlying = pos.get("underlying")
        quantity = float(pos.get("quantity", 0) or 0)
        if quantity <= 0:
            continue

        S_T = get_underlying_close_on_date(underlying, exp_date) if underlying else 0.0
        pnl = compute_option_pnl(pos, S_T)

        book[option_id] = {
            **pos,
            **pnl,
            "status": "expired",
            "underlying_close": S_T,
            "closed_at": today.isoformat(),
        }
        changed = True

    if changed:
        save_options_book(book)

def chatgpt_response(message: str):
    """
    Call OpenAI ChatGPT with portfolio/orders context and user question.

    Args:
        message: User prompt/question.
    Returns:
        str: Assistant response or error string.
    Side effects:
        - Network call to OpenAI API.
        - Reads OPENAI_API_KEY from environment.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        portfolio_data = json.dumps(fetch_portfolio(), indent=2)
        open_orders = json.dumps(fetch_open_orders(), indent=2)
        
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an expert AI Portfolio Manager specializing in portfolio analysis, "
                    "risk management, and strategic market insights. "
                    "Your goal is to provide data-driven evaluations and professional recommendations."
                )
            },
            {
                "role": "user",
                "content": f"""
                    Here is my current portfolio:
                    {portfolio_data}

                    Here are my open orders:
                    {open_orders}

                    Your tasks:
                    1. Evaluate the risk exposures of my current holdings.
                    2. Analyze the potential impact of open orders.
                    3. Provide insights into portfolio health, diversification, and trade adjustments.
                    4. Speculate on market outlook given current conditions.
                    5. Identify potential risks and suggest mitigation strategies.

                    Finally, answer this specific question:
                    {message}
                    """
            }
        ]
        
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            temperature=0.3
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}"

def get_max_entry_price(symbol):
    """
    Return the maximum filled average price for a symbol from recent orders.

    Args:
        symbol: Ticker symbol to inspect.
    Returns:
        float: Highest filled_avg_price or -1 if none.
    Side effects: Alpaca list_orders call; Streamlit error on failure.
    """
    try:
        orders = api.list_orders(status="filled", limit=50)
        prices = [float(order.filled_avg_price) for order in orders if order.filled_avg_price and order.symbol == symbol]
        return max(prices) if prices else -1
    except Exception as e:
        st.error(f"Error fetching orders: {e}")
        return 0

def place_initial_order(symbol):
    """
    Submit a market buy order (qty=1) via Alpaca.

    Args:
        symbol: Ticker to buy.
    Returns:
        bool: True if submitted, False on error.
    Side effects: Network order placement and Streamlit messaging.
    """
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side="buy",
            type="market",
            time_in_force="gtc"
        )
        st.success(f"Initial order placed for {symbol}")
        time.sleep(2)
        return True
    except Exception as e:
        st.error(f"Error placing initial order: {e}")
        return False

def place_limit_order(symbol, price):
    """
    Submit a limit buy order (qty=1) via Alpaca.

    Args:
        symbol: Ticker to buy.
        price: Limit price.
    Returns:
        bool: True on success, False on error.
    Side effects: Network order placement and Streamlit messaging.
    """
    try:
        api.submit_order(
            symbol=symbol,
            qty=1,
            side='buy',
            type='limit',
            time_in_force='gtc',
            limit_price=price
        )
        st.success(f"Placed limit order for {symbol} @ ${price}")
        return True
    except Exception as e:
        st.error(f"Error placing order: {e}")
        return False

# Always keep cached data fresh on each rerun
st.cache_data.clear()

# Process any expired options and realize their PnL
process_expired_options()

# Streamlit UI
st.title("📈 AI assisted Trading system")
st.markdown("---")

with st.expander("📘 Tutoriel d'utilisation de l'outil"):
    st.markdown("""
    ### 📘 Mise en situation : ce que vous essayez de faire
    
    Imaginez que vous êtes un investisseur particulier qui en a assez de trader "au feeling" : 
    vous voulez arrêter de courir après le marché, structurer vos décisions et savoir exactement 
    pourquoi vous entrez, renforcez ou sortez d'une position.
    
    Cet outil est là pour vous aider à **transformer votre trading en processus**. 
    Vous cherchez principalement à optimiser trois choses :
    
    - **Votre risque** : limiter la profondeur des pertes (drawdown) que vous êtes prêt à accepter
    - **Votre prix moyen d'entrée** : profiter des baisses pour améliorer vos points d'achat au lieu de paniquer
    - **Votre temps et votre charge mentale** : automatiser ce qui peut l'être, et garder votre énergie pour les décisions importantes
    
    L'idée n'est pas de prédire le futur, mais de mettre un cadre autour de votre comportement : 
    ouvrir l'application, voir en quelques secondes si tout est sous contrôle, puis décider calmement 
    s'il y a une action à prendre aujourd'hui ou non.
    
    ---
    
    ### 🧭 Comment lire l'application
    
    L'application est construite comme un **parcours logique de trader discipliné** :
    
    1. Vous **observez** votre situation (ce que vous possédez, comment ça évolue, ce que vos systèmes ont fait)
    2. Vous **décidez** où mettre du capital, ce que vous voulez renforcer ou alléger
    3. Vous **exécutez** des ordres manuels quand vous voulez intervenir directement
    4. Vous **automatisez** certaines parties de votre stratégie avec des systèmes basés sur le drawdown et des niveaux de prix
    5. Vous **analysez** avec l'IA pour challenger vos idées, comprendre vos risques et clarifier votre stratégie
    
    Chaque onglet correspond à une étape de ce parcours. 
    Dans les sections "📚 Comment utiliser ..." au bas de chaque onglet, 
    vous trouverez une explication détaillée de **ce que vous êtes en train d'y faire** 
    (qu'est-ce que vous optimisez, quels sont les enjeux, où se situe la difficulté mentale).
    
    ---
    
    ### 🎯 Comment bien démarrer
    
    Pour une première utilisation, vous pouvez suivre ce mini-scénario :
    
    1. Ouvrez l'application et prenez un moment pour comprendre que le but n'est pas de "trader plus", 
       mais de **trader mieux, avec un plan**.
    2. Allez ensuite dans chaque onglet, l'un après l'autre, sans forcément passer d'ordres au début :
       contentez-vous de lire le texte d'aide en bas de page et de repérer les boutons qui déclenchent de vraies actions.
    3. Quand vous vous sentez à l'aise, commencez petit : 
       un premier achat manuel, un premier système dans Trading Systems, une première question à l'IA.
    4. Revenez quelques jours plus tard pour voir ce qui s'est passé : 
       avez-vous respecté votre plan ? vos systèmes ont-ils réagi comme prévu ? qu'avez-vous appris ?
    
    Utilisé de cette manière, cet outil devient **un cadre d'apprentissage et d'optimisation** : 
    à chaque utilisation, vous comprenez un peu mieux votre propre comportement de trader, 
    et vous ajustez votre manière d'investir pour qu'elle soit plus cohérente, plus sereine et plus alignée avec votre tolérance au risque.
    """)

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    st.info("Data is refreshed automatically on each interaction")
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

# Main tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "💰 Buy/Sell",
    "📋 Trading Systems",
    "📄 Forwards",
    "📜 Options",
])
# Rendre la barre d'onglets principale sticky pour qu'elle reste visible au scroll.
st.markdown(
    """
    <style>
    .sticky-main-tabs {
        position: sticky;
        top: 0;
        z-index: 999;
        background: var(--background-color, #fff);
        padding: 0.3rem 0;
        border-bottom: 1px solid rgba(0, 0, 0, 0.1);
    }
    </style>
    <script>
    const lists = window.parent.document.querySelectorAll('div[data-baseweb="tab-list"]');
    if (lists.length > 0) {
        const mainTabs = lists[0];
        if (mainTabs && !mainTabs.classList.contains('sticky-main-tabs')) {
            mainTabs.classList.add('sticky-main-tabs');
        }
    }
    </script>
    """,
    unsafe_allow_html=True,
)
if st.session_state.pop("_switch_to_dashboard", False):
    st.markdown(
        """
        <script>
        const tabs = window.parent.document.querySelectorAll('button[data-baseweb="tab"]');
        for (const t of tabs) {
          const label = (t.innerText || "").trim();
          if (label.includes("📊 Dashboard") || label.includes("Dashboard")) {
            t.click();
            break;
          }
        }
        </script>
        """,
        unsafe_allow_html=True,
    )

# Tab 1: Dashboard
with tab1:
    # Quick P&L synthesis (spot, options, forwards) before AI widget
    spot_portfolio = load_portfolio()
    spot_total_pnl = 0.0
    spot_total_notional = 0.0
    for symbol, data in spot_portfolio.items():
        avg_price = float(data.get("avg_price", 0.0) or 0.0)
        quantity = float(data.get("quantity", 0.0) or 0.0)
        side = (data.get("side") or "long").lower()
        current_price = float(get_data(symbol).get("price", avg_price) or avg_price)
        position_value = quantity * current_price
        notional = avg_price * quantity
        if side == "long":
            pnl = (current_price - avg_price) * quantity
        else:
            pnl = (avg_price - current_price) * quantity
        spot_total_pnl += pnl
        spot_total_notional += notional

    # Options P&L (open MTM) + realized from expired
    options_book_all = load_options_book()
    open_options = {k: v for k, v in options_book_all.items() if v.get("status", "open") == "open"}
    open_options_pnl = 0.0
    for _, pos in open_options.items():
        qty = float(pos.get("quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        avg_price = float(pos.get("avg_price", pos.get("T_0_price", 0.0)) or 0.0)
        side = (pos.get("side") or "long").lower()
        _, _, _, mark_price, _ = mark_option_market_value(pos, chain_entry=None)
        mark_price = float(mark_price or 0.0)
        if side == "long":
            pnl = (mark_price - avg_price) * qty
        else:
            pnl = (avg_price - mark_price) * qty
        open_options_pnl += pnl
    realized_options_pnl = sum(
        float(opt.get("pnl_total", 0.0) or 0.0) for opt in load_expired_options().values()
    )
    total_options_pnl = open_options_pnl + realized_options_pnl

    # Forwards P&L
    fwd_positions = load_forwards()
    total_forward_pnl = 0.0
    total_forward_notional = 0.0
    for fwd in fwd_positions.values():
        qty = float(fwd.get("quantity", 0.0) or 0.0)
        if qty <= 0:
            continue
        side = (fwd.get("side") or "long").lower()
        mult = 1.0 if side == "long" else -1.0
        sym = fwd.get("symbol", "")
        spot_now = float(get_data(sym).get("price", 0.0) or 0.0) if sym else 0.0
        price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
        total_forward_pnl += mult * (spot_now - price_fwd) * qty
        total_forward_notional += abs(price_fwd * qty)

    st.markdown("### 📊 Synthèse P&L")
    col_spot, col_opt, col_fwd = st.columns(3)
    with col_spot:
        delta_spot = f"{(spot_total_pnl / spot_total_notional * 100):.2f}%" if spot_total_notional > 0 else None
        st.metric("P&L Spot", f"${spot_total_pnl:.2f}", delta=delta_spot)
    with col_opt:
        delta_opt = f"Open {open_options_pnl:+.2f} | Réalisé {realized_options_pnl:+.2f}"
        st.metric("P&L Options (open+réalisé)", f"${total_options_pnl:.2f}", delta=delta_opt)
    with col_fwd:
        delta_fwd = f"{(total_forward_pnl / total_forward_notional * 100):.2f}%" if total_forward_notional > 0 else None
        st.metric("P&L Forwards", f"${total_forward_pnl:.2f}", delta=delta_fwd)

    # AI Assistant section moved here
    st.subheader("🤖 AI Portfolio Assistant")
    st.markdown("Ask questions about your portfolio and get AI-powered insights")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask about your portfolio..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = chatgpt_response(prompt)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

    st.markdown("---")

    with st.expander("📘 Comprendre le Dashboard"):
        st.markdown("""
        ### 📊 Ce que vous faites dans le Dashboard
        
        Le Dashboard est votre **vue d’ensemble** : en haut, vous discutez avec l’IA à propos de votre portfolio; en dessous,
        vous voyez vos positions chiffrées de façon froide et objective.
        
        La section *AI Portfolio Manager* sert à poser vos questions stratégiques : risques, diversification, idées de gestion,
        explications de concepts. L’IA répond en tenant compte de vos positions et de vos ordres ouverts.
        
        Le bloc *My Portfolio* liste chaque actif (long ou short), avec quantité, prix moyen, prix spot, valeur et P&L.
        Plus bas, la section *Configured Trading Systems* montre vos robots actifs ou en attente.
        """)
    st.subheader("💰 My Portfolio")
    my_portfolio = load_portfolio()
    
    if my_portfolio:
        portfolio_data = []
        total_value = 0
        total_pnl = 0
        total_notional = 0

        for symbol, data in my_portfolio.items():
            current_price_data = get_data(symbol)
            avg_price = data['avg_price']
            quantity = data['quantity']
            side = data.get('side', 'long')

            current_price = current_price_data['price'] if current_price_data['price'] > 0 else avg_price
            position_value = quantity * current_price

            if side == 'long':
                pnl = (current_price - avg_price) * quantity
            else:
                pnl = (avg_price - current_price) * quantity

            notional = avg_price * quantity
            pnl_pct = (pnl / notional * 100) if notional > 0 else 0

            total_value += position_value
            total_pnl += pnl
            total_notional += notional
            
            portfolio_data.append({
                'Symbol': symbol,
                'Side': side.capitalize(),
                'Quantity': quantity,
                'S_0 Price': f"${avg_price:.2f}",
                'Current Price': f"${current_price:.2f}",
                'Value': f"${position_value:.2f}",
                'P&L': f"${pnl:.2f}",
                'P&L %': f"{pnl_pct:.2f}%",
                '_value_raw': position_value,
            })
        
        # Allocate % of each position relative to total portfolio value
        for row in portfolio_data:
            alloc = (row.get("_value_raw", 0.0) / total_value * 100) if total_value > 0 else 0.0
            row["Allocation %"] = f"{alloc:.2f}%"
        df_my_portfolio = pd.DataFrame(portfolio_data)
        # Drop helper column used for allocation and reorder columns
        df_my_portfolio = df_my_portfolio.drop(columns=[c for c in df_my_portfolio.columns if c.startswith("_")], errors="ignore")
        col_order = [
            "Allocation %",
            "Symbol",
            "Side",
            "Quantity",
            "S_0 Price",
            "Current Price",
            "Value",
            "P&L",
            "P&L %",
        ]
        df_my_portfolio = df_my_portfolio[[c for c in col_order if c in df_my_portfolio.columns]]
        st.dataframe(df_my_portfolio, width="stretch", hide_index=True)

        # Include realized PnL from expired options
        expired_options = load_expired_options()
        # Merge legacy options_book.json content for display/PnL, to ensure completeness
        legacy_book = load_options_book_legacy_only()
        for opt_id, entry in legacy_book.items():
            if opt_id not in expired_options and entry.get("status") in {"expired", "closed"}:
                expired_options[opt_id] = entry

        st.markdown("### ✅ Options expirées / clôturées")
        if expired_options:
            exp_rows = []
            for key, opt in expired_options.items():
                exp_rows.append({
                    "ID/Contract": key,
                    "Underlying": opt.get("underlying"),
                    "Type": str(opt.get("option_type") or opt.get("type", "")).capitalize(),
                    "Side": str(opt.get("side", "")).capitalize(),
                    "Strike": opt.get("strike"),
                    "Expiration": opt.get("expiration"),
                    "Qty": opt.get("quantity"),
                    "Avg Price": opt.get("avg_price"),
                    "Underlying close": opt.get("underlying_close"),
                    "Payoff/unit": opt.get("payoff_per_unit"),
                    "PnL/unit": opt.get("pnl_per_unit"),
                    "PnL total": opt.get("pnl_total"),
                    "Closed at": opt.get("closed_at"),
                    "Source": opt.get("_source", "book"),
                })
            df_exp = pd.DataFrame(exp_rows)
            st.dataframe(df_exp, width="stretch", hide_index=True)
            if legacy_book:
                st.caption("Source merge : options_portfolio.json + options_book.json (legacy).")

            with st.expander("🔎 Détails payoff / méthode de calcul (toutes les options)", expanded=False):
                for key, opt in expired_options.items():
                    info = describe_expired_option_payoff(opt)
                    title = f"{key} – {opt.get('product_type') or opt.get('product') or opt.get('type') or ''}"
                    st.markdown(f"**{title}**")
                    st.write(f"**Méthode** : {info['method_label']} (fonction `{info['function']}`)")
                    st.caption(info["description"])
                    c1, c2 = st.columns(2)
                    with c1:
                        st.write(f"Payoff/unité: {info['payoff_unit']:.4f}")
                        st.write(f"Payoff/unité signé (side): {info['payoff_signed_unit']:.4f}")
                    with c2:
                        st.write(f"Payoff total (qty x side): {info['payoff_total']:.4f}")
                        st.write(f"Sous-jacent à l'échéance: {info['underlying_close']}")
                        st.write(f"Side: {info['side']} | Qty: {info['quantity']}")
                    st.write("Paramètres utilisés :")
                    st.json(info["params"])
                    st.markdown("---")
        else:
            st.info("Aucune option expirée/close pour l’instant.")

        # Forward portfolio section
        st.markdown("---")
        st.markdown("### 🚀 Forward Portfolio")
        forwards_dash = load_forwards()
        if forwards_dash:
            fwd_rows = []
            today_dash = datetime.date.today()
            for key, fwd in forwards_dash.items():
                sym = fwd.get("symbol", "")
                qty = int(fwd.get("quantity", 0) or 0)
                price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
                side = fwd.get("side", "long")
                maturity_str = fwd.get("maturity")
                try:
                    maturity_dt = datetime.date.fromisoformat(maturity_str)
                    days_to_mat = (maturity_dt - today_dash).days
                except Exception:
                    maturity_dt = None
                    days_to_mat = None
                spot_now = float(get_data(sym).get("price", 0.0) or 0.0) if sym else 0.0
                mult = 1.0 if side == "long" else -1.0
                pnl_unit = mult * (spot_now - price_fwd)
                pnl_total = pnl_unit * qty
                fwd_rows.append(
                    {
                        "Symbol": sym,
                        "Side": side.capitalize(),
                        "Quantity": qty,
                        "Forward Price": price_fwd,
                        "Spot Now": round(spot_now, 4),
                        "Maturity": maturity_str,
                        "Days to mat": days_to_mat,
                        "P&L/unit": round(pnl_unit, 4),
                        "P&L total": round(pnl_total, 2),
                    }
                )
            df_fwds = pd.DataFrame(fwd_rows)
            st.dataframe(df_fwds, width="stretch", hide_index=True)
        else:
            st.info("No forward positions yet.")

        st.markdown("---")
        st.markdown("### 🧾 Custom option purchases (acheté depuis pricing)")
        custom_opts = {
            k: v for k, v in load_options_book().items() if v.get("status", "open") == "open"
        }
        if custom_opts:
            underlyings_custom = sorted({
                pos.get("underlying")
                for pos in custom_opts.values()
                if pos.get("underlying")
            })
            chains_by_underlying_custom = {}
            for und in underlyings_custom:
                chain_list = fetch_options_chain(und)
                chains_by_underlying_custom[und] = chain_list
            mark_map = {}
            def _pricing_metadata(prod: str) -> tuple[str, list[str]]:
                prod_low = (prod or "").lower()
                if "barrier" in prod_low:
                    return "barrier_pricer", ["option_type", "barrier_type", "knock", "direction", "barrier_level", "strike", "expiration", "sigma", "S0", "r", "q"]
                if "asian" in prod_low:
                    return "asian_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q", "n_obs"]
                if "digital" in prod_low:
                    return "digital_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q", "payout"]
                if "asset-or-nothing" in prod_low:
                    return "asset_or_nothing_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q"]
                if "forward-start" in prod_low:
                    return "forward_start_pricer", ["option_type", "strike", "T_start", "expiration", "sigma", "S0", "r", "q"]
                if "chooser" in prod_low:
                    return "chooser_pricer", ["option_type", "strike", "expiration", "t_choice", "sigma", "S0", "r", "q"]
                if "straddle" in prod_low:
                    return "straddle_pricer", ["S0", "strike", "expiration", "sigma", "r", "q"]
                if "strangle" in prod_low:
                    return "strangle_pricer", ["S0", "strike_put", "strike_call", "expiration", "sigma", "r", "q"]
                if "spread" in prod_low or "butterfly" in prod_low or "condor" in prod_low:
                    return "spread_pricer", ["legs (strike/side/type)", "expiration", "sigma", "S0", "r", "q"]
                if "lookback" in prod_low:
                    return "lookback_pricer", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q", "n_paths", "n_steps"]
                if "cliquet" in prod_low or "ratchet" in prod_low:
                    return "cliquet_pricer", ["n_periods", "cap", "floor", "sigma", "S0", "r", "q", "n_paths"]
                return "vanilla_bsm", ["option_type", "strike", "expiration", "sigma", "S0", "r", "q"]
            rows_custom = []
            def _safe_f(val, default=0.0):
                try:
                    return float(val)
                except Exception:
                    return default
            for key, pos in custom_opts.items():
                try:
                    underlying = pos.get("underlying")
                    option_type_raw = (
                        pos.get("option_type")
                        or pos.get("cpflag")
                        or pos.get("cp_flag")
                        or pos.get("cp")
                        or ""
                    )
                    if not option_type_raw:
                        t_val = str(pos.get("type", "")).lower()
                        if t_val in {"call", "put"}:
                            option_type_raw = t_val
                    option_type = str(option_type_raw).lower()
                    strike = _safe_f(pos.get("strike", 0.0), 0.0)
                    strike2_val = pos.get("strike2")
                    strike2 = _safe_f(strike2_val, None) if strike2_val not in (None, "") else None
                    quantity = _safe_f(pos.get("quantity", 0), 0.0)
                    avg_price = _safe_f(pos.get("avg_price", pos.get("T_0_price", 0.0)), 0.0)
                    side = pos.get("side", "long").lower()
                    product = (
                        pos.get("product_type")
                        or pos.get("structure")
                        or pos.get("product")
                        or pos.get("type")
                        or "vanilla"
                    )
                    misc = pos.get("misc")

                    chain_list = chains_by_underlying_custom.get(underlying, [])
                    chain_entry = None
                    if chain_list:
                        try:
                            expiry_date = datetime.date.fromisoformat(pos.get("expiration"))
                            days_to_expiry = (expiry_date - datetime.date.today()).days
                            target_T = max(days_to_expiry, 0) / 365.0
                        except Exception:
                            target_T = None

                        if target_T is not None and strike > 0:
                            nearest = None
                            best_score = float("inf")
                            for c in chain_list:
                                cT = float(c.get("T", 0.0) or 0.0)
                                cK = float(c.get("strike", 0.0) or 0.0)
                                scale = max(strike, 1.0)
                                score = abs(cT - target_T) + abs(cK - strike) / scale
                                if score < best_score:
                                    best_score = score
                                    nearest = c
                            chain_entry = nearest

                    spot, T_years, sigma_used, mark_price, mark_method = mark_option_market_value(pos, chain_entry=chain_entry)
                    has_price = mark_price is not None and mark_price > 0

                    if side == "long":
                        pnl_per_unit = mark_price - avg_price
                    else:
                        pnl_per_unit = avg_price - mark_price
                    total_pnl_opt = pnl_per_unit * quantity

                    mark_map[key] = {
                        "spot": spot,
                        "T": T_years,
                        "sigma": sigma_used,
                        "mark_price": mark_price,
                        "method": mark_method,
                        "pnl": total_pnl_opt,
                        "pnl_per_unit": pnl_per_unit,
                    }

                    rows_custom.append({
                        "ID/Contract": key,
                        "Product": product,
                        "Underlying": underlying,
                        "Type": option_type.capitalize(),
                        "Side": side.capitalize(),
                        "Strike": strike,
                        "Strike2": strike2,
                        "Expiration": pos.get("expiration"),
                        "Quantity": quantity,
                        "T_0 Price": avg_price,
                        "Current Price (model)": mark_price,
                        "Model P&L": total_pnl_opt,
                        "Misc (json)": json.dumps(misc or {}, ensure_ascii=False),
                    })
                except Exception as exc:
                    rows_custom.append({
                        "ID/Contract": key,
                        "Product": pos.get("product"),
                        "Error": str(exc),
                    })

        mark_map[key] = {
            "spot": spot,
            "T": T_years,
            "sigma": sigma_used,
            "mark_price": mark_price,
            "method": mark_method,
            "pnl": total_pnl_opt,
            "pnl_per_unit": pnl_per_unit,
        }

        rows_custom.append({
            "ID/Contract": key,
            "Product": product,
            "Underlying": underlying,
            "Type": option_type.capitalize(),
            "Side": side.capitalize(),
            "Strike": strike,
            "Strike2": strike2,
            "Expiration": pos.get("expiration"),
            "Quantity": quantity,
            "T_0 Price": avg_price,
            "Current Price (model)": mark_price,
            "Model P&L": total_pnl_opt,
            "Misc (json)": json.dumps(misc or {}, ensure_ascii=False),
        })

        if rows_custom:
            df_custom = pd.DataFrame(rows_custom)
            st.dataframe(df_custom, width="stretch", hide_index=True)
            with st.expander("Manage custom options", expanded=False):
                for key, pos in custom_opts.items():
                    snap = mark_map.get(key, {})
                    spot = float(snap.get("spot") or 0.0)
                    mark_price = float(snap.get("mark_price") or 0.0)
                    has_price = mark_price is not None and mark_price > 0
                    pnl_total = float(snap.get("pnl") or 0.0)
                    pnl_per_unit = float(snap.get("pnl_per_unit") or 0.0)
                    method = snap.get("method", "n/a")
                    sigma_used = float(snap.get("sigma") or 0.0)
                    T_years = snap.get("T")

                    strike = pos.get("strike")
                    qty = pos.get("quantity")
                    if qty is None:
                        qty = 0
                    avg_price = pos.get("avg_price")
                    if avg_price is None:
                        avg_price = pos.get("T_0_price", 0.0)
                    avg_price = float(avg_price or 0.0)
                    side = pos.get("side", "long").lower()
                    product = (
                        pos.get("product_type")
                        or pos.get("structure")
                        or pos.get("product")
                        or pos.get("type")
                        or "vanilla"
                    )
                    underlying = pos.get("underlying")
                    option_type_raw = (
                        pos.get("option_type")
                        or pos.get("cpflag")
                        or pos.get("cp_flag")
                        or pos.get("cp")
                        or ""
                    )
                    if not option_type_raw:
                        t_val = str(pos.get("type", "")).lower()
                        if t_val in {"call", "put"}:
                            option_type_raw = t_val
                    option_type = str(option_type_raw).lower()
                    strike2_val = pos.get("strike2")
                    strike2 = float(strike2_val) if strike2_val not in (None, "") else None
                    misc = pos.get("misc")
                    pricing_fn, needed = _pricing_metadata(product)
                    product_low = product.lower() if product else ""
                    available = set(k for k in [
                        "option_type",
                        "strike" if strike is not None else None,
                        "strike2" if strike2 is not None else None,
                        "expiration" if pos.get("expiration") else None,
                        "quantity" if qty else None,
                        "sigma" if sigma_used else None,
                        "S0" if spot else None,
                        "legs (strike/side/type)" if pos.get("legs") else None,
                    ] if k)
                    # We can compute r/q from rates_utils
                    available.update({"r", "q"})
                    # For spreads/structures without explicit legs, consider strike/strike2 as leg info
                    if any(keyword in product_low for keyword in ["spread", "butterfly", "condor", "straddle", "strangle"]):
                        available.add("legs (strike/side/type)")
                    if isinstance(misc, dict):
                        available.update(misc.keys())
                        if "barrier" in product_low and "knock" in misc and "direction" not in available:
                            available.add("direction")
                    missing = set(needed) - available
                    extra = available - set(needed)
                    btn_label = "Close"
                    close_max = max(1, int(qty)) if qty and qty > 0 else 1

                    with st.expander(f"{key} ({product})"):
                        side_label = side.upper() if side else "N/A"
                        opt_label = option_type.upper() if option_type else "N/A"
                        st.caption(f"Produit reconnu: {product} | Option type: {opt_label} | Side: {side_label}")
                        close_qty = st.selectbox(
                            "Quantity to close",
                            options=list(range(1, close_max + 1)),
                            index=close_max - 1 if close_max > 0 else 0,
                            key=f"close_qty_{key}",
                        )
                        col_a, col_b, col_c = st.columns(3)
                        with col_a:
                            st.metric("Prix actuel", f"${mark_price:.4f}")
                            st.caption(
                                f"S = {spot:.4f} | K = {strike}"
                                + (f" | K2 = {strike2}" if strike2 is not None else "")
                            )
                            st.caption(
                                f"T = {T_years:.4f}" if T_years is not None else "T inconnu"
                            )
                            st.caption(f"Méthode: {method} | σ={sigma_used:.4f}")
                        with col_b:
                            st.metric("T_0 Price", f"${avg_price:.4f}")
                            st.metric("Qty", f"{qty}")
                        with col_c:
                            st.metric("PnL if closed", f"${pnl_total:.2f}", delta=f"{pnl_per_unit:.4f}")
                        if isinstance(misc, dict) and misc:
                            st.caption(f"Misc: {misc}")
                        st.caption(f"Pricing fn: {pricing_fn}")
                        st.caption(f"Paramètres requis: {needed}")
                        st.caption(f"Paramètres disponibles: {sorted(available)}")
                        # Afficher les valeurs connues pour chaque paramètre requis
                        def _param_value(k: str):
                            if k == "option_type":
                                return option_type
                            if k == "strike":
                                return strike
                            if k == "strike2":
                                return strike2
                            if k == "expiration":
                                return pos.get("expiration")
                            if k == "quantity":
                                return qty
                            if k == "sigma":
                                return sigma_used
                            if k == "S0":
                                return spot
                            if k == "r":
                                return (misc or {}).get("r")
                            if k == "q":
                                return (misc or {}).get("q")
                            if k == "n_obs":
                                return (misc or {}).get("n_obs")
                            if k == "n_paths":
                                return (misc or {}).get("n_paths")
                            if k == "n_steps":
                                return (misc or {}).get("n_steps")
                            if k == "barrier_level":
                                return (misc or {}).get("barrier") or (misc or {}).get("barrier_level")
                            if k == "barrier_type":
                                return (misc or {}).get("barrier_type")
                            if k == "knock" or k == "direction":
                                return (misc or {}).get("knock") or (misc or {}).get("direction")
                            if k == "legs (strike/side/type)":
                                return pos.get("legs")
                            if k == "T_start":
                                return (misc or {}).get("T_start")
                            if k == "t_choice":
                                return (misc or {}).get("t_choice")
                            if k == "cap":
                                return (misc or {}).get("cap")
                            if k == "floor":
                                return (misc or {}).get("floor")
                            return (misc or {}).get(k) or pos.get(k)

                        if needed:
                            st.caption("Valeurs actuelles des paramètres requis :")
                            param_lines = [f"- `{k}` : { _param_value(k) }" for k in needed]
                            st.markdown("\n".join(param_lines))
                        if missing:
                            st.markdown(f"🔴 Paramètres manquants: {sorted(missing)}")
                        elif extra:
                            st.markdown(f"🟨 Paramètres en trop: {sorted(extra)}")
                        else:
                            st.markdown("🟩 Paramètres OK pour pricer")

                        price_issue = not has_price or mark_price <= 0
                        if price_issue:
                            st.markdown("🔴 PnL non calculable : prix de marché/spot manquant.")

                        if st.button(f"✅ {btn_label}", key=f"close_custom_{key}", disabled=bool(missing or price_issue)):
                            book = load_options_book()
                            entry = dict(book.get(key, pos))
                            entry["option_type"] = option_type
                            entry["type"] = option_type

                            close_entry = dict(entry)
                            close_entry["quantity"] = close_qty
                            close_entry.update({
                                "status": "expired" if close_qty >= qty else "open",
                                "closed_at": datetime.date.today().isoformat(),
                                "underlying_close": spot,
                                "mark_close": mark_price,
                            })
                            pnl_vals = compute_option_pnl(close_entry, spot, mark_price=mark_price)
                            close_entry.update(pnl_vals)

                            if close_qty >= qty:
                                book[key] = close_entry
                            else:
                                # partial close: reduce remaining quantity
                                remaining = max(qty - close_qty, 0)
                                entry["quantity"] = remaining
                                book[key] = entry
                                book[f"{key}_closed_{int(time.time())}"] = close_entry

                            save_options_book(book)
                            st.success(f"Closed {close_qty} of {key} ({product}), realized PnL ≈ ${pnl_vals['pnl_total']:.2f}")
                            time.sleep(1)
                            st.rerun()
        else:
            st.info("Aucune option custom à afficher pour le moment.")
    else:
        st.info("Aucune option custom dans la liste.")
    
    # Trading Systems Section
    st.markdown("---")
    st.subheader("🎯 Configured Trading Systems")
    equities = load_equities()
    sell_systems = load_sell_systems()
    
    if equities:
        systems_data = []
        for symbol, data in equities.items():
            direction = data.get('direction', 'long')
            systems_data.append({
                'Symbol': symbol,
                'Direction': direction.capitalize(),
                'Position': data['position'],
                'Entry Price': f"${data['entry_price']:.2f}",
                'Drawdown': f"{data['drawdown']*100:.1f}%",
                'Levels': len(data['levels']),
                'Status': data['status']
            })
        
        df_systems = pd.DataFrame(systems_data)
        st.dataframe(df_systems, width="stretch", hide_index=True)

        st.markdown("#### 🔎 Niveaux d'achat / vente par système")
        for symbol, data in equities.items():
            buy_levels = data.get("levels", {}) or {}
            sell_cfg = sell_systems.get(symbol, {}) if isinstance(sell_systems, dict) else {}
            sell_levels = sell_cfg.get("levels", {}) if isinstance(sell_cfg, dict) else {}
            with st.expander(f"{symbol} – niveaux d'achat/vente", expanded=False):
                col_buy, col_sell = st.columns(2)
                with col_buy:
                    if buy_levels:
                        rows_buy = []
                        for lvl_key, price in sorted(buy_levels.items(), key=lambda kv: str(kv[0])):
                            rows_buy.append({"Level": str(lvl_key), "Buy @": price})
                        st.table(pd.DataFrame(rows_buy))
                    else:
                        st.caption("Aucun niveau d'achat configuré.")
                with col_sell:
                    if sell_levels:
                        rows_sell = []
                        for lvl_key, lvl in sorted(sell_levels.items(), key=lambda kv: str(kv[0])):
                            rows_sell.append({
                                "Level": str(lvl_key),
                                "Sell @": lvl.get("price"),
                                "Qty": lvl.get("quantity"),
                                "Triggered": bool(lvl.get("triggered")),
                            })
                        st.table(pd.DataFrame(rows_sell))
                    else:
                        st.caption("Aucun niveau de vente configuré.")
        
        # Quick remove section
        st.markdown("**Quick Remove:**")
        remove_cols = st.columns(len(equities) if len(equities) <= 5 else 5)
        for idx, symbol in enumerate(list(equities.keys())[:5]):
            with remove_cols[idx]:
                if st.button(f"🗑️ {symbol}", key=f"dash_remove_{symbol}"):
                    del equities[symbol]
                    save_equities(equities)
                    st.success(f"Removed {symbol}")
                    time.sleep(0.5)
                    st.rerun()
    else:
        st.info("No trading systems configured. Add one in the 'Trading Systems' tab.")

# Tab 2: Buy/Sell
with tab2:
    with st.expander("📘 Comprendre Buy/Sell"):
        st.markdown("""
        ### 💰 Ce que vous faites dans Buy/Sell
        
        Cet onglet est votre **poste d’exécution manuelle** : c’est ici que vous décidez
        consciemment d’entrer, renforcer, réduire ou retourner une position, en contrôlant précisément prix et quantité.
        
        Le bloc *Buy / Cover Asset* permet soit d’acheter pour être ou rester **Long**, soit d’acheter pour **couvrir un short**.
        Vous choisissez la *Direction* (Long/Short), le symbole, la quantité et le prix d’exécution.
        
        Le bloc *Sell / Short Asset* sert à gérer les positions existantes : vendre une partie d’un long, le clôturer entièrement,
        ou vendre au-delà de votre quantité actuelle pour devenir **net short** sur un actif.
        
        À droite, vous voyez à chaque fois la position en place (quantité, prix moyen, sens long/short) et le P&L estimé du trade
        avant de cliquer, ce qui vous aide à visualiser l’impact concret de l’ordre sur votre portefeuille.
        
        Utilisez cet onglet pour **intervenir manuellement** malgré vos systèmes automatiques : prendre des profits, couper une perte,
        inverser une position ou initier un short tactique, tout en gardant en tête le P&L et le risque global de votre compte.
        """)
    st.subheader("💰 Buy/Sell Assets")
    
    col1, col2 = st.columns(2)
    
    # BUY Section
    with col1:
        st.markdown("### 📈 Buy / Cover Asset")
        buy_side = st.radio(
            "Direction",
            options=["Long", "Short"],
            index=0,
            horizontal=True,
            key="buy_side"
        )
        buy_symbol = st.text_input("Symbol to Buy", placeholder="e.g., AAPL", key="buy_symbol").upper()
        buy_quantity = st.number_input("Quantity", min_value=1, value=1, step=1, key="buy_qty")
        
        if buy_symbol:
            price_data = get_data(buy_symbol)
            if price_data['price'] > 0:
                st.info(f"Current price: ${price_data['price']:.2f}")
                buy_price = st.number_input("Buy Price", min_value=0.01, value=float(price_data['price']), step=0.01, key="buy_price")
                total_cost = buy_quantity * buy_price
                st.metric("Total Cost", f"${total_cost:.2f}")
                
                if st.button("✅ Execute Order", type="primary", key="exec_buy"):
                    if buy_side == "Long":
                        result = buy_asset(buy_symbol, buy_quantity, buy_price)
                        st.success(f"Bought {buy_quantity} units of {buy_symbol} @ ${buy_price:.2f}")
                        if result:
                            side = result.get('side', 'long').upper()
                            st.info(
                                f"New position: {result['quantity']} units @ avg ${result['avg_price']:.2f} "
                                f"({side})"
                            )
                        else:
                            st.info("Position fully closed.")
                    else:
                        if sell_asset(buy_symbol, buy_quantity, buy_price):
                            st.success(f"Shorted {buy_quantity} units of {buy_symbol} @ ${buy_price:.2f}")
                            portfolio_after = load_portfolio()
                            new_pos = portfolio_after.get(buy_symbol)
                            if new_pos:
                                side = new_pos.get("side", "short").upper()
                                st.info(
                                    f"New position: {new_pos['quantity']} units @ avg ${new_pos['avg_price']:.2f} "
                                    f"({side})"
                                )
                    time.sleep(1)
                    st.rerun()
            else:
                st.error(f"Could not fetch price for {buy_symbol}")
    
    # SELL / SHORT Section
    with col2:
        st.markdown("### 📉 Sell / Short Asset")
        my_portfolio = load_portfolio()
        
        if my_portfolio:
            sell_symbol = st.selectbox("Symbol to Sell/Short", options=list(my_portfolio.keys()), key="sell_symbol")
            
            if sell_symbol:
                position = my_portfolio[sell_symbol]
                current_qty = position['quantity']
                avg_price = position['avg_price']
                side = position.get('side', 'long')
                
                st.info(
                    f"Current position: {current_qty} units @ avg ${avg_price:.2f} "
                    f"({side.upper()})"
                )
                
                sell_quantity = st.number_input(
                    "Quantity to Sell (you can sell more than you hold to go net short)",
                    min_value=1,
                    value=1,
                    step=1,
                    key="sell_qty"
                )

                action_options = ["Sell/Short more"]
                if side == "short":
                    action_options.append("Buy to cover")
                action = st.radio(
                    "Action",
                    options=action_options,
                    index=0,
                    horizontal=True,
                    key="sell_action",
                )

                price_data = get_data(sell_symbol)
                if price_data['price'] > 0:
                    market_price = float(price_data['price'])
                    st.info(f"Current market price used: ${market_price:.2f}")

                    if action == "Buy to cover":
                        cash_flow = -sell_quantity * market_price  # cash out
                        pnl = (avg_price - market_price) * sell_quantity
                        notional = avg_price * sell_quantity
                        pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                        st.metric("Cash Outlay", f"${-cash_flow:.2f}")
                        st.metric("P&L (per this cover)", f"${pnl:.2f}", delta=f"{pnl_pct:.2f}%")

                        if st.button("✅ Buy to cover", type="primary", key="exec_cover"):
                            result = buy_asset(sell_symbol, sell_quantity, market_price)
                            st.success(f"Bought {sell_quantity} units of {sell_symbol} @ market ${market_price:.2f} to cover short")
                            if result:
                                side_new = result.get('side', 'long').upper()
                                st.info(
                                    f"New position: {result['quantity']} units @ avg ${result['avg_price']:.2f} "
                                    f"({side_new})"
                                )
                            else:
                                st.info("Position fully closed.")
                            time.sleep(1)
                            st.rerun()
                    else:
                        total_proceeds = sell_quantity * market_price
                        if side == 'long':
                            pnl = (market_price - avg_price) * sell_quantity
                        else:
                            pnl = (avg_price - market_price) * sell_quantity
                        notional = avg_price * sell_quantity
                        pnl_pct = (pnl / notional * 100) if notional > 0 else 0.0
                        
                        st.metric("Total Proceeds", f"${total_proceeds:.2f}")
                        st.metric("P&L (per this trade)", f"${pnl:.2f}", delta=f"{pnl_pct:.2f}%")
                        
                        if st.button("✅ Execute Sell / Short", type="primary", key="exec_sell"):
                            if sell_asset(sell_symbol, sell_quantity, market_price):
                                st.success(f"Sold {sell_quantity} units of {sell_symbol} @ market ${market_price:.2f}")
                                st.info(f"Trade P&L (approx.): ${pnl:.2f}")
                                time.sleep(1)
                                st.rerun()
                            else:
                                st.error("Failed to execute sell order")
                else:
                    st.error(f"Could not fetch price for {sell_symbol}")
        else:
            st.info("No assets in portfolio to sell or short")

# Tab 3: Trading Systems
with tab3:
    with st.expander("📘 Comprendre Trading Systems"):
        st.markdown("""
        ### 📋 Ce que vous faites dans Trading Systems
        
        Ici, vous ne passez pas d’ordres immédiats : vous **concevez des systèmes automatiques** (long ou short)
        qui interviendront pour vous à différents niveaux de prix prédéfinis.
        
        Le champ *Symbol* sert à choisir l’actif que vous voulez suivre
        de façon structurée (indice, action, ETF, crypto, etc.).
        
        Le champ *Direction* vous permet de choisir si le système doit exploiter une **hausse** (Long) ou une **baisse** (Short) :
        - **Long** : le robot cherche à accumuler ou renforcer sur l’actif
        - **Short** : le robot cherche à construire ou renforcer une position vendeuse
        
        *Number of Levels* définit combien de paliers d’intervention vous voulez.
        Chaque niveau correspond à un prix où le système déclenchera automatiquement un ordre (dans le sens choisi).
        
        *Drawdown %* contrôle l’écart et la direction des niveaux :
        - **Valeur négative** : niveaux en dessous du prix d’entrée (buy the dip / rachat de short)
        - **Valeur positive** : niveaux au-dessus du prix d’entrée (short plus haut / pyramider sur une tendance haussière)
        
        Plus le pourcentage est faible, plus les niveaux sont serrés; plus il est élevé, plus les niveaux sont espacés.
        
        Quand vous cliquez sur *Add Equity*, l’outil calcule tous les niveaux autour du prix actuel, enregistre le système en mode *Off*,
        puis vous laisse l’activer et le surveiller dans la section *Manage Your Trading Systems* plus bas sur cette page.
        C’est ici que vous transformez une idée en robot.
        
        Utilisez cet onglet pour **planifier à l’avance** comment vous voulez que vos positions se construisent ou se réduisent,
        sans avoir à rester devant les écrans à chaque mouvement de marché.
        """)

    st.subheader("📋 Trading Systems")
    
    # Check current count
    equities = load_equities()
    current_count = len(equities)
    
    st.info(f"Trading Systems: {current_count}/10")
    
    if current_count >= 10:
        st.error("⚠️ Maximum limit reached! You cannot add more than 10 equities. Please remove one first.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            symbol = st.text_input("Symbol", placeholder="e.g., AAPL").upper()
        
        with col2:
            direction = st.radio("Direction", options=["Long", "Short"], index=0, horizontal=True, key="add_equity_direction")
        
        with col3:
            levels = st.number_input("Number of Levels", min_value=1, max_value=10, value=5)
        
        with col4:
            drawdown = st.number_input("Drawdown %", min_value=-50.0, max_value=50.0, value=5.0, step=0.1)
        
        if st.button("➕ Add Equity", type="primary"):
            if symbol:
                equities = load_equities()
                
                # Double check limit
                if len(equities) >= 10:
                    st.error("Maximum limit of 10 equities reached!")
                elif symbol in equities:
                    st.warning(f"{symbol} already exists!")
                else:
                    price_data = get_data(symbol)
                    entry_price = price_data['price']
                    
                    if entry_price > 0:
                        drawdown_decimal = drawdown / 100
                        
                        # Si drawdown négatif: niveaux à la baisse (en dessous du prix d'entrée)
                        # Si drawdown positif: niveaux à la hausse (au dessus du prix d'entrée)
                        if drawdown_decimal < 0:
                            # Drawdown négatif → niveaux en dessous
                            level_prices = {str(i+1): round(entry_price * (1 + drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        else:
                            # Drawdown positif → niveaux au dessus
                            level_prices = {str(i+1): round(entry_price * (1 + drawdown_decimal * (i+1)), 2) for i in range(levels)}
                        
                        stored_drawdown = drawdown_decimal
                        
                        equities[symbol] = {
                            "position": 0,
                            "entry_price": entry_price,
                            "levels": level_prices,
                            "drawdown": stored_drawdown,
                            "direction": direction.lower(),
                            "status": "Off"
                        }
                        
                        save_equities(equities)
                        st.success(f"✅ Added {symbol} ({direction}) at ${entry_price:.2f}")
                        st.balloons()
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"Could not fetch price for {symbol}")
            else:
                st.error("Please enter a symbol")
    
    # Trading Systems Management (moved from separate tab)
    st.markdown("---")
    st.markdown("### 📋 Manage Your Trading Systems")
    st.markdown("""
    Pilotez vos systèmes automatiques déjà configurés : c'est votre salle de contrôle pour voir comment vos robots long/short sont positionnés.
    Le toggle *Active* active/désactive un système. Le tableau *Price Levels* montre tous les niveaux d'intervention.
    """)
    
    equities_list = load_equities()
    
    if equities_list:
        for symbol, data in equities_list.items():
            direction = data.get('direction', 'long')
            with st.expander(f"{symbol} ({direction.upper()}) - Status: {data['status']}", expanded=False):
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    st.metric("Direction", direction.capitalize())
                
                with col2:
                    st.metric("Position", data['position'])
                
                with col3:
                    st.metric("Entry Price", f"${data['entry_price']:.2f}")
                
                with col4:
                    st.metric("Drawdown", f"{data['drawdown']*100:.1f}%")
                
                with col5:
                    current_status = data['status']
                    new_status = st.toggle(
                        "Active", 
                        value=current_status == "On",
                        key=f"toggle_manage_{symbol}"
                    )
                    
                    if (new_status and current_status == "Off") or (not new_status and current_status == "On"):
                        equities_list[symbol]['status'] = "On" if new_status else "Off"
                        save_equities(equities_list)
                        st.rerun()
                
                # Display levels
                st.markdown("**Price Levels:**")
                levels_df = pd.DataFrame([
                    {"Level": k, "Price": f"${v:.2f}"} 
                    for k, v in data['levels'].items()
                ])
                st.dataframe(levels_df, width="stretch", hide_index=True)
                
                # Remove button
                if st.button(f"🗑️ Remove {symbol}", key=f"remove_manage_{symbol}"):
                    del equities_list[symbol]
                    save_equities(equities_list)
                    st.success(f"Removed {symbol}")
                    time.sleep(1)
                    st.rerun()
    else:
        st.info("No trading systems configured yet. Add an equity above to get started.")

# Tab 4: Forwards
with tab4:
    with st.expander("📘 Comprendre Forwards"):
        st.markdown(
            """
            ### 📄 Ce que vous faites dans Forwards
            
            Cet onglet vous permet d'**acheter (ou vendre) un forward** sur un sous-jacent : vous fixez aujourd'hui un prix d'échange futur.
            
            - Vous choisissez le sous-jacent, la date d'échéance, le sens (long/short) et le prix forward.
            - Le forward est enregistré dans un fichier JSON (`forwards.json`).
            - Il apparaît ensuite dans le Dashboard avec P&L mark-to-market basé sur le prix spot actuel.
            
            Un **long forward** gagne lorsque le prix spot est au-dessus du prix forward à l'échéance; un **short forward** gagne dans le cas inverse.
            """
        )

    st.subheader("🚀 Trade Forward")

    col_sym, col_fetch = st.columns([3, 1], vertical_alignment="bottom")
    with col_sym:
        fwd_symbol = st.text_input(
            "Underlying symbol",
            placeholder="e.g., AAPL",
            key="fwd_symbol",
        ).upper()
    with col_fetch:
        if st.button("🔍 Fetch spot", key="btn_fetch_forward_spot"):
            price_data = get_data(fwd_symbol) if fwd_symbol else {"price": 0}
            spot_now_fetch = float(price_data.get("price", 0.0) or 0.0)
            if spot_now_fetch > 0:
                st.session_state["fwd_spot_symbol"] = fwd_symbol
                st.session_state["fwd_spot_value"] = spot_now_fetch
                st.success(f"Spot {fwd_symbol} ≈ ${spot_now_fetch:.4f}")
            else:
                st.warning("Spot introuvable pour ce ticker.")

    today = datetime.date.today()
    default_maturity = today + datetime.timedelta(days=30)
    fwd_maturity = st.date_input(
        "Maturity date",
        value=default_maturity,
        min_value=today,
        key="fwd_maturity",
    )

    fwd_qty = st.number_input(
        "Notional (units)",
        min_value=1,
        value=1,
        step=1,
        key="fwd_qty",
    )

    spot_now = 0.0
    if fwd_symbol and st.session_state.get("fwd_spot_symbol") == fwd_symbol:
        spot_now = float(st.session_state.get("fwd_spot_value", 0.0) or 0.0)
    if spot_now <= 0 and fwd_symbol:
        price_data = get_data(fwd_symbol)
        spot_now = float(price_data.get("price", 0.0) or 0.0)
    if fwd_symbol:
        if spot_now > 0:
            days_to_mat = max((fwd_maturity - today).days, 0)
            T_years = days_to_mat / 365.0
            r_forward = get_r(T_years) if T_years > 0 else get_r(0.1)
            forward_price = spot_now * math.exp(r_forward * T_years)
            pill = (
                f"<div style='display:flex;flex-wrap:wrap;gap:0.6rem;margin-top:0.6rem;'>"
                f"<span style='background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"Spot {fwd_symbol} = {spot_now:.4f}</span>"
                f"<span style='background:#e3f2fd;color:#0d47a1;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"r = {r_forward:.4f}</span>"
                f"<span style='background:#fffde7;color:#f57f17;padding:6px 14px;border-radius:999px;font-weight:600;'>"
                f"T = {T_years:.3f}y</span>"
                f"<span style='background:#e8f5e9;color:#1b5e20;padding:6px 14px;border-radius:999px;font-weight:700;'>"
                f"F ≈ {forward_price:.4f}</span></div>"
            )
            st.markdown(pill, unsafe_allow_html=True)
        else:
            st.warning("Impossible de récupérer le spot, tu peux quand même enregistrer mais le prix forward restera nul.")
            forward_price = 0.0
    else:
        forward_price = 0.0

    fwd_side_label = st.radio(
        "Position",
        options=["Long forward", "Short forward"],
        horizontal=True,
        key="fwd_side",
    )
    fwd_side = "long" if fwd_side_label.startswith("Long") else "short"

    if st.button("Enregistrer le forward", type="primary", key="btn_save_forward"):
        if fwd_symbol and forward_price > 0:
            forwards = load_forwards()
            uid = f"{fwd_symbol}_{fwd_maturity.isoformat()}_{int(time.time())}"
            forwards[uid] = {
                "symbol": fwd_symbol,
                "maturity": fwd_maturity.isoformat(),
                "forward_price": round(float(forward_price), 4),
                "quantity": int(fwd_qty),
                "side": fwd_side,
                "created_at": time.strftime('%Y-%m-%d %H:%M:%S'),
                "spot_at_trade": round(spot_now, 4),
            }
            save_forwards(forwards)
            st.success(f"Forward {fwd_side.upper()} sur {fwd_symbol} enregistré pour {fwd_maturity} à {forward_price:.4f}.")
            time.sleep(1)
            st.rerun()
        else:
            st.error("Renseigne un symbole et assure-toi que le spot est disponible (>0) pour fixer le prix forward.")

    st.markdown("---")
    st.markdown("### 🚀 Forward Portfolio")
    forwards = load_forwards()
    if forwards:
        rows = []
        for key, fwd in forwards.items():
            sym = fwd.get("symbol", "")
            qty = int(fwd.get("quantity", 0) or 0)
            price_fwd = float(fwd.get("forward_price", 0.0) or 0.0)
            side = fwd.get("side", "long")
            maturity_str = fwd.get("maturity")
            try:
                maturity_dt = datetime.date.fromisoformat(maturity_str)
            except Exception:
                maturity_dt = None
            days_to_mat = (maturity_dt - today).days if maturity_dt else None
            spot_now = float(get_data(sym).get("price", 0.0) or 0.0) if sym else 0.0
            mult = 1.0 if side == "long" else -1.0
            pnl_unit = mult * (spot_now - price_fwd)
            pnl_total = pnl_unit * qty
            rows.append({
                "Symbol": sym,
                "Side": side.capitalize(),
                "Quantity": qty,
                "Forward Price": price_fwd,
                "Spot Now": round(spot_now, 4),
                "Maturity": maturity_str,
                "Days to mat": days_to_mat,
                "P&L/unit": round(pnl_unit, 4),
                "P&L total": round(pnl_total, 2),
            })
        if rows:
            df_fwd = pd.DataFrame(rows)
            st.dataframe(df_fwd, width="stretch", hide_index=True)
        if st.button("🧹 Clear forwards", key="clear_forwards"):
            save_forwards({})
            st.success("Forward portfolio cleared.")
            time.sleep(1)
            st.rerun()
    else:
        st.info("Aucun forward pour le moment.")

# Tab 5: app_options (copie intégrale)
with tab5:
    run_app_options()

# Footer
st.markdown("---")
