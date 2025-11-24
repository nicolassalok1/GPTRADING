"""
Stochastic process definitions and simulators used by Longstaff-Schwartz pricing.

Responsibilities:
- Provide base abstract process contract.
- Implement GBM and Heston processes with vectorized path generation.

External dependencies:
- NumPy for random sampling and vectorized math.
- Pandas for returning path matrices in DataFrame form.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd


class StochasticProcess(ABC):
    """Abstract stochastic process interface; subclasses must implement simulate()."""

    @abstractmethod
    def simulate(self):
        ...


@dataclass
class GeometricBrownianMotion(StochasticProcess):
    """
    Geometric Brownian Motion with closed-form path generation.

    Attributes:
        mu: Drift term (annualized).
        sigma: Volatility term.

    Notes:
        Paths are generated in a fully vectorized way for performance.
    """

    mu: float
    sigma: float

    def simulate(
        self, s0: float, T: int, n: int, m: int, v0: float = None
    ) -> pd.DataFrame:  # n = number of paths, m = number of discretization points
        """
        Generate GBM paths using exact discretization.

        Args:
            s0: Initial spot.
            T: Horizon in years.
            n: Number of simulated paths.
            m: Number of time steps (discretization points).
            v0: Unused, kept for interface compatibility.
        Returns:
            DataFrame of shape (m+1, n) with simulated spot levels.
        """
        dt = T / m
        np.random.seed(0)
        W = np.cumsum(np.sqrt(dt) * np.random.randn(m + 1, n), axis=0)
        W[0] = 0

        T = np.ones(n).reshape(1, -1) * np.linspace(0, T, m + 1).reshape(-1, 1)

        s = s0 * np.exp((self.mu - 0.5 * self.sigma**2) * T + self.sigma * W)

        return s


@dataclass
class HestonProcess(StochasticProcess):
    """
    Heston stochastic volatility process simulated via Milstein scheme.

    Attributes:
        mu: Drift term on price.
        kappa: Mean reversion speed of variance.
        theta: Long-term variance level.
        eta: Volatility of volatility.
        rho: Correlation between price and variance Brownian motions.
    """

    mu: float
    kappa: float
    theta: float
    eta: float
    rho: float

    def simulate(
        self, s0: float, v0: float, T: int, n: int, m: int
    ) -> pd.DataFrame:  # n = number of paths, m = number of discretization points
        """
        Simulate Heston paths with Milstein correction for variance.

        Args:
            s0: Initial spot.
            v0: Initial variance.
            T: Horizon in years.
            n: Number of paths.
            m: Number of time steps.
        Returns:
            DataFrame of shape (m+1, n) with simulated spot levels.
        Notes:
            Variance is floored to stay non-negative; correlation is applied via
            Cholesky-equivalent construction.
        """
        dt = T / m
        z1 = np.random.randn(m, n)
        z2 = self.rho * z1 + np.sqrt(1 - self.rho**2) * np.random.randn(m, n)

        s = np.zeros((m + 1, n))
        x = np.zeros((m + 1, n))
        v = np.zeros((m + 1, n))

        s[0] = s0
        v[0] = v0

        for i in range(m):

            v[i + 1] = (
                v[i]
                + self.kappa * (self.theta - v[i]) * dt
                + self.eta * np.sqrt(v[i] * dt) * z1[i]
                + self.eta**2 / 4 * (z1[i] ** 2 - 1) * dt
            )
            v = np.where(v > 0, v, -v)

            x[i + 1] = x[i] + (self.mu - v[i] / 2) * dt + np.sqrt(v[i] * dt) * z2[i]

            s[1:] = s[0] * np.exp(x[1:])

        return s
