"""
Plain option data structure used by Longstaff-Schwartz pricers.

Responsibilities:
- Hold option contract parameters (strike, maturity, call/put flag).
- Provide vectorized payoff evaluation.

Dependencies:
- NumPy for array-based payoff calculation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Option:
    """
    Vanilla European option definition.

    Attributes:
        s0: Initial underlying level.
        T: Maturity in years.
        K: Strike price.
        v0: Optional initial variance (for stochastic vol models).
        call: True for call, False for put.
    """

    s0: float
    T: int
    K: int
    v0: float = None
    call: bool = True

    def payoff(self, s: np.ndarray) -> np.ndarray:
        """
        Compute intrinsic payoff for a vector of underlying prices.

        Args:
            s: NumPy array of underlying prices at maturity.
        Returns:
            NumPy array of payoffs (elementwise max for call/put).
        """
        payoff = np.maximum(s - self.K, 0) if self.call else np.maximum(self.K - s, 0)
        return payoff
