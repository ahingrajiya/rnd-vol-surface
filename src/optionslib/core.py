from dataclasses import dataclass, field
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class FlatDiscountCurve:
    rate: float

    def __post_init__(self):
        if self.rate < 0:
            """Coding for american options market where historically interest rates have been non negative. 
            This assumption does not work in markets like Japan where interest rates were negative."""
            raise ValueError(f"Interest Rate can not be negative, got {self.rate}")

    def discount_factor(self, T):
        return np.exp(-self.rate * T)


@dataclass(frozen=True)
class ContinuousDividend:
    yield_rate: float

    def __post_init__(self):
        if self.yield_rate < 0:
            raise ValueError(
                f"Dividende yield rate can not be negative, got {self.yield_rate}"
            )

    def growth_factor(self, T):
        return np.exp(-T * self.yield_rate)


@dataclass(frozen=True)
class Market:
    spot_price: float
    curve: FlatDiscountCurve
    dividend: ContinuousDividend

    def __post_init__(self):
        if self.spot_price <= 0:
            raise ValueError(f"Spot Price must be positive, got {self.spot_price}")

    def discount_factor(self, T):
        return self.curve.discount_factor(T)

    def forward(self, T):
        return (
            self.spot_price
            * self.dividend.growth_factor(T)
            / self.curve.discount_factor(T)
        )


@dataclass(frozen=True)
class Payoff:
    strike: float
    call_or_put: Literal[-1, 1]  # takes values +/-1 for call and puts respectively

    def __post_init__(self):
        if self.call_or_put not in [-1, 1]:
            raise ValueError(
                f"Call or put value has to be either +1 or -1, respectively. Got {self.call_or_put}"
            )
        if self.strike <= 0:
            raise ValueError(f"Strike must be positive, got {self.strike}")

    def intrinsic(self, spot):
        return np.maximum(self.call_or_put * (spot - self.strike), 0)


@dataclass(frozen=True)
class PricingResult:
    price: float
    engine: str
    greeks: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
