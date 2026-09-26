import math

import numpy as np
from scipy.special import ndtr

from optionslib.core import Market, Payoff, PricingResult
from optionslib.models import GBMModel


def _d1(forward: float, strike: float, vol: float, T: float) -> float:
    return (np.log(forward / strike) + vol * vol * T / 2.0) / (vol * np.sqrt(T))


def _d2(d1: float, vol: float, T: float) -> float:
    return d1 - vol * np.sqrt(T)


def _vega(D: float, F: float, d1: float, T: float) -> float:
    return D * F * np.sqrt(T) * np.exp(-d1 * d1 / 2.0) / np.sqrt(2.0 * math.pi)


def black_scholes(
    payoff: Payoff, model: GBMModel, market: Market, T: float
) -> PricingResult:
    if T <= 0:
        raise ValueError(f"Expiry must be positive, got {T}")

    vol = model.volatility
    K = payoff.strike
    call_or_put = payoff.call_or_put
    F = market.forward(T)
    D = market.discount_factor(T)

    d1 = _d1(F, K, vol, T)
    d2 = _d2(d1, vol, T)
    vega = _vega(D, F, d1, T)

    price = call_or_put * D * (F * ndtr(call_or_put * d1) - K * ndtr(call_or_put * d2))
    greeks = {"vega": vega}

    return PricingResult(price=price, engine="black_scholes_analytic", greeks=greeks)
