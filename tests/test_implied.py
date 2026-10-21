import math
from itertools import pairwise

import numpy as np
import pytest

from optionslib.core import ContinuousDividend, FlatDiscountCurve, Market, Payoff
from optionslib.engines.analytic import black_scholes
from optionslib.models import GBMModel
from optionslib.vol.implied import (
    DEFAULT_CONFIG,
    Method,
    NormalizedProblem,
    Reason,
    bracket_check,
    brent_method,
    implied_vol,
    initial_guess,
    newton_method,
    normalized_bounds,
    normalized_price,
    normalized_vega,
)


@pytest.mark.parametrize(
    "x, call_or_put",
    [
        (-1.0, 1),
        (1.0, -1),
        (-0.5, 1),
        (0.5, -1),
        (-0.05, 1),
        (0.05, -1),
        (-0.005, 1),
        (0.005, -1),
    ],
)
def test_otm_floor_is_zero(x, call_or_put):
    bounds = normalized_bounds(x, call_or_put)
    assert bounds.floor == 0


@pytest.mark.parametrize("x,call_or_put", [(0.0, 1), (0.0, -1)])
def test_atm_bounds(x, call_or_put):
    bounds = normalized_bounds(x, call_or_put)
    assert bounds.floor == 0
    assert bounds.ceiling == 1.0


@pytest.mark.parametrize(
    "x, call_or_put",
    [
        (1.0, 1),
        (-1.0, -1),
        (10.0, 1),
        (-10.0, -1),
        (5.0, 1),
        (-5.0, -1),
        (0.005, 1),
        (-0.005, -1),
    ],
)
def test_itm_floor_positive(x, call_or_put):
    bounds = normalized_bounds(x, call_or_put)
    if call_or_put == 1:
        expected = max(math.exp(x / 2) - math.exp(-x / 2), 0.0)
    else:
        expected = max(math.exp(-x / 2) - math.exp(x / 2), 0.0)
    assert bounds.floor == pytest.approx(expected)


@pytest.mark.parametrize(
    "F, K, call_or_put",
    [(100.0, 120.0, 1), (100.0, 80.0, 1), (100.0, 120.0, -1), (100.0, 80.0, -1)],
)
def test_ceiling(F, K, call_or_put):
    x = math.log(F / K)
    bounds = normalized_bounds(x, call_or_put)
    expected = (F if call_or_put == 1 else K) / math.sqrt(F * K)
    assert bounds.ceiling == pytest.approx(expected)


@pytest.mark.parametrize("x", [0.0, -1.0, 1.0, 0.05, -0.05, 1.5, -1.5])
def test_symmetry(x):
    assert normalized_bounds(x, -1) == normalized_bounds(-x, 1)


@pytest.mark.parametrize(
    "S,K,r,q,T,call_or_put",
    [
        (100.0, 150.0, 0.0, 0.0, 1.0, 1),
        (100.0, 150.0, 0.0, 0.0, 1.0, -1),
        (100.0, 50.0, 0.0, 0.0, 1.0, 1),
        (100.0, 50.0, 0.0, 0.0, 1.0, -1),
        (100.0, 150.0, 0.1, 0.05, 1.0, 1),
        (100.0, 150.0, 0.1, 0.05, 1.0, -1),
    ],
)
def test_small_sigma_approaches_floor(S, K, r, q, T, call_or_put):
    sigma = 1e-8
    payoff = Payoff(K, call_or_put)
    model = GBMModel(sigma)
    discout = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, discout, dividend)
    price = black_scholes(payoff, model, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)
    x = math.log(F / K)
    b = price.price / (D * math.sqrt(F * K))

    assert normalized_bounds(x, call_or_put).floor == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize(
    "S,K,r,q,T,call_or_put",
    [
        (100.0, 150.0, 0.0, 0.0, 1.0, 1),
        (100.0, 150.0, 0.0, 0.0, 1.0, -1),
        (100.0, 50.0, 0.0, 0.0, 1.0, 1),
        (100.0, 50.0, 0.0, 0.0, 1.0, -1),
        (100.0, 150.0, 0.1, 0.05, 1.0, 1),
        (100.0, 150.0, 0.1, 0.05, 1.0, -1),
    ],
)
def test_large_sigma_approaches_ceiling(S, K, r, q, T, call_or_put):
    sigma = 100
    payoff = Payoff(K, call_or_put)
    model = GBMModel(sigma)
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, discount, dividend)
    price = black_scholes(payoff, model, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)
    x = math.log(F / K)
    b = price.price / (D * math.sqrt(F * K))

    assert normalized_bounds(x, call_or_put).ceiling == pytest.approx(b, abs=1e-9)


@pytest.mark.parametrize("s", [0.01, 0.1, 1.0])
@pytest.mark.parametrize("x", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
def test_normalized_price_symmetry(
    s,
    x,
):
    assert normalized_price(x=x, call_or_put=-1, s=s) == pytest.approx(
        normalized_price(x=-x, call_or_put=1, s=s), abs=1e-9
    )


@pytest.mark.parametrize(
    "S,K,r,q,T,call_or_put",
    [
        (100.0, 150.0, 0.0, 0.0, 1.0, 1),
        (100.0, 150.0, 0.0, 0.0, 1.0, -1),
        (100.0, 50.0, 0.0, 0.0, 1.0, 1),
        (100.0, 50.0, 0.0, 0.0, 1.0, -1),
        (100.0, 150.0, 0.1, 0.05, 1.0, 1),
        (100.0, 150.0, 0.1, 0.05, 1.0, -1),
    ],
)
@pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1.0, 2.5])
def test_normalized_price_agreement_with_black_scholes(
    S, K, r, q, T, call_or_put, sigma
):
    payoff = Payoff(K, call_or_put)
    model = GBMModel(sigma)
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, discount, dividend)
    price = black_scholes(payoff, model, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)
    x = math.log(F / K)
    s = sigma * math.sqrt(T)

    assert normalized_price(x, call_or_put, s=s) == pytest.approx(
        price.price / (D * math.sqrt(F * K)), rel=1e-13
    )


XT_CASES = [(-0.5, 1), (0.5, 1), (-0.5, -1), (0.5, -1), (0.0, 1), (0.0, -1)]


@pytest.mark.parametrize("x,call_or_put", XT_CASES)
def test_normalized_price_approach_floor_at_small_s(x, call_or_put):
    s = 1e-8
    assert normalized_price(x, call_or_put, s) == pytest.approx(
        normalized_bounds(x, call_or_put).floor, abs=1e-8
    )


@pytest.mark.parametrize("x,call_or_put", XT_CASES)
def test_normalized_price_approach_ceiling_at_large_s(x, call_or_put):
    s = 100
    assert normalized_price(x, call_or_put, s) == pytest.approx(
        normalized_bounds(x, call_or_put).ceiling, rel=1e-13
    )


S_SEQUENCE = (0.01, 0.1, 1.0, 1.5, 2.0, 2.5, 5.0, 10.0)


@pytest.mark.parametrize("x,call_or_put", XT_CASES)
def test_normalized_price_monotonicity(x, call_or_put):
    for lo, hi in pairwise(S_SEQUENCE):
        print(lo, hi)
        assert normalized_price(x, call_or_put, lo) < normalized_price(
            x, call_or_put, hi
        )


@pytest.mark.parametrize(
    "S,K,r,q,T,call_or_put",
    [
        (100.0, 150.0, 0.0, 0.0, 1.0, 1),
        (100.0, 150.0, 0.0, 0.0, 1.5, -1),
        (100.0, 50.0, 0.0, 0.0, 2.0, 1),
        (100.0, 50.0, 0.0, 0.0, 3.0, -1),
        (100.0, 150.0, 0.1, 0.05, 10.0, 1),
        (100.0, 150.0, 0.1, 0.05, 1.0, -1),
    ],
)
@pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1.0, 2.5])
def test_vega_agreement_with_black_scholes(S, K, r, q, T, call_or_put, sigma):
    payoff = Payoff(K, call_or_put)
    model = GBMModel(sigma)
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, discount, dividend)
    result = black_scholes(payoff, model, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)
    x = math.log(F / K)
    s = sigma * math.sqrt(T)

    assert normalized_vega(x, call_or_put, s) == pytest.approx(
        result.greeks["vega"] / (D * math.sqrt(F * K) * math.sqrt(T)), rel=1e-13
    )


@pytest.mark.parametrize("x", [-10.0, -1.0, -0.1, 0.001, 0.01, 0.1, 1.0, 2.5])
@pytest.mark.parametrize("call_or_put", [+1, -1])
def test_vega_peaks_at_critical_value(x, call_or_put):
    s_c = math.sqrt(2 * math.fabs(x))
    epsilon = 1e-7
    assert normalized_vega(x, call_or_put, s_c) > normalized_vega(
        x, call_or_put, s_c - epsilon
    )
    assert normalized_vega(x, call_or_put, s_c) > normalized_vega(
        x, call_or_put, s_c + epsilon
    )


@pytest.mark.parametrize(
    "S,K,r,q,T,call_or_put",
    [
        (100.0, 150.0, 0.0, 0.0, 1.0, 1),
        (100.0, 150.0, 0.0, 0.0, 1.5, -1),
        (100.0, 50.0, 0.0, 0.0, 2.0, 1),
        (100.0, 50.0, 0.0, 0.0, 3.0, -1),
        (100.0, 150.0, 0.1, 0.05, 10.0, 1),
        (100.0, 150.0, 0.1, 0.05, 1.0, -1),
    ],
)
@pytest.mark.parametrize("sigma", [0.001, 0.01, 0.1, 1.0, 2.5])
def test_vega_finite_difference_consistency(S, K, r, q, T, call_or_put, sigma):
    """
    This test compares closed form vega against engine price. Price is validated independently. Hence, this test will catch derivation errors, if any.

    Vega is partial(V)/partial(sigma). So central difference of black-scholes prices approximates vega, (V(sigma+h) - V(sigma-h))/(2h). This is why comparison
    tolerance is calculated for each case. From BS F*N(theta*d1) and K*N(theta*d2) each carry absolute error ~epsilon*D*max(F,K). Difference preserves the
    absolute error. This is catastrophic cancellation. The relative error of the result can be arbitrarily bad because the difference is small while the error
    is not. Differencing two prices and dividing by 2h gives epsilon*D*max(F,K)/h. The slack factor of 10 is empirical. The effect is worst deep ITM/OTM,
    where vega is orders of magnitude smaller than the price.

    """
    h = 1e-6 * sigma
    payoff = Payoff(K, call_or_put)
    model_lo = GBMModel(sigma - h)
    model_hi = GBMModel(sigma + h)
    model_mid = GBMModel(sigma)
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, discount, dividend)
    result_lo = black_scholes(payoff, model_lo, market, T)
    result_hi = black_scholes(payoff, model_hi, market, T)
    result_mid = black_scholes(payoff, model_mid, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)

    eps = np.finfo(float).eps
    price_scale = D * max(F, K)
    expected_error = eps * price_scale / (2.0 * h)

    assert result_mid.greeks["vega"] == pytest.approx(
        (result_hi.price - result_lo.price) / (2.0 * h), abs=10 * expected_error
    )


@pytest.mark.parametrize(
    "S,K,r,q,call_or_put",
    [
        (100.0, 100.0, 0.0, 0.0, 1),
        (100.0, 100.0, 0.0, 0.0, -1),
        (100.0, 120.0, 0.0, 0.0, 1),
        (100.0, 80.0, 0.0, 0.0, -1),
        (100.0, 120.0, 0.05, 0.02, 1),
        (100.0, 80.0, 0.05, 0.02, -1),
    ],
)
@pytest.mark.parametrize(
    "sigma,T",
    [
        (1.0, 0.01),
        (0.3, 0.05),
        (0.1, 1.0),
        (1.0, 1.0),
        (4.90, 1.0),
        (2.5, 1.0),
        (0.1, 0.50),
        (0.1, 10.0),
        (1.0, 2.0),
        (3.0, 1.5),
        (2.5, 2.0),
    ],
)
def test_implied_vol_round_trip(S, K, r, q, call_or_put, sigma, T):
    model = GBMModel(sigma)
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    result = black_scholes(payoff, model, market, T)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)
    iv_result = implied_vol(norm_problem, result.price)
    tol_sigma = DEFAULT_CONFIG.tolerance / math.sqrt(T)

    assert iv_result.reason == Reason.OK
    assert iv_result.sigma == pytest.approx(sigma, abs=tol_sigma)
    assert iv_result.s == pytest.approx(
        sigma * math.sqrt(T), abs=DEFAULT_CONFIG.tolerance
    )


@pytest.mark.parametrize(
    "S,K,r,q,call_or_put,T",
    [
        (100.0, 80.0, 0.0, 0.0, 1, 1.0),
        (100.0, 120.0, 0.0, 0.0, -1, 1.0),
        (100.0, 80.0, 0.05, 0.02, 1, 1.0),
        (100.0, 120.0, 0.05, 0.02, -1, 1.0),
        (100.0, 90.0, 0.0, 0.0, 1, 0.25),
        (100.0, 110.0, 0.0, 0.0, -1, 2.0),
    ],
)
def test_below_floor(S, K, r, q, call_or_put, T):
    """
    Price is constructed by hand as Black Scholes can not construct price outside arbitrage bracket
    """
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)
    floor = D * max(F - K, 0.0) if call_or_put == 1 else D * max(K - F, 0.0)
    bad_price = floor - 1.0
    iv_result = implied_vol(norm_problem, bad_price)

    assert iv_result.iterations == 0
    assert iv_result.reason == Reason.BELOW_FLOOR
    assert math.isnan(iv_result.sigma)
    assert math.isnan(iv_result.s)
    assert iv_result.method == Method.NONE


@pytest.mark.parametrize(
    "S,K,r,q,call_or_put,T",
    [
        (100.0, 100.0, 0.0, 0.0, 1, 1.0),
        (100.0, 100.0, 0.0, 0.0, -1, 1.0),
        (100.0, 80.0, 0.0, 0.0, 1, 1.0),
        (100.0, 120.0, 0.0, 0.0, -1, 1.0),
        (100.0, 120.0, 0.0, 0.0, 1, 1.0),
        (100.0, 80.0, 0.0, 0.0, -1, 1.0),
        (100.0, 80.0, 0.05, 0.02, 1, 1.0),
        (100.0, 120.0, 0.05, 0.02, -1, 1.0),
        (100.0, 100.0, 0.05, 0.0, 1, 2.0),
    ],
)
def test_above_ceiling(S, K, r, q, call_or_put, T):
    """
    Price is constructed by hand as Black Scholes can not construct price outside arbitrage bracket
    """
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)
    D = market.discount_factor(T)
    F = market.forward(T)
    ceiling = D * F if call_or_put == 1 else D * K
    bad_price = ceiling + 1.0
    iv_result = implied_vol(norm_problem, bad_price)

    assert iv_result.iterations == 0
    assert iv_result.reason == Reason.ABOVE_CEILING
    assert math.isnan(iv_result.sigma)
    assert math.isnan(iv_result.s)
    assert iv_result.method == Method.NONE


@pytest.mark.parametrize(
    "S,K,r,q,call_or_put,sigma,T",
    [
        (100.0, 100.0, 0.0, 0.0, 1, 0.1, 0.05),
        (100.0, 100.0, 0.0, 0.0, -1, 0.1, 0.05),
        (100.0, 100.0, 0.0, 0.0, 1, 0.01, 1.0),
        (100.0, 100.0, 0.0, 0.0, -1, 0.01, 1.0),
        (100.0, 100.0, 0.05, 0.02, 1, 0.02, 1.0),
        (100.0, 100.0, 0.05, 0.02, -1, 0.02, 1.0),
    ],
)
def test_root_below_bracket(S, K, r, q, call_or_put, sigma, T):
    """
    Prices are constructed using Black Scholes but implied volatility is well outside the range of [0.05,5.0]
    """
    model = GBMModel(sigma)
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    result = black_scholes(payoff, model, market, T)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)
    iv_result = implied_vol(norm_problem, result.price)

    assert iv_result.reason == Reason.ROOT_BELOW_BRACKET
    assert math.isnan(iv_result.sigma)
    assert math.isnan(iv_result.s)
    assert iv_result.method == Method.NONE
    assert iv_result.iterations == 0


@pytest.mark.parametrize(
    "S,K,r,q,call_or_put,sigma,T",
    [
        (100.0, 100.0, 0.0, 0.0, 1, 8.0, 1.0),
        (100.0, 100.0, 0.0, 0.0, -1, 8.0, 1.0),
        (100.0, 120.0, 0.0, 0.0, 1, 6.0, 1.0),
        (100.0, 80.0, 0.0, 0.0, -1, 6.0, 1.0),
        (100.0, 100.0, 0.05, 0.02, 1, 4.0, 2.25),
        (100.0, 100.0, 0.05, 0.02, -1, 4.0, 2.25),
    ],
)
def test_root_above_bracket(S, K, r, q, call_or_put, sigma, T):
    """
    Prices are constructed using Black Scholes but implied volatility is well outside the range of [0.05,5.0]
    """
    model = GBMModel(sigma)
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    result = black_scholes(payoff, model, market, T)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)
    iv_result = implied_vol(norm_problem, result.price)

    assert iv_result.reason == Reason.ROOT_ABOVE_BRACKET
    assert math.isnan(iv_result.sigma)
    assert math.isnan(iv_result.s)
    assert iv_result.method == Method.NONE
    assert iv_result.iterations == 0


@pytest.mark.parametrize(
    "S,K,r,q,call_or_put",
    [
        (100.0, 100.0, 0.0, 0.0, 1),
        (100.0, 100.0, 0.0, 0.0, -1),
        (100.0, 120.0, 0.0, 0.0, 1),
        (100.0, 80.0, 0.0, 0.0, -1),
        (100.0, 120.0, 0.05, 0.02, 1),
        (100.0, 80.0, 0.05, 0.02, -1),
    ],
)
@pytest.mark.parametrize(
    "sigma,T",
    [
        (1.0, 0.01),
        (0.3, 0.05),
        (0.1, 1.0),
        (1.0, 1.0),
        (4.90, 1.0),
        (2.5, 1.0),
        (0.1, 0.50),
        (0.1, 10.0),
        (1.0, 2.0),
        (3.0, 1.5),
        (2.5, 2.0),
    ],
)
def test_newton_brent_agreement(S, K, r, q, call_or_put, sigma, T):
    model = GBMModel(sigma)
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    result = black_scholes(payoff, model, market, T)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)

    b_target = result.price / norm_problem.normalization_factor
    brackets = bracket_check(
        DEFAULT_CONFIG, norm_problem.x, norm_problem.call_or_put, b_target
    )
    s_initial = initial_guess(
        DEFAULT_CONFIG, norm_problem.x, norm_problem.call_or_put, b_target
    )
    newton = newton_method(
        DEFAULT_CONFIG, norm_problem.x, norm_problem.call_or_put, b_target, s_initial
    )
    brent = brent_method(
        DEFAULT_CONFIG,
        norm_problem.x,
        norm_problem.call_or_put,
        b_target,
        brackets.b_lo,
        brackets.b_hi,
    )

    print(
        f"Newton Exit: {newton.exit_reason}, Initial Guess: {s_initial}, Newton Iterations: {newton.iterations}"
    )
    if newton.exit_reason == brent.exit_reason:
        assert newton.s == pytest.approx(brent.s, abs=2e-8)


@pytest.mark.parametrize(
    "sigma,T,S,K,r,q,call_or_put",
    [
        (1.0, 0.01, 100.0, 120.0, 0.0, 0.0, 1),
        (1.0, 0.01, 100.0, 80.0, 0.0, 0.0, -1),
        (1.0, 0.01, 100.0, 120.0, 0.05, 0.02, 1),
        (1.0, 0.01, 100.0, 80.0, 0.05, 0.02, -1),
        (0.30, 0.05, 100.0, 120.0, 0.0, 0.0, 1),
        (0.30, 0.05, 100.0, 80.0, 0.0, 0.0, -1),
        (0.30, 0.05, 100.0, 80.0, 0.05, 0.01, -1),
        (0.10, 1.0, 100.0, 120.0, 0.0, 0.0, 1),
        (0.10, 1.0, 100.0, 80.0, 0.0, 0.0, -1),
        (0.10, 1.0, 100.0, 80.0, 0.05, 0.02, -1),
        (0.10, 0.50, 100.0, 120.0, 0.0, 0.0, 1),
        (0.10, 0.50, 100.0, 80.0, 0.0, 0.0, -1),
        (0.10, 0.50, 100.0, 80.0, 0.05, 0.02, -1),
    ],
)
def test_newton_fails_brent_recovers(S, K, r, q, call_or_put, sigma, T):
    model = GBMModel(sigma)
    payoff = Payoff(K, call_or_put)
    curve = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(S, curve, dividend)
    result = black_scholes(payoff, model, market, T)
    norm_problem = NormalizedProblem.from_market(payoff, market, T)
    iv_result = implied_vol(norm_problem, result.price)
    tol = DEFAULT_CONFIG.tolerance / math.sqrt(T)

    assert iv_result.method == Method.BRENT
    assert iv_result.reason == Reason.OK
    assert iv_result.sigma == pytest.approx(sigma, abs=tol)
