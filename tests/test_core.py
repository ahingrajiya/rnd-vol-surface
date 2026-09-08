import pytest
from optionslib.core import Market, FlatDiscountCurve, ContinuousDividend, Payoff, PricingResult
import numpy as np

def test_forward_equals_spot_when_rates_zero():
    discount = FlatDiscountCurve(rate=0.)
    dividend = ContinuousDividend(yield_rate=0.)
    market = Market(spot_price=100.0, curve=discount, dividend=dividend)
    assert market.forward(10.0)==pytest.approx(market.spot_price)

def test_forward_when_yield_zero():
    discount = FlatDiscountCurve(rate=0.05)
    dividend = ContinuousDividend(yield_rate=0.0)
    market = Market(spot_price=100.0, curve=discount, dividend=dividend)
    assert market.forward(1.0)==pytest.approx(100.0 * np.exp(0.05 * 1.0))
    
def test_forward_when_rate_zero():
    discount = FlatDiscountCurve(rate=0.00)
    dividend = ContinuousDividend(yield_rate=0.05)
    market = Market(spot_price=100.0, curve=discount, dividend=dividend)
    assert market.forward(1.0)<market.spot_price

def test_negative_spot_price():
    discount = FlatDiscountCurve(rate=0.0)
    dividend = ContinuousDividend(yield_rate=0.0)
    with pytest.raises(ValueError):
        market = Market(spot_price=-100.0, curve=discount, dividend= dividend)

def test_intrinsic_with_call_strike_lower_than_spot():
    strike = 100.0
    spot = 110.0
    payoff = Payoff(strike = strike,call_or_put=1)
    assert payoff.intrinsic(spot) == pytest.approx(spot-strike)
    
def test_intrinsic_with_call_strike_higher_than_spot():
    strike = 100.0
    spot = 90.0
    payoff = Payoff(strike=strike, call_or_put=1)
    assert payoff.intrinsic(spot) == pytest.approx(0.0)

def test_intrinsic_with_put_strike_higher_than_spot():
    strike = 100.0
    spot = 90.0
    payoff = Payoff(strike=strike, call_or_put=-1)
    assert payoff.intrinsic(spot) ==pytest.approx(strike-spot)

def test_intrinsic_works_on_array():
    strike = 200.0
    spot =np.array([150.,200.,50.,30.,500.])
    expected = np.array([50., 0., 150., 170., 0.])
    call_or_put = -1
    payoff = Payoff(strike=strike,call_or_put=call_or_put)
    intrinsic_payoff = payoff.intrinsic(spot=spot)
    np.testing.assert_allclose(intrinsic_payoff, expected)
    
def test_call_or_put_value_check():
    with pytest.raises(ValueError):
        Payoff(100.0,-10) # type: ignore[arg-type]

def test_strike_price_value_check():
    with pytest.raises(ValueError):
        Payoff(-10.0,1)
    
def test_pricing_result_mutable_diagnostics_is_empty():
    price_1 = PricingResult(price=1.0,engine="bs")
    price_2 = PricingResult(price =10.0, engine="sb")
    price_1.diagnostics["something"] = 5
    assert  len(price_2.diagnostics) == 0

def test_pricing_result_mutable_diagnostics_are_different():
    price_1 = PricingResult(price=1.0,engine="bs")
    price_2 = PricingResult(price =10.0, engine="bs")
    # price_1.diagnostics["something"] = 5
    # price_2.diagnostics["something"] =5
    assert price_1.diagnostics is not price_2.diagnostics

