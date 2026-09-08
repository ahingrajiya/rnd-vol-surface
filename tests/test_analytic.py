import pytest
import numpy as np
from optionslib.core import Market, FlatDiscountCurve, ContinuousDividend, Payoff
from optionslib.models import GBMModel
from optionslib.engines.analytic import black_scholes

@pytest.mark.parametrize("K",[10.0,50.0,100.0,200.0,500.0])
@pytest.mark.parametrize("T",[0.001,0.01,0.1,1.0,10.0,100.0])
def test_put_call_parity(K,T):
    #This test check only if put call parity holds even though sigma handling could be wrong.
    
    discount = FlatDiscountCurve(0.10)
    dividend = ContinuousDividend(0.05)
    market = Market(spot_price=200.,curve =discount,dividend= dividend)
    model = GBMModel(0.25)
    F = market.forward(T=T)
    D = market.discount_factor(T=T)
    
    payoff_put = Payoff(strike = K,call_or_put= -1)
    payoff_call = Payoff(strike=K, call_or_put=1)
    C = black_scholes(payoff_call,model=model,market=market,T=T )
    P = black_scholes(payoff_put,model=model,market=market,T=T )

    assert C.price-P.price == pytest.approx(D*(F-K), rel=1e-13)
    
def test_benchmark():
    #Test to check if volitility is handled correctly
    S = 100
    K = 100
    r = 0.05
    q = 0.0
    sigma = 0.2
    T = 1.0
    
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(spot_price=S,curve =discount,dividend= dividend)
    model = GBMModel(sigma)
    F = market.forward(T=T)
    D = market.discount_factor(T=T)    
    payoff_call = Payoff(strike=K, call_or_put=1)
    C = black_scholes(payoff=payoff_call, model=model,market=market,T=T)
    assert C.price == pytest.approx(10.4506, abs=1e-4)

@pytest.mark.parametrize("put_call",[1,-1])
def test_vega_positivity(put_call):
    S = 100.
    K = 100.
    r = 0.05
    q = 0.10
    T = 1.0
    
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(spot_price=S,curve =discount,dividend= dividend)
    model_1= GBMModel(0.15)
    model_2= GBMModel(0.35)
    payoff = Payoff(strike=K, call_or_put=put_call)
    model_1_price = black_scholes(payoff=payoff, model=model_1,market=market,T=T)
    model_2_price = black_scholes(payoff=payoff, model=model_2,market=market,T=T)
    assert model_1_price.price < model_2_price.price
    
@pytest.mark.parametrize("K",[10.,50.,100.,200.,500.,1000.])
def test_no_arbitrage_bounds(K):
    S = 100.
    r = 0.05
    q = 0.10
    T = 1.0
    
    discount = FlatDiscountCurve(r)
    dividend = ContinuousDividend(q)
    market = Market(spot_price=S,curve =discount,dividend= dividend)
    model= GBMModel(0.15)
    F = market.forward(T=T)
    D = market.discount_factor(T=T)    
    payoff = Payoff(strike=K, call_or_put=1)
    C = black_scholes(payoff=payoff, model=model,market=market,T=T)
    tol = 1e-12
    assert C.price >= max(D*(F-K),0)-tol 
    assert C.price <= D*F + tol
    

def test_expiry_positivity():
    discount = FlatDiscountCurve(0.15)
    dividend = ContinuousDividend(0.15)
    market = Market(spot_price=100.,curve =discount,dividend= dividend)
    model= GBMModel(0.15)
    payoff = Payoff(strike=100., call_or_put=1)
    
    with pytest.raises(ValueError):
        black_scholes(payoff=payoff,model=model,market=market,T=-10.0)

