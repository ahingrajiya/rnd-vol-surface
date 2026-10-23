"""
Tests for the options chain data layer.

Vendor data is not redistributable, so every test builds its own chain in the vendor's file format,
priced by black_scholes. The analytic engine is therefore the independent reference here: a chain
priced at known (r, q, sigma) must come back through the parity regression and the solver with those
same values.
"""

import math

import numpy as np
import pandas as pd
import pytest

from optionslib.chain import (
    ForwardConfig,
    ForwardStatus,
    QuoteStatus,
    bracket_report,
    classify_quotes,
    compare_to_vendor,
    fit_forward,
    invert_chain,
    put_call_agreement,
    read_options_csv,
    time_to_expiry,
)
from optionslib.core import ContinuousDividend, FlatDiscountCurve, Market, Payoff
from optionslib.engines.analytic import black_scholes
from optionslib.models import GBMModel
from optionslib.vol.implied import Reason

QUOTE_DATE = "2022-07-01"  # a Friday
SPOT = 3800.0
RATE = 0.03
DIV = 0.015
SIGMA = 0.20


def _bs_price(K, cp, T, S=SPOT, r=RATE, q=DIV, sigma=SIGMA):
    market = Market(S, FlatDiscountCurve(r), ContinuousDividend(q))
    return float(black_scholes(Payoff(K, cp), GBMModel(sigma), market, T).price)


def make_chain(
    buckets, strikes, half_spread=0.05, style="E", S=SPOT, sigma=SIGMA, vendor_iv=True
):
    """
    Vendor-format rows for every (underlying, expiration, settlement) bucket x strike x call/put.

    Prices are exact Black-Scholes values, quoted symmetrically at mid +/- half_spread, so the mid is
    the model price and every recovered quantity has a known true value. A price below the
    half-spread is quoted one-sided (no bid), as real wing quotes are.
    """
    rows = []
    for underlying, expiration, settlement in buckets:
        T = float(time_to_expiry([QUOTE_DATE], [expiration], [settlement])[0])
        for K in strikes:
            for cp, typ in ((1, "call"), (-1, "put")):
                price = (
                    _bs_price(K, cp, T, S=S, sigma=sigma)
                    if T > 0
                    else max(cp * (S - K), 0.0)
                )
                bid = price - half_spread
                rows.append(
                    {
                        "contract": f"{underlying}{expiration.replace('-', '')[2:]}{typ[0].upper()}{int(K * 1000):08d}",
                        "underlying": underlying,
                        "expiration": expiration,
                        "type": typ,
                        "strike": K,
                        "style": style,
                        "quote_date": QUOTE_DATE,
                        "bid": bid if bid > 0 else np.nan,
                        "ask": price + half_spread,
                        "underlying_close": S,
                        "settlement_time": settlement,
                        "volume": 10,
                        "iv": sigma if vendor_iv else np.nan,
                        "iv_flag": 0,
                    }
                )
    return pd.DataFrame(rows)


def write_chain(tmp_path, frame, name=f"{QUOTE_DATE}_options.csv"):
    path = tmp_path / name
    frame.to_csv(path, index=False)
    return path


STRIKES = [float(k) for k in range(3200, 4401, 50)]
TWO_BUCKETS = [("SPX", "2022-08-19", "AM"), ("SPXW", "2022-08-31", "PM")]


# ------------------------------------------------------------------------------------- time to expiry


def test_pm_contract_on_expiration_day_has_no_time_left():
    assert time_to_expiry([QUOTE_DATE], [QUOTE_DATE], ["PM"])[0] == 0.0


def test_pm_contract_expiring_next_day_is_one_calendar_day():
    assert time_to_expiry(["2022-07-05"], ["2022-07-06"], ["PM"])[0] == pytest.approx(
        1 / 365, abs=1e-15
    )


def test_am_contract_ends_six_and_a_half_hours_before_pm():
    am, pm = time_to_expiry([QUOTE_DATE] * 2, ["2022-07-15"] * 2, ["AM", "PM"])
    assert pm - am == pytest.approx(6.5 / 24 / 365, abs=1e-15)


def test_prior_close_convention_ends_am_contract_at_previous_weekday_close():
    """Friday AM expiry under the vendor convention ends Thursday 16:00: 13 whole days from Friday 1 July."""
    T = time_to_expiry([QUOTE_DATE], ["2022-07-15"], ["AM"], am_expiry="prior_close")[0]
    assert T == pytest.approx(13 / 365, abs=1e-15)


def test_prior_close_convention_leaves_pm_contracts_alone():
    a = time_to_expiry([QUOTE_DATE], ["2022-07-15"], ["PM"], am_expiry="open")[0]
    b = time_to_expiry([QUOTE_DATE], ["2022-07-15"], ["PM"], am_expiry="prior_close")[0]
    assert a == b


def test_daylight_saving_end_adds_an_hour():
    """Clocks go back on 2022-11-06, so Friday 16:00 to Monday 16:00 is 73 hours, not 72."""
    T = time_to_expiry(["2022-11-04"], ["2022-11-07"], ["PM"])[0]
    assert T == pytest.approx(73 / 24 / 365, abs=1e-15)


def test_unknown_settlement_time_raises():
    with pytest.raises(ValueError, match="settlement_time"):
        time_to_expiry([QUOTE_DATE], ["2022-07-15"], ["NOON"])


# --------------------------------------------------------------------------------------------- reader


def test_reader_keeps_only_requested_underlyings_and_adds_columns(tmp_path):
    frame = pd.concat(
        [
            make_chain(TWO_BUCKETS, STRIKES),
            make_chain([("NDX", "2022-08-19", "AM")], STRIKES),
        ]
    )
    chain = read_options_csv(write_chain(tmp_path, frame), underlyings=["SPX", "SPXW"])
    assert set(chain["underlying"]) == {"SPX", "SPXW"}
    assert set(chain["call_or_put"]) == {1, -1}
    assert (chain.loc[chain["type"] == "call", "call_or_put"] == 1).all()
    assert "vendor_iv" in chain and "iv" not in chain, (
        "vendor IV must be renamed so it cannot pass as ours"
    )
    assert (chain["T"] > 0).all()


def test_reader_rejects_american_style(tmp_path):
    frame = make_chain([("SPY", "2022-08-19", "PM")], STRIKES, style="A")
    with pytest.raises(ValueError, match="not European-style"):
        read_options_csv(write_chain(tmp_path, frame), underlyings=["SPY"])


def test_reader_rejects_bucket_mixing_am_and_pm(tmp_path):
    frame = make_chain([("SPX", "2022-08-19", "AM")], STRIKES)
    frame.loc[frame.index[:4], "settlement_time"] = "PM"
    with pytest.raises(ValueError, match="mix AM and PM"):
        read_options_csv(write_chain(tmp_path, frame), underlyings=["SPX"])


def test_reader_rejects_absent_underlying(tmp_path):
    frame = make_chain(TWO_BUCKETS, STRIKES)
    with pytest.raises(ValueError, match="no rows for underlyings"):
        read_options_csv(write_chain(tmp_path, frame), underlyings=["SPX", "RUT"])


def test_reader_rejects_file_without_required_columns(tmp_path):
    frame = make_chain(TWO_BUCKETS, STRIKES).drop(
        columns=["settlement_time", "underlying_close"]
    )
    with pytest.raises(ValueError, match="lacks required columns"):
        read_options_csv(write_chain(tmp_path, frame))


def test_reader_rejects_two_quote_dates(tmp_path):
    frame = make_chain(TWO_BUCKETS, STRIKES)
    frame.loc[frame.index[0], "quote_date"] = "2022-07-05"
    with pytest.raises(ValueError, match="one quote_date"):
        read_options_csv(write_chain(tmp_path, frame))


def test_reader_chunked_read_matches_single_read(tmp_path):
    path = write_chain(tmp_path, make_chain(TWO_BUCKETS, STRIKES))
    whole = read_options_csv(path, chunksize=10_000)
    chunked = read_options_csv(path, chunksize=7)
    pd.testing.assert_frame_equal(whole, chunked)


# ------------------------------------------------------------------------------------- quote classes


def test_classify_quotes_assigns_each_status_once():
    frame = pd.DataFrame(
        {
            "bid": [1.0, np.nan, 0.0, 2.0, 1.0, np.nan],
            "ask": [1.2, 1.0, 0.5, 1.5, 1.2, 1.0],
            "T": [0.1, 0.1, 0.1, 0.1, 0.0, 0.0],
        }
    )
    out = classify_quotes(frame)
    assert (
        list(out["quote_status"])
        == [
            QuoteStatus.OK,
            QuoteStatus.ONE_SIDED,
            QuoteStatus.ONE_SIDED,
            QuoteStatus.CROSSED,
            QuoteStatus.EXPIRED,
            QuoteStatus.EXPIRED,  # expired and one-sided: no time left is the more fundamental reason
        ]
    )
    assert out["mid"].iloc[0] == pytest.approx(1.1)
    assert out["half_spread"].iloc[0] == pytest.approx(0.1)
    assert np.isnan(out["mid"].iloc[1])


# ------------------------------------------------------------------------------------ forward fitting


def _parity_inputs(D, F, K, half_spread=0.05):
    K = np.asarray(K, dtype=float)
    c_minus_p = D * (F - K)
    h = np.full_like(K, half_spread)
    return K, c_minus_p, np.zeros_like(K), h, h  # put mid 0 keeps C - P exact


def test_fit_forward_recovers_exact_line():
    D, F = 0.9876, 3812.5
    fit = fit_forward(*_parity_inputs(D, F, STRIKES), spot=SPOT)
    assert fit.status == ForwardStatus.OK
    assert fit.D == pytest.approx(D, abs=1e-12)
    assert fit.F == pytest.approx(F, abs=1e-9)
    assert fit.n_dropped == 0


@pytest.mark.parametrize("T", [7 / 365, 0.25, 1.0, 3.0])
def test_fit_forward_recovers_black_scholes_carry(T):
    """Independent route: D and F implied by black_scholes call and put prices at known r and q."""
    K = np.array(STRIKES)
    calls = np.array([_bs_price(k, 1, T) for k in K])
    puts = np.array([_bs_price(k, -1, T) for k in K])
    h = np.full_like(K, 0.05)
    fit = fit_forward(K, calls, puts, h, h, spot=SPOT)
    assert fit.D == pytest.approx(math.exp(-RATE * T), abs=1e-11)
    assert fit.F == pytest.approx(SPOT * math.exp((RATE - DIV) * T), abs=1e-7)


def test_fit_forward_down_weights_wide_quotes():
    """An error inside a wide quote's spread moves the weighted fit far less than an unweighted one."""
    D, F = 0.99, 3820.0
    K, y, p, hc, hp = _parity_inputs(D, F, STRIKES)
    wide = len(K) - 1
    hc = hc.copy()
    hc[wide] = 20.0  # a deep wing with a $40-wide quote
    y = y.copy()
    y[wide] += 0.9 * math.sqrt(
        hc[wide] ** 2 + hp[wide] ** 2
    )  # off by 0.9 of its own spread: kept
    weighted = fit_forward(K, y, p, hc, hp, spot=SPOT)
    equal = fit_forward(
        K,
        y,
        p,
        np.full_like(K, 0.05),
        np.full_like(K, 0.05),
        spot=SPOT,
        config=ForwardConfig(max_abs_z=1e9),
    )
    assert weighted.n_dropped == 0
    assert abs(weighted.F - F) < 0.1 * abs(equal.F - F)


def test_fit_forward_drops_parity_violation_and_refits():
    D, F = 0.99, 3820.0
    K, y, p, hc, hp = _parity_inputs(D, F, STRIKES)
    y = y.copy()
    y[5] += (
        2.0  # 2 dollars on a 10-cent combined-spread pair: parity violated at quoted prices
    )
    fit = fit_forward(K, y, p, hc, hp, spot=SPOT)
    # Exactly the bad pair goes; every good pair survives. A one-shot "drop everything above the
    # threshold" filter would also discard the good pairs the outlier's pull pushed past it (swamping).
    assert fit.n_pairs == len(K) - 1
    assert fit.n_dropped == 1
    assert fit.D == pytest.approx(D, abs=1e-12)
    assert fit.F == pytest.approx(F, abs=1e-9)


def test_fit_forward_needs_min_pairs():
    fit = fit_forward(*_parity_inputs(0.99, 3820.0, STRIKES[:4]), spot=SPOT)
    assert fit.status == ForwardStatus.TOO_FEW_PAIRS
    assert math.isnan(fit.F)


def test_fit_forward_flags_implausible_forward():
    fit = fit_forward(*_parity_inputs(0.99, 3820.0, STRIKES), spot=SPOT / 10)
    assert fit.status == ForwardStatus.IMPLAUSIBLE


def test_fit_forward_standard_error_matches_monte_carlo_spread():
    """se_F from the WLS covariance and the delta method should match the scatter of F over noisy replicas."""
    rng = np.random.default_rng(20221021)
    D, F = 0.99, 3820.0
    K, y0, p, hc, hp = _parity_inputs(D, F, STRIKES, half_spread=0.5)
    noise_sd = 0.1
    estimates, reported = [], []
    for _ in range(400):
        fit = fit_forward(
            K,
            y0 + rng.normal(0, noise_sd, len(K)),
            p,
            hc,
            hp,
            spot=SPOT,
            config=ForwardConfig(max_abs_z=1e9),
        )
        estimates.append(fit.F)
        reported.append(fit.se_F)
    assert np.median(reported) == pytest.approx(np.std(estimates), rel=0.15)


# ------------------------------------------------------------------------------------------- pipeline


@pytest.fixture
def inverted(tmp_path):
    frame = make_chain(TWO_BUCKETS, STRIKES)
    chain = read_options_csv(write_chain(tmp_path, frame))
    return invert_chain(chain)


def test_pipeline_recovers_forward_and_discount_of_the_generating_market(inverted):
    _, forwards = inverted
    assert (forwards["status"] == ForwardStatus.OK).all()
    T = forwards["T"].to_numpy()
    np.testing.assert_allclose(forwards["D"], np.exp(-RATE * T), atol=1e-11)
    np.testing.assert_allclose(
        forwards["F"], SPOT * np.exp((RATE - DIV) * T), atol=1e-7
    )
    np.testing.assert_allclose(forwards["r"], RATE, atol=1e-8)
    np.testing.assert_allclose(forwards["q"], DIV, atol=1e-8)


def test_pipeline_round_trips_sigma_on_every_solved_contract(inverted):
    results, _ = inverted
    solved = results[results["reason"] == Reason.OK]
    assert len(solved) > 0.9 * (results["quote_status"] == QuoteStatus.OK).sum()
    np.testing.assert_allclose(solved["iv"], SIGMA, atol=1e-7)


def test_pipeline_explains_every_row_exactly_once(inverted):
    results, _ = inverted
    reached = results["quote_status"] == QuoteStatus.OK
    assert results.loc[reached, "reason"].notna().all(), (
        "every inverted row needs a solver reason"
    )
    assert results.loc[~reached, "reason"].isna().all(), (
        "rows never inverted must not carry a solver reason"
    )
    assert results.loc[~reached, "iv"].isna().all()


def test_bid_and_ask_vols_bracket_the_mid_vol(inverted):
    results, _ = inverted
    both = results[
        (results["reason"] == Reason.OK)
        & results["iv_bid"].notna()
        & results["iv_ask"].notna()
    ]
    assert len(both) > 0
    assert (both["iv_bid"] <= both["iv"]).all() and (both["iv"] <= both["iv_ask"]).all()


def test_bucket_without_enough_strikes_is_no_forward(tmp_path):
    frame = pd.concat(
        [
            make_chain([("SPX", "2022-08-19", "AM")], STRIKES),
            make_chain([("SPX", "2022-09-16", "AM")], STRIKES[10:13]),
        ]
    )
    results, _ = invert_chain(
        read_options_csv(write_chain(tmp_path, frame), underlyings=["SPX"])
    )
    thin = results["expiration"] == "2022-09-16"
    assert (results.loc[thin, "quote_status"] == QuoteStatus.NO_FORWARD).all()
    assert (
        results.loc[
            ~thin & (results["quote_status"] == QuoteStatus.OK), "reason"
        ].notna()
    ).all()


def test_expired_bucket_is_classified_not_inverted(tmp_path):
    frame = pd.concat(
        [
            make_chain([("SPXW", QUOTE_DATE, "PM")], STRIKES),
            make_chain(TWO_BUCKETS, STRIKES),
        ]
    )
    results, _ = invert_chain(read_options_csv(write_chain(tmp_path, frame)))
    today = results["expiration"] == QUOTE_DATE
    assert (results.loc[today, "quote_status"] == QuoteStatus.EXPIRED).all()
    assert results.loc[today, "iv"].isna().all()


# ---------------------------------------------------------------------------------------- diagnostics


def test_call_and_put_vols_agree_under_the_fitted_forward(inverted):
    results, _ = inverted
    pc = put_call_agreement(results)
    assert len(pc) == len(TWO_BUCKETS)
    assert (pc["max_abs_gap"] < 1e-7).all()


def test_vendor_comparison_reports_agreement_on_matching_vols(inverted):
    results, _ = inverted
    report = compare_to_vendor(results)
    assert report["n"] > 0
    assert report["abs_diff"]["max"] < 1e-7
    assert report["within_1e-4"] == 1.0


def test_bracket_report_headroom(inverted):
    results, _ = inverted
    report = bracket_report(results)
    assert report["s_lo"] < report["s_min"] <= report["s_max"] < report["s_hi"]
    assert report["headroom_low"] > 1 and report["headroom_high"] > 1
