"""
Chain pipeline: classify quotes, fit forwards, and invert every usable quote for implied volatility.

Two layers of rejection are kept separate on purpose, each recorded with a reason:

    quote_status (this module) - is there a usable price at all? Missing side, crossed market,
        no trading time left, or no forward for the bucket. These are properties of the data.
    reason (optionslib.vol.implied.Reason) - given a usable price, what did the solver find?
        No-arbitrage violation, root outside the bracket, or convergence failure.

A row is inverted only when quote_status is OK, and then always gets a solver reason. Nothing is
silently dropped: every input row appears in the output with exactly one explanation.

The inversion itself is delegated unchanged to implied_vol. The chain supplies D and F measured by
put-call parity, through the NormalizedProblem constructor that exists for exactly this case, so no
interest rate or dividend yield is invented to build a Market.
"""

import numpy as np
import pandas as pd

from optionslib.chain.forwards import (
    BUCKET,
    DEFAULT_FORWARD_CONFIG,
    ForwardConfig,
    ForwardStatus,
    fit_forwards,
)
from optionslib.chain.quotes import QuoteStatus, classify_quotes
from optionslib.vol.implied import (
    DEFAULT_CONFIG,
    NormalizedProblem,
    Reason,
    SolverConfig,
    implied_vol,
)


def _invert_prices(
    T: np.ndarray,
    K: np.ndarray,
    F: np.ndarray,
    D: np.ndarray,
    call_or_put: np.ndarray,
    price: np.ndarray,
    config: SolverConfig,
) -> dict[str, np.ndarray]:
    """Call implied_vol once per row. Scalar by design; see the implied module docstring for why."""
    n = len(T)
    sigma = np.full(n, np.nan)
    s = np.full(n, np.nan)
    iterations = np.zeros(n, dtype=np.int64)
    reason = np.empty(n, dtype=object)
    method = np.empty(n, dtype=object)
    # In the flat-vega wings Newton's step (b - b*) / vega overflows to infinity. The solver handles
    # that correctly - an infinite step leaves the bracket and Brent takes over - so the warning
    # numpy raises on the way is suppressed here rather than printed once per wing contract.
    with np.errstate(over="ignore"):
        for i in range(n):
            if not np.isfinite(price[i]) or price[i] <= 0:
                # A one-sided leg when inverting bid or ask separately. Zero price is below every floor.
                reason[i] = Reason.BELOW_FLOOR
                method[i] = None
                continue
            problem = NormalizedProblem(
                T=float(T[i]),
                K=float(K[i]),
                F=float(F[i]),
                D=float(D[i]),
                call_or_put=int(call_or_put[i]),
            )
            result = implied_vol(problem, float(price[i]), config)
            sigma[i] = result.sigma
            s[i] = result.s
            iterations[i] = result.iterations
            reason[i] = result.reason
            method[i] = result.method
    return {
        "sigma": sigma,
        "s": s,
        "iterations": iterations,
        "reason": reason,
        "method": method,
    }


def invert_chain(
    chain: pd.DataFrame,
    solver_config: SolverConfig = DEFAULT_CONFIG,
    forward_config: ForwardConfig = DEFAULT_FORWARD_CONFIG,
    with_bid_ask: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Full pipeline for one or more days: classify quotes, fit forwards, invert usable quotes.

    Args:
        chain: normalized chain from vendor.read_options_csv (several days may be concatenated;
            forwards are fitted per (quote_date, underlying, expiration))
        solver_config: SolverConfig for implied_vol
        forward_config: ForwardConfig for the parity regression
        with_bid_ask: also invert the bid and the ask separately. Their implied vols bracket the mid's
            and measure how precisely the quote determines sigma - the quantity that justified s_lo.

    Returns:
        (results, forwards).
        results: every input row, with mid, half_spread, quote_status, D, F, x, and for inverted
            rows iv, s, reason, method, iterations; plus iv_bid and iv_ask when with_bid_ask.
        forwards: one row per bucket from forwards.fit_forwards.
    """
    quotes = classify_quotes(chain)
    forwards = fit_forwards(quotes, forward_config)

    good = forwards.loc[forwards["status"] == ForwardStatus.OK, BUCKET + ["D", "F"]]
    results = quotes.merge(good, on=BUCKET, how="left")
    no_forward = (results["quote_status"] == QuoteStatus.OK) & results["F"].isna()
    results.loc[no_forward, "quote_status"] = QuoteStatus.NO_FORWARD

    has_fwd = results["F"].notna().to_numpy()
    results["x"] = np.where(has_fwd, np.log(results["F"] / results["strike"]), np.nan)

    for col in ("iv", "s"):
        results[col] = np.nan
    results["iterations"] = 0
    results["reason"] = None
    results["method"] = None
    if with_bid_ask:
        results["iv_bid"] = np.nan
        results["iv_ask"] = np.nan

    ok = (results["quote_status"] == QuoteStatus.OK).to_numpy()
    if ok.any():
        sub = results.loc[ok]
        args = (
            sub["T"].to_numpy(dtype=float),
            sub["strike"].to_numpy(dtype=float),
            sub["F"].to_numpy(dtype=float),
            sub["D"].to_numpy(dtype=float),
            sub["call_or_put"].to_numpy(dtype=np.int64),
        )
        mid = _invert_prices(*args, sub["mid"].to_numpy(dtype=float), solver_config)
        results.loc[ok, "iv"] = mid["sigma"]
        results.loc[ok, "s"] = mid["s"]
        results.loc[ok, "iterations"] = mid["iterations"]
        results.loc[ok, "reason"] = mid["reason"]
        results.loc[ok, "method"] = mid["method"]
        if with_bid_ask:
            results.loc[ok, "iv_bid"] = _invert_prices(
                *args, sub["bid"].to_numpy(dtype=float), solver_config
            )["sigma"]
            results.loc[ok, "iv_ask"] = _invert_prices(
                *args, sub["ask"].to_numpy(dtype=float), solver_config
            )["sigma"]

    return results, forwards
