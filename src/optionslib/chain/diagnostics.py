"""
Cross-checks on an inverted chain. None of these change a result; they tell you whether to trust it.

    compare_to_vendor    our IV against the vendor's on rows both solved. An independent
                         implementation fed the same quotes; agreement validates T, D, F and the solver
                         together.
    put_call_agreement   call IV minus put IV at the same strike. Parity makes them equal under the
                         correct forward and discount factor, so the gap measures forward error
                         directly - and needs no outside reference.
    bracket_report       how close any solved s came to the bracket endpoints. A fixed bracket is a
                         property of the numerics, not of the data; this is how you notice if data
                         ever outgrows it, without making the bracket data-dependent.
    status_table         where every row ended up, crossed with the vendor's iv_flag.
"""

import numpy as np
import pandas as pd

from optionslib.chain.quotes import QuoteStatus
from optionslib.vol.implied import DEFAULT_CONFIG, Reason, SolverConfig

# Vendor iv_flag 0: solved with a put-call parity forward - the same method as ours.
VENDOR_SOLVED_PARITY = 0


def _solved(results: pd.DataFrame) -> pd.Series:
    return (results["quote_status"] == QuoteStatus.OK) & (
        results["reason"] == Reason.OK
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {k: float("nan") for k in ("median", "p90", "p99", "max")}
    return {
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "max": float(np.max(values)),
    }


def compare_to_vendor(results: pd.DataFrame) -> dict:
    """
    Our implied vol against the vendor's, on rows where both of us solved with a parity forward.

    Differences are reported in vol points (0.01 = one vol point) and, more usefully, relative to the
    quote's own resolution: the width between the vendor's bid and ask implied vols. A difference far
    inside that width is agreement to better than the data can distinguish.

    Args:
        results: output of pipeline.invert_chain, carrying vendor_iv and vendor_iv_flag

    Returns:
        dict with n (rows compared), abs_diff quantiles, the fraction within 1e-4 and 1e-3 absolute,
        and the fraction whose difference is under 10% of the vendor's bid-ask vol width.
    """
    if "vendor_iv" not in results:
        raise ValueError(
            "results carry no vendor_iv column; the source file had no vendor IV"
        )
    both = (
        _solved(results)
        & (results["vendor_iv_flag"] == VENDOR_SOLVED_PARITY)
        & results["vendor_iv"].notna()
    )
    sub = results.loc[both]
    diff = (sub["iv"] - sub["vendor_iv"]).to_numpy(dtype=float)
    abs_diff = np.abs(diff)
    out = {
        "n": len(sub),
        "abs_diff": _quantiles(abs_diff),
        "mean_signed_diff": float(np.mean(diff)) if len(diff) else float("nan"),
    }
    out["within_1e-4"] = float(np.mean(abs_diff <= 1e-4)) if len(diff) else float("nan")
    out["within_1e-3"] = float(np.mean(abs_diff <= 1e-3)) if len(diff) else float("nan")
    if {"vendor_iv_bid", "vendor_iv_ask"} <= set(sub.columns):
        width = (sub["vendor_iv_ask"] - sub["vendor_iv_bid"]).to_numpy(dtype=float)
        has_w = np.isfinite(width) & (width > 0)
        out["within_10pct_of_quote_width"] = (
            float(np.mean(abs_diff[has_w] <= 0.1 * width[has_w]))
            if has_w.any()
            else float("nan")
        )
    return out


def put_call_agreement(results: pd.DataFrame) -> pd.DataFrame:
    """
    Call IV minus put IV at each strike where both solved, per bucket.

    With the right F and D, parity makes the call and put prices at a strike carry one implied vol, so
    this gap is zero up to quote noise. A forward error dF shifts the two in opposite directions, so a
    systematic gap is the fingerprint of a bad forward - visible without any outside reference.

    Returns:
        pd.DataFrame, one row per bucket: n strikes, median and max |call IV - put IV|, and the median
        signed gap (a non-zero median signed gap indicates forward bias rather than noise).
    """
    keys = ["quote_date", "underlying", "expiration", "strike"]
    solved = results.loc[_solved(results), keys + ["call_or_put", "iv"]]
    wide = solved.pivot_table(
        index=keys, columns="call_or_put", values="iv", aggfunc="first"
    )
    wide = wide.dropna()
    if wide.empty:
        return pd.DataFrame(
            columns=[
                "quote_date",
                "underlying",
                "expiration",
                "n",
                "median_abs_gap",
                "max_abs_gap",
                "median_gap",
            ]
        )
    wide["gap"] = wide[1] - wide[-1]
    g = wide.reset_index().groupby(["quote_date", "underlying", "expiration"])["gap"]
    return pd.DataFrame(
        {
            "n": g.size(),
            "median_abs_gap": g.apply(lambda x: float(np.median(np.abs(x)))),
            "max_abs_gap": g.apply(lambda x: float(np.max(np.abs(x)))),
            "median_gap": g.median(),
        }
    ).reset_index()


def bracket_report(
    results: pd.DataFrame, config: SolverConfig = DEFAULT_CONFIG
) -> dict:
    """
    Distribution of solved total volatility s, against the solver's fixed bracket.

    Returns:
        dict with n, s quantiles (min, p1, median, p99, max), the bracket, and the headroom factors
        min_s / s_lo and s_hi / max_s. A headroom near 1 on either side means the data is reaching the
        edge of the bracket and the choice of endpoint is starting to matter.
    """
    s = results.loc[_solved(results), "s"].to_numpy(dtype=float)
    if len(s) == 0:
        return {"n": 0}
    return {
        "n": len(s),
        "s_min": float(s.min()),
        "s_p01": float(np.quantile(s, 0.01)),
        "s_median": float(np.median(s)),
        "s_p99": float(np.quantile(s, 0.99)),
        "s_max": float(s.max()),
        "s_lo": config.s_lo,
        "s_hi": config.s_hi,
        "headroom_low": float(s.min() / config.s_lo),
        "headroom_high": float(config.s_hi / s.max()),
    }


def status_table(results: pd.DataFrame) -> pd.DataFrame:
    """
    Count of rows by final outcome: quote_status, then solver reason for rows that reached the solver.

    When the vendor's iv_flag is present it is added as columns, so our classification can be read
    against theirs (e.g. our BELOW_FLOOR against their flag 3, our ONE_SIDED against their flag 2).
    """
    outcome = results["quote_status"].astype(str)
    reached = results["quote_status"] == QuoteStatus.OK
    outcome = outcome.where(~reached, "OK/" + results["reason"].astype(str))
    if "vendor_iv_flag" in results:
        flag = (
            results["vendor_iv_flag"].astype("Int64").astype(str).radd("vendor_flag_")
        )
        return pd.crosstab(outcome, flag, margins=True)
    return outcome.value_counts().to_frame("rows")
