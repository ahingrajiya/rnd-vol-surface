"""
Quote classification: does a row carry a price the solver can use?

This is the data-layer half of the two-layer rejection scheme described in pipeline.py. It judges
the quote only - never the implied volatility, which is the solver's job.
"""

from enum import StrEnum, auto

import numpy as np
import pandas as pd


class QuoteStatus(StrEnum):
    """
    Whether a row carries a price the solver can use. Checked in this order; the first that applies wins.

    Attributes:
        OK: two-sided, uncrossed quote with time remaining and a fitted forward.
        EXPIRED: T <= 0. No trading time remains, so implied volatility is undefined.
        ONE_SIDED: bid or ask missing or non-positive. A mid needs both sides.
        CROSSED: bid > ask. Not a tradeable market.
        NO_FORWARD: the quote is fine but its bucket had no usable forward fit. Assigned by the
            pipeline after the forwards are fitted, not by classify_quotes.
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name

    OK = auto()
    EXPIRED = auto()
    ONE_SIDED = auto()
    CROSSED = auto()
    NO_FORWARD = auto()


def classify_quotes(chain: pd.DataFrame) -> pd.DataFrame:
    """
    Add mid, half_spread and a first-pass quote_status (everything except NO_FORWARD).

    Args:
        chain: normalized chain from vendor.read_options_csv

    Returns:
        A copy of chain with mid, half_spread and quote_status columns. mid and half_spread are NaN
        where the quote is one-sided.
    """
    out = chain.copy()
    bid = out["bid"].to_numpy(dtype=float)
    ask = out["ask"].to_numpy(dtype=float)
    T = out["T"].to_numpy(dtype=float)

    two_sided = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > 0)
    status = np.full(len(out), QuoteStatus.OK, dtype=object)
    status[two_sided & (bid > ask)] = QuoteStatus.CROSSED
    status[~two_sided] = QuoteStatus.ONE_SIDED
    status[T <= 0] = QuoteStatus.EXPIRED  # assigned last so it takes precedence

    out["mid"] = np.where(two_sided, 0.5 * (bid + ask), np.nan)
    out["half_spread"] = np.where(two_sided, 0.5 * (ask - bid), np.nan)
    out["quote_status"] = status
    return out
