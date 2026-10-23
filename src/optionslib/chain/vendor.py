"""
Reader for historicaldata.net end-of-day options files, and the time-to-expiry convention.

This is the only module that knows the vendor's file format. Everything downstream works on the
normalized frame it returns, so supporting another vendor means writing another reader that
produces the same columns, not touching the pipeline.

Three properties of the data are enforced here rather than trusted, because each one silently
corrupts the forward regression if it fails:

    1. European exercise only. The solver inverts Black-Scholes, which prices a European claim; an
       American quote carries early-exercise premium that inversion would fold into sigma.
    2. One settlement time per (underlying, expiration) bucket. AM- and PM-settled contracts have
       different effective expiries, so mixing them in one put-call parity regression gives a
       forward that belongs to neither.
    3. One quote date per file. Each file is a single end-of-day snapshot.
"""

import datetime as dt
from collections.abc import Iterable
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

# Columns the pipeline needs. A file missing any of these cannot be processed.
REQUIRED_COLUMNS = (
    "contract",
    "underlying",
    "expiration",
    "type",
    "strike",
    "style",
    "quote_date",
    "bid",
    "ask",
    "underlying_close",
    "settlement_time",
)

# Columns carried through when present: liquidity fields for filtering, and the vendor's own
# implied volatility for cross-checking. Never used as inputs to our inversion.
OPTIONAL_COLUMNS = (
    "bid_size",
    "ask_size",
    "volume",
    "open_interest",
    "last_trade_date",
    "iv_bid",
    "iv_ask",
    "iv",
    "iv_flag",
)

# Vendor columns are renamed with this prefix so they can never be mistaken for our own results.
VENDOR_RENAME = {
    "iv": "vendor_iv",
    "iv_bid": "vendor_iv_bid",
    "iv_ask": "vendor_iv_ask",
    "iv_flag": "vendor_iv_flag",
}

DEFAULT_UNDERLYINGS = ("SPX", "SPXW")

# Time-to-expiry convention. Quotes are an end-of-day snapshot, taken here as the 16:00 New York
# close, which is the time underlying_close refers to. PM-settled contracts expire at that same
# close on the expiration date. AM-settled contracts settle on the special opening quotation of
# the expiration morning, so the distribution they price ends at the 09:30 open, 6.5 hours earlier.
# Times are converted in New York local time, so a daylight-saving change inside the life of the
# option is accounted for exactly.
NEW_YORK = ZoneInfo("America/New_York")
QUOTE_TIME = dt.time(16, 0)
SETTLEMENT_CLOCK = {"AM": dt.time(9, 30), "PM": dt.time(16, 0)}
DAYS_PER_YEAR = 365.0

# How an AM-settled contract's expiry is placed in time.
#   "open"        09:30 on the expiration date, when the settlement value is fixed. Default: the
#                 payoff depends on the Friday open, so the overnight move is risk the option carries.
#   "prior_close" 16:00 on the previous weekday, when AM-settled contracts stop trading. This is the
#                 vendor's convention, 17.5 hours shorter; use it to reproduce their numbers.
# Neither is variance time. Overnight variance per clock hour is well below intraday variance, so the
# economically right T for an AM contract lies between the two.
AM_EXPIRY_CONVENTIONS = ("open", "prior_close")


def time_to_expiry(
    quote_date,
    expiration,
    settlement_time,
    days_per_year: float = DAYS_PER_YEAR,
    am_expiry: str = "open",
) -> np.ndarray:
    """
    Year fraction from the end-of-day quote to the moment the contract's settlement price is fixed.

    T = (expiry instant - quote instant) / days_per_year, both instants in New York time. The quote
    instant is QUOTE_TIME on quote_date; the expiry instant is SETTLEMENT_CLOCK[settlement_time] on
    expiration, or for AM contracts under am_expiry="prior_close", 16:00 on the previous weekday. A
    PM-settled contract on its expiration date therefore has T = 0: no trading time remains, and the
    caller must exclude it rather than invert it.

    Args:
        quote_date: dates of the snapshot (array-like of date strings or datetimes)
        expiration: expiration dates, same length
        settlement_time: "AM" or "PM" per row
        days_per_year: year basis; calendar days over 365 by default
        am_expiry: one of AM_EXPIRY_CONVENTIONS. "prior_close" steps back one weekday and does not
            know exchange holidays, so it is off by a day when the weekday before an AM expiry is a
            holiday (e.g. Juneteenth 2025). It exists for vendor comparison, not as a default.

    Returns:
        np.ndarray of year fractions, one per row. Negative when the settlement instant has passed.
    """
    if am_expiry not in AM_EXPIRY_CONVENTIONS:
        raise ValueError(
            f"am_expiry must be one of {AM_EXPIRY_CONVENTIONS}, got {am_expiry!r}"
        )
    quote_date = (
        pd.to_datetime(pd.Series(quote_date)).dt.normalize().reset_index(drop=True)
    )
    expiration = (
        pd.to_datetime(pd.Series(expiration)).dt.normalize().reset_index(drop=True)
    )
    settlement_time = pd.Series(settlement_time).reset_index(drop=True)

    unknown = set(settlement_time.unique()) - set(SETTLEMENT_CLOCK)
    if unknown:
        raise ValueError(
            f"settlement_time must be one of {sorted(SETTLEMENT_CLOCK)}, got {sorted(unknown)}"
        )

    def _offset(clock: dt.time) -> pd.Timedelta:
        return pd.Timedelta(hours=clock.hour, minutes=clock.minute)

    is_am = (settlement_time == "AM").to_numpy()
    expiry_day = expiration
    settle_offset = settlement_time.map(lambda st: _offset(SETTLEMENT_CLOCK[st]))
    if am_expiry == "prior_close" and is_am.any():
        prior = np.busday_offset(
            expiration.to_numpy().astype("datetime64[D]"), -1, roll="backward"
        )
        expiry_day = expiration.where(~is_am, pd.Series(pd.to_datetime(prior)))
        settle_offset = settle_offset.where(~is_am, _offset(QUOTE_TIME))

    quote_instant = (quote_date + _offset(QUOTE_TIME)).dt.tz_localize(NEW_YORK)
    expiry_instant = (expiry_day + settle_offset).dt.tz_localize(NEW_YORK)

    seconds = (expiry_instant - quote_instant).dt.total_seconds().to_numpy()
    return seconds / (days_per_year * 86400.0)


def read_options_csv(
    path: str | Path,
    underlyings: Iterable[str] = DEFAULT_UNDERLYINGS,
    chunksize: int = 250_000,
    days_per_year: float = DAYS_PER_YEAR,
    am_expiry: str = "open",
) -> pd.DataFrame:
    """
    Read one vendor day file, keeping only the requested underlyings, and return a normalized chain.

    The file is read in chunks and filtered as it is read, so a full-market day (about 1.5 million
    rows) never sits in memory whole. Rows are kept exactly as quoted; no quote is dropped here.
    Quote quality is judged later, in the pipeline, where every rejection is recorded with a reason.

    Added columns:
        call_or_put: +1 for calls, -1 for puts, matching Payoff and NormalizedProblem
        T: time to expiry in years, from time_to_expiry

    Args:
        path: path to a historicaldata.net options CSV
        underlyings: underlying symbols to keep. Every one must be present in the file, so a typo or
            a symbol the archive does not carry fails loudly instead of returning an empty chain.
        chunksize: rows per read chunk
        days_per_year: year basis passed to time_to_expiry
        am_expiry: AM-settlement convention passed to time_to_expiry

    Returns:
        pd.DataFrame with the required columns, whichever optional columns the file has (vendor
        implied-volatility columns renamed with a vendor_ prefix), call_or_put and T.

    Raises:
        ValueError: if required columns are missing, a requested underlying is absent, any row is
            American-style, a bucket mixes settlement times, or the file holds more than one quote date.
    """
    path = Path(path)
    wanted = tuple(dict.fromkeys(underlyings))
    if not wanted:
        raise ValueError("underlyings must name at least one symbol")

    header = pd.read_csv(path, nrows=0).columns
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(
            f"{path.name} lacks required columns {missing}; this reader expects the "
            "historicaldata.net 34-column options format"
        )
    usecols = list(REQUIRED_COLUMNS) + [c for c in OPTIONAL_COLUMNS if c in header]

    # Strings for identifiers and dates; numbers for prices. Explicit dtypes stop pandas guessing a
    # column's type from its first chunk and then disagreeing with itself on a later one.
    string_cols = [
        "contract",
        "underlying",
        "expiration",
        "type",
        "style",
        "quote_date",
        "settlement_time",
    ]
    if "last_trade_date" in usecols:
        string_cols.append("last_trade_date")
    dtypes = {c: "string" for c in string_cols}
    dtypes.update({c: "float64" for c in usecols if c not in dtypes})

    chunks = []
    for chunk in pd.read_csv(path, usecols=usecols, dtype=dtypes, chunksize=chunksize):
        chunks.append(chunk[chunk["underlying"].isin(wanted)])
    chain = (
        pd.concat(chunks, ignore_index=True)
        if chunks
        else pd.DataFrame(columns=usecols)
    )

    found = set(chain["underlying"].unique())
    absent = [u for u in wanted if u not in found]
    if absent:
        raise ValueError(f"{path.name} has no rows for underlyings {absent}")

    _check_invariants(chain, path.name)

    chain = chain.rename(
        columns={k: v for k, v in VENDOR_RENAME.items() if k in chain.columns}
    )
    chain["call_or_put"] = np.where(chain["type"] == "call", 1, -1).astype(np.int64)
    chain["T"] = time_to_expiry(
        chain["quote_date"],
        chain["expiration"],
        chain["settlement_time"],
        days_per_year,
        am_expiry,
    )
    return chain


def _check_invariants(chain: pd.DataFrame, name: str) -> None:
    """Enforce the three data properties the module docstring lists. Raises ValueError on violation."""
    american = sorted(chain.loc[chain["style"] != "E", "underlying"].unique())
    if american:
        raise ValueError(
            f"{name}: underlyings {american} are not European-style. Black-Scholes inversion and the "
            "Breeden-Litzenberger density are only valid for European exercise."
        )

    bad_type = sorted(set(chain["type"].unique()) - {"call", "put"})
    if bad_type:
        raise ValueError(f"{name}: unexpected option type values {bad_type}")

    n_settle = chain.groupby(["underlying", "expiration"])["settlement_time"].nunique()
    mixed = n_settle[n_settle > 1]
    if not mixed.empty:
        raise ValueError(
            f"{name}: buckets {list(mixed.index)} mix AM and PM settlement; they have different "
            "effective expiries and cannot share one forward"
        )

    dates = chain["quote_date"].unique()
    if len(dates) != 1:
        raise ValueError(
            f"{name}: expected one quote_date per file, found {sorted(dates)}"
        )
