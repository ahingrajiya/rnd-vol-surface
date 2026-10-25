"""
Forward price and discount factor per expiry, extracted from the option quotes by put-call parity.

For European options on the same underlying and expiry, put-call parity holds at every strike:

    C(K) - P(K) = D * (F - K) = D*F - D*K

which is a straight line in K with slope -D and intercept D*F. Regressing the quoted C - P on K
across strikes therefore measures D and F directly from the options, with no assumed interest
rate or dividend yield. Implied rate and carry follow as diagnostics:

    r = -ln(D) / T,     q = r - ln(F/S) / T

Why this and not F = S * exp((r - q) T): the vendor publishes no dividend yield, and the index close
and the option quotes are not simultaneous. On 2022-07-01 the 4-day forward sits 2.4 index points
below the reported close, a gap carry cannot produce at that horizon. A forward built from the close
would shift short-dated implied vols by most of a vol point; the parity forward is measured from the
same snapshot as the quotes it is used to invert.

The fit is weighted least squares. Each strike's C - P mid is uncertain by roughly the two quotes'
half-spreads, so the weight is 1 / (h_call^2 + h_put^2). Deep in-the-money legs have wide quotes
and are down-weighted automatically; the liquid strikes near the money dominate.

Robustness: a pair whose residual exceeds its own combined half-spread violates parity at the quoted
prices - a stale leg or a bad print. Such pairs are removed one at a time, the worst first, refitting
after each removal, until every remaining pair sits within max_abs_z half-spreads. One at a time
matters: a gross outlier pulls the first fit and inflates every other residual, so dropping all pairs
above the threshold at once would discard good pairs along with the bad one (swamping). Backward
elimination removes only what is inconsistent with a fit the outlier no longer distorts.
"""

from dataclasses import dataclass
from enum import StrEnum, auto
from typing import NamedTuple

import numpy as np
import pandas as pd

from optionslib.chain.quotes import QuoteStatus


class ForwardStatus(StrEnum):
    """
    Outcome of a forward fit for one bucket.

    Attributes:
        OK: D and F fitted and plausible.
        TOO_FEW_PAIRS: fewer than min_pairs strikes have usable quotes on both legs; no fit.
        IMPLAUSIBLE: a fit was produced but D or F lies outside the plausibility bounds, which
            indicates bad quotes rather than a real market; the bucket is not used.
    """

    @staticmethod
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        return name

    OK = auto()
    TOO_FEW_PAIRS = auto()
    IMPLAUSIBLE = auto()


@dataclass(frozen=True)
class ForwardConfig:
    """
    Settings for the parity regression.

    Attributes:
        min_pairs: minimum strikes with two-sided quotes on both legs. Two parameters are fitted,
            and backward elimination needs slack to remove pairs, so 5 rather than 3.
        max_abs_z: residuals larger than this many combined half-spreads are treated as parity
            violations and removed, worst first, with a refit after each removal. At 1.0, a pair
            survives exactly when parity holds within its own quotes.
        min_half_spread: floor on each leg's half-spread in the weights, so a locked market
            (bid == ask) cannot receive infinite weight. Half a cent.
        d_bounds: plausible range for D. Below 0.5 or above 1.2 is not a USD money market.
        f_over_s_bounds: plausible range for F / spot.
    """

    min_pairs: int = 5
    max_abs_z: float = 1.0
    min_half_spread: float = 0.005
    d_bounds: tuple[float, float] = (0.5, 1.2)
    f_over_s_bounds: tuple[float, float] = (0.5, 2.0)

    def __post_init__(self):
        if self.min_pairs < 3:
            raise ValueError(
                f"min_pairs must be at least 3 to fit two parameters, got {self.min_pairs}"
            )
        if self.max_abs_z <= 0:
            raise ValueError(f"max_abs_z must be positive, got {self.max_abs_z}")
        if self.min_half_spread <= 0:
            raise ValueError(
                f"min_half_spread must be positive, got {self.min_half_spread}"
            )


DEFAULT_FORWARD_CONFIG = ForwardConfig()


class ForwardFit(NamedTuple):
    """
    Result of the parity regression for one bucket.

    Attributes:
        status: ForwardStatus
        D: discount factor (NaN unless a fit was produced)
        F: forward price
        se_D, se_F: standard errors from the weighted fit; se_F by the delta method on F = -a/b.
            These assume errors independent across strikes. Quotes in one expiry share errors (a
            common quoting model, a common snapshot time), so on real data they are lower bounds:
            on 2022-07-01 the implied rate scatters between neighbouring short expiries by several
            times se_r.
        n_pairs: pairs used in the final fit
        n_dropped: pairs removed by backward elimination
        resid_scale: weighted residual standard deviation, in units of combined half-spread
        max_abs_z: largest |residual| / combined half-spread among pairs kept
    """

    status: ForwardStatus
    D: float
    F: float
    se_D: float
    se_F: float
    n_pairs: int
    n_dropped: int
    resid_scale: float
    max_abs_z: float


def _nan_fit(status: ForwardStatus, n_pairs: int, n_dropped: int = 0) -> ForwardFit:
    nan = float("nan")
    return ForwardFit(status, nan, nan, nan, nan, n_pairs, n_dropped, nan, nan)


def _wls(K: np.ndarray, y: np.ndarray, v: np.ndarray):
    """Weighted least squares of y = a + b K with weights 1/v. Returns (a, b, cov, z)."""
    sw = 1.0 / np.sqrt(v)
    X = np.column_stack([np.ones_like(K), K])
    coef, *_ = np.linalg.lstsq(X * sw[:, None], y * sw, rcond=None)
    a, b = coef
    resid = y - (a + b * K)
    z = resid / np.sqrt(v)
    dof = len(K) - 2
    sigma2 = float(np.sum(z**2) / dof) if dof > 0 else float("nan")
    XtWX = (X * (1.0 / v)[:, None]).T @ X
    cov = np.linalg.inv(XtWX) * sigma2
    return a, b, cov, z


def fit_forward(
    strike: np.ndarray,
    call_mid: np.ndarray,
    put_mid: np.ndarray,
    call_half_spread: np.ndarray,
    put_half_spread: np.ndarray,
    spot: float | None = None,
    config: ForwardConfig = DEFAULT_FORWARD_CONFIG,
) -> ForwardFit:
    """
    Fit D and F for one bucket from paired call and put quotes at common strikes.

    Inputs are aligned arrays, one entry per strike that has usable two-sided quotes on both legs.
    Pure numpy, so the statistics are testable without building a chain.

    Args:
        strike: strikes K
        call_mid, put_mid: mid prices of the call and put at each strike
        call_half_spread, put_half_spread: (ask - bid) / 2 for each leg
        spot: underlying close, used only for the F / spot plausibility check. None skips it.
        config: ForwardConfig

    Returns:
        ForwardFit. status is OK only when a fit was produced and passed the plausibility bounds.
    """
    K = np.asarray(strike, dtype=float)
    y = np.asarray(call_mid, dtype=float) - np.asarray(put_mid, dtype=float)
    hc = np.maximum(np.asarray(call_half_spread, dtype=float), config.min_half_spread)
    hp = np.maximum(np.asarray(put_half_spread, dtype=float), config.min_half_spread)
    v = hc**2 + hp**2

    n = len(K)
    if n < config.min_pairs:
        return _nan_fit(ForwardStatus.TOO_FEW_PAIRS, n)

    a, b, cov, z = _wls(K, y, v)
    n_dropped = 0
    while np.max(np.abs(z)) > config.max_abs_z:
        if len(K) - 1 < config.min_pairs:
            return _nan_fit(ForwardStatus.TOO_FEW_PAIRS, len(K) - 1, n_dropped + 1)
        keep = np.ones(len(K), dtype=bool)
        keep[np.argmax(np.abs(z))] = False
        K, y, v = K[keep], y[keep], v[keep]
        n_dropped += 1
        a, b, cov, z = _wls(K, y, v)

    D = -b
    F = -a / b
    var_a, var_b, cov_ab = cov[0, 0], cov[1, 1], cov[0, 1]
    var_F = var_a / b**2 + a**2 * var_b / b**4 - 2.0 * a * cov_ab / b**3
    fit = ForwardFit(
        status=ForwardStatus.OK,
        D=float(D),
        F=float(F),
        se_D=float(np.sqrt(var_b)),
        se_F=float(np.sqrt(max(var_F, 0.0))),
        n_pairs=len(K),
        n_dropped=n_dropped,
        resid_scale=float(np.sqrt(np.mean(z**2))),
        max_abs_z=float(np.max(np.abs(z))),
    )

    d_lo, d_hi = config.d_bounds
    plausible = d_lo <= D <= d_hi and F > 0
    if plausible and spot is not None:
        f_lo, f_hi = config.f_over_s_bounds
        plausible = f_lo <= F / spot <= f_hi
    if not plausible:
        return fit._replace(status=ForwardStatus.IMPLAUSIBLE)
    return fit


BUCKET = ["quote_date", "underlying", "expiration"]


def fit_forwards(
    chain: pd.DataFrame, config: ForwardConfig = DEFAULT_FORWARD_CONFIG
) -> pd.DataFrame:
    """
    Fit a forward for every (quote_date, underlying, expiration) bucket of a chain.

    The chain must already carry quote_status, mid and half_spread (see quotes.classify_quotes).
    Only rows with quote_status OK are paired: a strike enters the regression when both its call and
    its put have usable two-sided quotes.

    Args:
        chain: normalized chain with quote_status, mid, half_spread, T, underlying_close
        config: ForwardConfig

    Returns:
        pd.DataFrame with one row per bucket: the bucket keys, T, settlement_time, spot, every
        ForwardFit field, and the implied rate r, its standard error se_r and the carry q as
        diagnostics. At short T, D is barely distinguishable from 1, so r = -ln(D)/T is noisy, and
        more so than se_r says (see ForwardFit on why it is a lower bound). This is nearly harmless
        for pricing - an error dD moves an at-the-money price by only dD/D, about 0.007 vol points for
        a 50 bp rate error at three weeks - but read the short end of the rate curve with care.
    """
    usable = chain[chain["quote_status"] == QuoteStatus.OK]
    cols = ["mid", "half_spread"]
    calls = usable[usable["call_or_put"] == 1].set_index(BUCKET + ["strike"])[cols]
    puts = usable[usable["call_or_put"] == -1].set_index(BUCKET + ["strike"])[cols]
    pairs = calls.join(puts, lsuffix="_call", rsuffix="_put", how="inner").reset_index()

    meta = (
        chain.groupby(BUCKET, sort=True)
        .agg(
            T=("T", "first"),
            settlement_time=("settlement_time", "first"),
            spot=("underlying_close", "first"),
        )
        .reset_index()
    )

    grouped = {key: g for key, g in pairs.groupby(BUCKET, sort=False)}
    rows = []
    for rec in meta.itertuples(index=False):
        key = (rec.quote_date, rec.underlying, rec.expiration)
        g = grouped.get(key)
        if g is None or rec.T <= 0:
            fit = _nan_fit(ForwardStatus.TOO_FEW_PAIRS, 0 if g is None else len(g))
        else:
            fit = fit_forward(
                g["strike"].to_numpy(),
                g["mid_call"].to_numpy(),
                g["mid_put"].to_numpy(),
                g["half_spread_call"].to_numpy(),
                g["half_spread_put"].to_numpy(),
                spot=rec.spot,
                config=config,
            )
        r = se_r = q = float("nan")
        if fit.status == ForwardStatus.OK:
            r = -np.log(fit.D) / rec.T
            se_r = fit.se_D / (fit.D * rec.T)  # delta method on r = -ln(D)/T
            q = r - np.log(fit.F / rec.spot) / rec.T
        rows.append({**rec._asdict(), **fit._asdict(), "r": r, "se_r": se_r, "q": q})
    return pd.DataFrame(rows)
