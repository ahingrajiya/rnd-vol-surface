"""
Options chain data layer: read vendor files, measure forwards by put-call parity, invert for implied vol.

    vendor       read historicaldata.net files; time to expiry from AM/PM settlement
    quotes       per-row quote classification (QuoteStatus)
    forwards     per-expiry D and F by weighted put-call parity regression
    pipeline     classify, fit, invert - every input row comes back with exactly one explanation
    diagnostics  cross-checks: vendor IV, put-call IV agreement, bracket monitoring

This subpackage depends on pandas; the pricing core and the solver do not.
"""

from optionslib.chain.diagnostics import (
    bracket_report,
    compare_to_vendor,
    put_call_agreement,
    status_table,
)
from optionslib.chain.forwards import (
    ForwardConfig,
    ForwardFit,
    ForwardStatus,
    fit_forward,
    fit_forwards,
)
from optionslib.chain.pipeline import invert_chain
from optionslib.chain.quotes import QuoteStatus, classify_quotes
from optionslib.chain.vendor import read_options_csv, time_to_expiry

__all__ = [
    "ForwardConfig",
    "ForwardFit",
    "ForwardStatus",
    "QuoteStatus",
    "bracket_report",
    "classify_quotes",
    "compare_to_vendor",
    "fit_forward",
    "fit_forwards",
    "invert_chain",
    "put_call_agreement",
    "read_options_csv",
    "status_table",
    "time_to_expiry",
]
