#!/usr/bin/env python3
"""
Invert one or more vendor day files for implied volatility and report the cross-checks.

    python scripts/invert_chain.py data/2022-07-01_options.csv
    python scripts/invert_chain.py data/2022-07-*_options.csv --out results/2022-07.parquet
    python scripts/invert_chain.py data/2022-07-01_options.csv --am-expiry prior_close

Prints a worked example for one contract (every intermediate from the quote to sigma), then the
chain-level checks: forward fits, where every row ended up, agreement with the vendor's IV,
call-put IV agreement, and the solver bracket against the solved s range.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from optionslib.chain import (
    ForwardStatus,
    bracket_report,
    compare_to_vendor,
    invert_chain,
    put_call_agreement,
    read_options_csv,
    status_table,
)
from optionslib.vol.implied import DEFAULT_CONFIG, Reason, normalized_bounds


def worked_example(
    results: pd.DataFrame,
    forwards: pd.DataFrame,
    expiration: str | None,
    strike: float | None,
) -> None:
    """Print one contract end to end. Defaults to the first SPX monthly beyond 30 days, call nearest the forward."""
    solved = results[(results["reason"] == Reason.OK) & (results["call_or_put"] == 1)]
    if expiration is None:
        cands = solved[(solved["underlying"] == "SPX") & (solved["T"] > 30 / 365)]
        if cands.empty:
            cands = solved[solved["T"] > 30 / 365]
        if cands.empty:
            print("no solved call beyond 30 days to use as an example")
            return
        expiration = cands.sort_values("T")["expiration"].iloc[0]
        pool = cands[cands["expiration"] == expiration]
    else:
        pool = solved[solved["expiration"] == expiration]
    if pool.empty:
        print(f"no solved call for expiration {expiration}")
        return
    if strike is None:
        row = pool.iloc[(pool["strike"] - pool["F"]).abs().argmin()]
    else:
        row = pool.iloc[(pool["strike"] - strike).abs().argmin()]

    fwd = forwards[
        (forwards["underlying"] == row["underlying"])
        & (forwards["expiration"] == row["expiration"])
    ].iloc[0]
    norm = row["D"] * np.sqrt(row["F"] * row["strike"])
    b_target = row["mid"] / norm
    bounds = normalized_bounds(row["x"], int(row["call_or_put"]))

    print("=" * 78)
    print(f"WORKED EXAMPLE  {row['contract']}")
    print("=" * 78)
    print(
        f"  quote            {row['quote_date']}  {row['underlying']} {row['type']}  K = {row['strike']:.0f}"
        f"  expires {row['expiration']} ({row['settlement_time']}-settled)"
    )
    print(
        f"  market           bid {row['bid']:.2f}  ask {row['ask']:.2f}  ->  mid V = {row['mid']:.4f}"
        f"     index close {row['underlying_close']:.2f}"
    )
    print(f"  time to expiry   T = {row['T']:.6f} years ({row['T'] * 365:.3f} days)")
    print(
        f"  parity fit       {int(fwd['n_pairs'])} strike pairs ({int(fwd['n_dropped'])} dropped),"
        f" max |residual| = {fwd['max_abs_z']:.2f} half-spreads"
    )
    print(
        f"                   D = {row['D']:.6f} +/- {fwd['se_D']:.1e}    implied r = {fwd['r'] * 100:.3f}%"
    )
    print(
        f"                   F = {row['F']:.3f} +/- {fwd['se_F']:.3f}    implied q = {fwd['q'] * 100:.3f}%"
    )
    print(f"  normalized       x = ln(F/K) = {row['x']:+.6f}")
    print(
        f"                   b* = V / (D sqrt(FK)) = {row['mid']:.4f} / {norm:.4f} = {b_target:.6f}"
    )
    print(
        f"                   no-arb interval ({bounds.floor:.6f}, {bounds.ceiling:.6f})  -> inside"
    )
    print(
        f"  solver           s = {row['s']:.8f} by {row['method']} in {int(row['iterations'])} iterations"
    )
    print(
        f"  result           sigma = s / sqrt(T) = {row['iv']:.6f}   ({row['iv'] * 100:.3f} vol points)"
    )
    if "iv_bid" in row and np.isfinite(row["iv_bid"]) and np.isfinite(row["iv_ask"]):
        print(
            f"                   bid/ask implied vols [{row['iv_bid']:.6f}, {row['iv_ask']:.6f}]"
            f"  -> quote pins sigma to +/- {(row['iv_ask'] - row['iv_bid']) / 2 * 100:.3f} vol points"
        )
    if "vendor_iv" in row and np.isfinite(row["vendor_iv"]):
        print(
            f"  vendor           sigma = {row['vendor_iv']:.6f}   difference {row['iv'] - row['vendor_iv']:+.6f}"
        )
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("paths", nargs="+", help="vendor options CSV files")
    ap.add_argument("--underlyings", nargs="+", default=["SPX", "SPXW"])
    ap.add_argument("--am-expiry", choices=["open", "prior_close"], default="open")
    ap.add_argument("--no-bid-ask", action="store_true", help="invert the mid only")
    ap.add_argument("--example-expiration", default=None)
    ap.add_argument("--example-strike", type=float, default=None)
    ap.add_argument(
        "--out", default=None, help="write results to .parquet (needs pyarrow) or .csv"
    )
    args = ap.parse_args()

    t0 = time.time()
    chain = pd.concat(
        [
            read_options_csv(p, args.underlyings, am_expiry=args.am_expiry)
            for p in args.paths
        ],
        ignore_index=True,
    )
    t1 = time.time()
    results, forwards = invert_chain(chain, with_bid_ask=not args.no_bid_ask)
    t2 = time.time()
    print(
        f"{len(args.paths)} file(s), {len(chain):,} contracts on {sorted(chain['underlying'].unique())}"
        f"   read {t1 - t0:.1f}s, invert {t2 - t1:.1f}s   (AM expiry convention: {args.am_expiry})\n"
    )

    worked_example(results, forwards, args.example_expiration, args.example_strike)

    print("FORWARD FITS")
    counts = forwards["status"].value_counts().to_dict()
    print("  " + "  ".join(f"{k}: {v}" for k, v in counts.items()))
    ok = forwards[forwards["status"] == ForwardStatus.OK]
    if not ok.empty:
        print(
            f"  max |residual| across buckets {ok['max_abs_z'].max():.2f} half-spreads;"
            f" pairs dropped {int(ok['n_dropped'].sum())}"
        )
        show = ok.assign(
            days=(ok["T"] * 365).round(1),
            r_pct=ok["r"] * 100,
            se_r_pct=ok["se_r"] * 100,
            q_pct=ok["q"] * 100,
        )
        show = show.sort_values("T").iloc[:: max(1, len(show) // 12)]
        print(
            show[
                [
                    "underlying",
                    "expiration",
                    "days",
                    "n_pairs",
                    "F",
                    "se_F",
                    "r_pct",
                    "se_r_pct",
                    "q_pct",
                ]
            ].to_string(index=False, float_format=lambda v: f"{v:.3f}")
        )
        print(
            "  (r is noisy at the shortest expiries, where D is barely distinguishable from 1;"
            " se_r assumes independent errors across strikes and is a lower bound.)"
        )
    print()

    print("WHERE EVERY ROW ENDED UP")
    print(status_table(results).to_string())
    print()

    if "vendor_iv" in results:
        v = compare_to_vendor(results)
        a = v["abs_diff"]
        print("AGREEMENT WITH VENDOR IV (rows both solved with a parity forward)")
        print(
            f"  n = {v['n']:,}   median |diff| {a['median']:.2e}   p90 {a['p90']:.2e}   p99 {a['p99']:.2e}"
        )
        print(
            f"  within 1e-4: {v['within_1e-4']:.1%}   within 1e-3: {v['within_1e-3']:.1%}"
            f"   within 10% of the quote's own bid-ask vol width: {v.get('within_10pct_of_quote_width', float('nan')):.1%}"
        )
        print(
            "  The two systematic differences are conventions, not errors: the vendor ends AM contracts at"
        )
        print(
            "  the prior day's close (rerun with --am-expiry prior_close), and discounts at Treasury rates"
        )
        print("  where this pipeline uses the rate implied by the quotes themselves.")
        print()

    pc = put_call_agreement(results)
    if not pc.empty:
        print(
            "CALL-PUT IV AGREEMENT (equal under the correct forward; needs no outside reference)"
        )
        print(
            f"  {len(pc)} buckets   median |call IV - put IV| per bucket: median {pc['median_abs_gap'].median():.2e},"
            f" worst {pc['median_abs_gap'].max():.2e}   median signed gap {pc['median_gap'].median():+.2e}"
        )
        print()

    br = bracket_report(results, DEFAULT_CONFIG)
    if br.get("n"):
        print("SOLVER BRACKET")
        print(
            f"  solved s in [{br['s_min']:.4f}, {br['s_max']:.4f}], median {br['s_median']:.4f};"
            f" bracket [{br['s_lo']}, {br['s_hi']}]  headroom x{br['headroom_low']:.0f} below, x{br['headroom_high']:.1f} above"
        )
        print()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        flat = results.copy()
        for col in ("quote_status", "reason", "method"):
            flat[col] = flat[col].astype("string")
        if out.suffix == ".parquet":
            flat.to_parquet(out, index=False)
        else:
            flat.to_csv(out, index=False)
        print(f"wrote {len(flat):,} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
