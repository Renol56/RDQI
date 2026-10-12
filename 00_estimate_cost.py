#!/usr/bin/env python3
"""
Estimate API cost BEFORE you run the scorer.

Counts the characters actually sent per document (after condensing) and converts
to an approximate token count and cost. Run this after 01_extract.py.

Usage:
    python 00_estimate_cost.py --out out
    python 00_estimate_cost.py --out out --in-price 3 --out-price 15
"""
import argparse, importlib.util, json, sys
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "scorer", Path(__file__).with_name("03_score_quality.py"))
_scorer = importlib.util.module_from_spec(_spec)
sys.argv, _argv = [sys.argv[0]], sys.argv
_spec.loader.exec_module(_scorer)
sys.argv = _argv

MAX_CHARS = 120_000        # must match the cap in 03_score_quality.py
CHARS_PER_TOKEN = 3.0      # conservative for dense report text
OUT_TOKENS = 1_600         # trimmed rubric JSON response per document


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--rubric", default="rubric_v2.json")
    ap.add_argument("--in-price", type=float, default=1.0,
                    help="USD per million INPUT tokens for your chosen model")
    ap.add_argument("--out-price", type=float, default=5.0,
                    help="USD per million OUTPUT tokens for your chosen model")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--n-docs", type=int, default=None,
                    help="project the cost to a full panel of this many firm-years")
    args = ap.parse_args()

    files = sorted((Path(args.out) / "text").glob("*.json"))
    if not files:
        raise SystemExit("No extracted text found. Run 01_extract.py first.")

    rubric = json.load(open(args.rubric, encoding="utf-8"))
    sent, condensed, empty, raw = 0, 0, 0, 0
    for fp in files:
        t = json.load(open(fp, encoding="utf-8"))["reserve_text"] or ""
        if len(t) < 200:
            empty += 1
            continue
        raw += len(t)
        kept, was = _scorer.trim(t, rubric)
        condensed += int(was)
        sent += len(kept)

    docs = len(files) - empty
    in_tok = sent / CHARS_PER_TOKEN * args.runs
    out_tok = docs * OUT_TOKENS * args.runs
    cost = in_tok / 1e6 * args.in_price + out_tok / 1e6 * args.out_price

    print(f"Documents found        : {len(files)}")
    print(f"  scoreable            : {docs}")
    print(f"  too short / skipped  : {empty}")
    print(f"  trimmed              : {condensed}")
    print(f"  chars before trim    : {raw:,}")
    print(f"  chars after trim     : {sent:,}  ({sent/raw:.0%} of original)" if raw else "")
    print(f"Runs per document      : {args.runs}")
    print()
    print(f"Approx input tokens    : {in_tok:,.0f}")
    print(f"Approx output tokens   : {out_tok:,.0f}")
    print(f"ESTIMATED COST         : US$ {cost:,.2f}")
    if args.n_docs and docs:
        print(f"PROJECTED for {args.n_docs:,} docs : "
              f"US$ {cost / docs * args.n_docs:,.2f}")
    print()
    print("Prices default to $1 / $5 per million tokens (Haiku 4.5). Check the current rate")
    print("for your model at https://platform.claude.com/docs/en/about-claude/pricing")
    print("and pass --in-price / --out-price to match.")
    if docs:
        print(f"\nPer document: ~{in_tok/docs/args.runs:,.0f} input tokens, "
              f"~US$ {cost/docs/args.runs:.3f}")


if __name__ == "__main__":
    main()
