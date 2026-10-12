#!/usr/bin/env python3
"""
Step 9 - Audit the S&P extract, then fill gaps from the PDF extraction.

Runs three checks on the S&P file BEFORE merging anything:
  1. DUPLICATED VALUES   the same figure repeated across consecutive years for a
                         firm - the signature of a plugin refresh that pasted one
                         year's numbers down a column
  2. MISSING             which firm-years and variables have no value
  3. DISAGREEMENT        where both S&P and the PDF have a value, how far apart

Then writes a merged panel with a source flag on every filled cell, so you can
report how many observations came from each source and run a robustness test
excluding PDF-sourced ones.

Usage:
    python 09_reconcile_financials.py --sp data/sp_financials.csv --out out
    python 09_reconcile_financials.py --sp data/sp_financials.csv --out out --tol 0.02
    python 09_reconcile_financials.py --sp data/sp_financials.csv --out out --gaps-only
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

VARS = ["revenue", "ebit", "total_assets", "total_debt", "gross_ppe",
        "goodwill", "cash_and_equivalents", "shares_outstanding"]


def norm_id(v):
    s = str(v).strip()
    try:
        f = float(s)
        if f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sp", required=True, help="your S&P extract CSV")
    ap.add_argument("--out", default="out")
    ap.add_argument("--tol", type=float, default=0.05,
                    help="relative difference above which S&P and PDF are flagged")
    ap.add_argument("--gaps-only", action="store_true",
                    help="write gaps.csv and stop, for feeding to 08 --gaps")
    a = ap.parse_args()

    out = Path(a.out)
    sp = pd.read_csv(a.sp)
    sp.columns = [c.strip().lower().replace(" ", "_") for c in sp.columns]
    sp["firm_id"] = sp["firm_id"].map(norm_id)
    sp["year"] = sp["year"].astype(float).astype(int)
    present = [v for v in VARS if v in sp.columns]
    if not present:
        raise SystemExit(f"None of {VARS} found in {a.sp}. Columns: {list(sp.columns)}")

    # ---------- check 1: duplicated values down a column ----------
    print("=" * 66)
    print("CHECK 1  DUPLICATED VALUES (same figure repeated in consecutive years)")
    dups = []
    sp = sp.sort_values(["firm_id", "year"])
    for fid, g in sp.groupby("firm_id"):
        for v in present:
            s = g[v]
            same = s.eq(s.shift()) & s.notna() & (s != 0)
            run = 0
            for (idx, flag), yr in zip(same.items(), g["year"]):
                run = run + 1 if flag else 0
                if run >= 1:
                    dups.append(dict(firm_id=fid, variable=v, year=yr,
                                     value=s.loc[idx], repeats=run + 1))
    dd = pd.DataFrame(dups)
    if dd.empty:
        print("  none found")
    else:
        worst = (dd.groupby(["firm_id", "variable"])["repeats"].max()
                   .sort_values(ascending=False).head(15))
        print(f"  {len(dd)} repeated cells across "
              f"{dd.groupby(['firm_id','variable']).ngroups} firm-variable series")
        print("  worst runs (a run of 3+ is almost certainly a refresh error):")
        print(worst.to_string())
        dd.to_csv(out / "sp_duplicate_flags.csv", index=False)
        print(f"  -> {out/'sp_duplicate_flags.csv'}")

    # ---------- check 2: missing ----------
    print("\n" + "=" * 66)
    print("CHECK 2  MISSING VALUES")
    miss = sp[["firm_id", "year"]].copy()
    for v in present:
        miss[v] = sp[v].isna()
    rates = {v: sp[v].isna().mean() for v in present}
    for v, r in sorted(rates.items(), key=lambda x: -x[1]):
        print(f"  {v:<24}{sp[v].isna().sum():>6} missing  ({r:>6.1%})")

    gaps = []
    for _, row in sp.iterrows():
        for v in present:
            if pd.isna(row[v]):
                gaps.append(dict(firm_id=row["firm_id"], year=row["year"], variable=v))
    gp = pd.DataFrame(gaps)
    if not gp.empty:
        gp.drop_duplicates(["firm_id", "year"])[["firm_id", "year"]].to_csv(
            out / "gaps.csv", index=False)
        print(f"\n  {len(gp)} missing cells over "
              f"{gp[['firm_id','year']].drop_duplicates().shape[0]} firm-years")
        print(f"  -> {out/'gaps.csv'}  (feed to: 08_extract_financials.py --gaps)")
    if a.gaps_only:
        return

    # ---------- check 3: disagreement, then merge ----------
    fp = out / "financials_pdf.csv"
    if not fp.exists():
        print(f"\nNo {fp} yet - run 08_extract_financials.py first. Stopping here.")
        return
    pdf = pd.read_csv(fp)
    pdf = pdf[pdf["status"] == "ok"].copy()
    pdf["firm_id"] = pdf["firm_id"].map(norm_id)
    pdf["year"] = pdf["year"].astype(float).astype(int)

    print("\n" + "=" * 66)
    print(f"CHECK 3  S&P vs PDF, flagged above {a.tol:.0%} relative difference")
    m = sp.merge(pdf, on=["firm_id", "year"], how="left", suffixes=("", "_pdf"))
    diffs = []
    for v in present:
        col = v + "_abs"
        if col not in m.columns:
            continue
        both = m[v].notna() & m[col].notna() & (m[v] != 0)
        rel = (m.loc[both, col] - m.loc[both, v]).abs() / m.loc[both, v].abs()
        bad = rel[rel > a.tol]
        print(f"  {v:<24}{both.sum():>5} comparable, {len(bad):>4} disagree")
        for idx in bad.index:
            diffs.append(dict(firm_id=m.at[idx, "firm_id"], year=m.at[idx, "year"],
                              variable=v, sp=m.at[idx, v], pdf=m.at[idx, col],
                              rel_diff=round(rel[idx], 4),
                              currency=m.at[idx, "currency"], units=m.at[idx, "units"],
                              evidence=m.at[idx, v + "_evidence"]))
    if diffs:
        pd.DataFrame(diffs).to_csv(out / "sp_pdf_disagreements.csv", index=False)
        print(f"  -> {out/'sp_pdf_disagreements.csv'}")
        print("  A wrong 'units' value is the usual cause of a 1000x difference.")

    # ---------- merge: S&P wins, PDF fills gaps ----------
    for v in present:
        col = v + "_abs"
        if col not in m.columns:
            m[v + "_source"] = np.where(m[v].notna(), "sp", "missing")
            continue
        fill = m[v].isna() & m[col].notna()
        m[v + "_source"] = np.where(m[v].notna(), "sp",
                             np.where(fill, "pdf", "missing"))
        m[v] = m[v].where(m[v].notna(), m[col])

    keep = ["firm_id", "year"] + present + [v + "_source" for v in present]
    merged = m[keep]
    merged.to_csv(out / "financials_merged.csv", index=False)

    print("\n" + "=" * 66)
    print("MERGED PANEL")
    for v in present:
        c = merged[v + "_source"].value_counts()
        print(f"  {v:<24}sp={c.get('sp',0):>5}  pdf={c.get('pdf',0):>4}  "
              f"missing={c.get('missing',0):>5}")
    print(f"\n  -> {out/'financials_merged.csv'}")
    print("  Report the pdf counts in your data section, and re-run the main")
    print("  regressions on sp-only observations as a robustness test.")


if __name__ == "__main__":
    main()
