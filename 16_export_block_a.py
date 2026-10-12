#!/usr/bin/env python3
"""
Step 16 - Export Block A (valuation) from the collection template, and audit it.

Reads the Block A sheet straight from SP_Collection_Template.xlsx - skipping the
notes row under the headers - so there is no manual "Save As CSV" step and no
risk of the notes row landing in the data.

Unit convention recorded for this block (applied to every row):
    market_cap, book_equity         USD millions
    shares_outstanding              millions of shares   (verified below)
    eps_before_xo, share_price      USD, absolute (per share)

Audit checks
  1. Entries that are not numbers ("NA", "-", "n/a", text)  -> must be fixed
  2. Missing cells by variable, and a gap list
  3. Values repeated across consecutive years for a firm    -> the signature of
     the earlier plugin-refresh fault
  4. Impossible values: market cap, shares or price <= 0
  5. THE IDENTITY CHECK: market_cap should equal share_price x shares_outstanding.
     A ratio far from 1 exposes a scale error (shares in units rather than
     millions), a currency slip, or a price taken on a different date.

Outputs (in out/):
    sp_valuation.csv        clean extract, keyed firm_id x year, units recorded
    blockA_audit.csv        every flagged cell with the reason
    blockA_gaps.csv         firm-years with any missing Block A value

Usage:
    python 16_export_block_a.py --xlsx SP_Collection_Template.xlsx --out out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SHEET = "Block A Valuation"
VARS = ["market_cap", "shares_outstanding", "book_equity", "eps_before_xo", "share_price"]
MUST_BE_POSITIVE = ["market_cap", "shares_outstanding", "share_price"]
IDENTITY_TOL = (0.80, 1.25)     # market cap vs price x shares, as a ratio


def norm_id(v):
    try:
        return int(float(str(v).strip()))
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="SP_Collection_Template.xlsx")
    ap.add_argument("--out", default="out")
    ap.add_argument("--sheet", default=SHEET)
    ap.add_argument("--shares-in", default="millions",
                    choices=["units", "thousands", "millions"],
                    help="scale shares_outstanding was COLLECTED in; converted to millions")
    ap.add_argument("--frame", default="revised_frame.csv",
                    help="frame file with an exchange column, for the currency diagnosis")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)

    # header in row 1, notes in row 2, data from row 3
    # keep_default_na=False so a typed "NA", "N/A" or "-" survives as text and is
    # flagged, instead of pandas silently reading it as a missing value
    raw = pd.read_excel(a.xlsx, sheet_name=a.sheet, header=0, skiprows=[1], dtype=object,
                        keep_default_na=False, na_values=[""])
    raw.columns = [str(c).strip() for c in raw.columns]
    missing_cols = [c for c in ["entity_id", "year"] + VARS if c not in raw.columns]
    if missing_cols:
        raise SystemExit(f"Columns not found in '{a.sheet}': {missing_cols}")

    df = pd.DataFrame({
        "firm_id": raw["entity_id"].map(norm_id),
        "company": raw.get("company", ""),
        "year": raw["year"].map(norm_id),
    })
    df = df[df["firm_id"].notna() & df["year"].notna()].copy()
    raw = raw.loc[df.index]
    flags = []

    # ---------- 1. non-numeric entries ----------
    for v in VARS:
        col = raw[v]
        txt = col.notna() & (col.astype(str).str.strip() != "")
        num = pd.to_numeric(col.astype(str).str.replace(",", "").str.strip(), errors="coerce")
        bad = txt & num.isna()
        for i in col[bad].index:
            flags.append(dict(firm_id=df.at[i, "firm_id"], year=df.at[i, "year"],
                              variable=v, value=str(col[i]),
                              check="not a number - leave the cell EMPTY instead"))
        df[v] = num

    # convert shares to millions so they match market cap and book equity (USD m)
    scale = {"units": 1e-6, "thousands": 1e-3, "millions": 1.0}[a.shares_in]
    df["shares_outstanding"] = df["shares_outstanding"] * scale
    print(f"shares_outstanding collected in {a.shares_in}, converted to millions "
          f"(x {scale:g})\n")

    # ---------- 2. missing ----------
    print(f"BLOCK A  {len(df)} firm-years, {df.firm_id.nunique()} firms, "
          f"{int(df.year.min())}-{int(df.year.max())}\n")
    print("Missing values:")
    for v in VARS:
        m = df[v].isna().sum()
        print(f"  {v:<20}{m:>5}  ({m/len(df):.1%})")
    gaps = df[df[VARS].isna().any(axis=1)][["firm_id", "company", "year"]]
    gaps.to_csv(out / "blockA_gaps.csv", index=False)

    # ---------- 3. repeated values in consecutive years ----------
    df = df.sort_values(["firm_id", "year"])
    rep = 0
    for fid, g in df.groupby("firm_id"):
        for v in ["market_cap", "book_equity", "share_price", "eps_before_xo"]:
            s = g[v]
            same = s.eq(s.shift()) & s.notna() & (s != 0)
            for i in s[same].index:
                rep += 1
                flags.append(dict(firm_id=fid, year=df.at[i, "year"], variable=v,
                                  value=s[i],
                                  check="same value as the previous year - possible refresh fault"))
    # shares can legitimately be flat, so only flag long runs
    for fid, g in df.groupby("firm_id"):
        s = g["shares_outstanding"]
        run = (s.eq(s.shift()) & s.notna()).astype(int)
        streak = run.groupby((run == 0).cumsum()).cumsum()
        for i in streak[streak >= 3].index:
            flags.append(dict(firm_id=fid, year=df.at[i, "year"],
                              variable="shares_outstanding", value=s[i],
                              check="identical share count 4+ years running - verify"))

    # ---------- 4. impossible values ----------
    for v in MUST_BE_POSITIVE:
        for i in df[df[v] <= 0].index:
            flags.append(dict(firm_id=df.at[i, "firm_id"], year=df.at[i, "year"],
                              variable=v, value=df.at[i, v], check="must be positive"))
    neg_bv = (df["book_equity"] < 0).sum()

    # ---------- 5. identity: market cap = price x shares ----------
    both = df[["market_cap", "share_price", "shares_outstanding"]].notna().all(axis=1) \
        & (df["share_price"] > 0) & (df["shares_outstanding"] > 0)
    implied = df.loc[both, "share_price"] * df.loc[both, "shares_outstanding"]
    ratio = df.loc[both, "market_cap"] / implied
    med = ratio.median() if len(ratio) else np.nan

    print(f"\nIdentity check: market_cap / (share_price x shares_outstanding)")
    print(f"  firm-years testable : {len(ratio)}")
    if len(ratio):
        print(f"  median ratio        : {med:,.4g}")
        if 0.8 < med < 1.25:
            print("  -> shares_outstanding is in MILLIONS, consistent with market cap in USD m")
        elif 8e-7 < med < 1.25e-6:
            print("  -> shares_outstanding looks like ABSOLUTE shares, not millions.")
            print("     Divide by 1,000,000 before computing per-share variables.")
        else:
            print("  -> the scale of market cap, price or shares does not reconcile.")
            print("     Check units before using any per-share variable.")
        # ---- diagnosis: is the mismatch a constant, a per-firm, or a per-year factor?
        tmp = df.loc[both, ["firm_id", "year"]].copy()
        tmp["ratio"] = ratio.values
        fr = Path(a.frame)
        if fr.exists():
            ex = pd.read_csv(fr)
            idc = "entity_id" if "entity_id" in ex.columns else "firm_id"
            if "exchange" in ex.columns:
                ex = ex[[idc, "exchange"]].rename(columns={idc: "firm_id"})
                ex["firm_id"] = ex["firm_id"].map(norm_id)
                tmp = tmp.merge(ex, on="firm_id", how="left")
                print("\n  Median ratio by listing exchange:")
                t = tmp.groupby("exchange")["ratio"].agg(["median", "count"])
                for exn, rw in t.iterrows():
                    print(f"    {str(exn):<10}{rw['median']:>14,.4g}   ({int(rw['count'])} firm-years)")
        within = tmp.groupby("firm_id")["ratio"].agg(
            lambda s: s.max() / s.min() if s.min() > 0 else np.nan)
        stable = (within < 1.5).mean()
        print(f"\n  Firms whose ratio is roughly CONSTANT across years: {stable:.0%}")
        if 0.9 < med < 1.15:
            # stable AT 1 is the correct outcome, not a currency problem
            print("  -> market cap, share price and shares outstanding reconcile.")
            print("     Currency and scale are consistent across the panel. Block A is sound.")
            print("     Rows outside the band below are usually the date offset between the")
            print("     market cap measurement point and the year-end closing price.")
        elif stable > 0.6:
            print("  -> the mismatch is a per-FIRM factor, i.e. a currency: share price")
            print("     (and EPS) are almost certainly in each firm's trading currency.")
            print("     Re-pull share_price and eps_before_xo in USD from Capital IQ.")
        else:
            print("  -> the ratio moves year to year within firms: price and market cap")
            print("     were probably taken on DIFFERENT dates.")
        tmp.to_csv(out / "blockA_identity_by_firm.csv", index=False)

        lo, hi = IDENTITY_TOL
        off = ratio[(ratio < med * lo) | (ratio > med * hi)]
        for i in off.index:
            flags.append(dict(firm_id=df.at[i, "firm_id"], year=df.at[i, "year"],
                              variable="market_cap",
                              value=round(float(ratio[i] / med), 3),
                              check="market cap does not reconcile with price x shares "
                                    "(value = ratio to the sample norm)"))
        print(f"  outside {lo:.0%}-{hi:.0%} of the norm: {len(off)}")

    # ---------- write ----------
    df["currency"] = "USD"
    df["units_monetary"] = "millions"
    df["units_per_share"] = "absolute"
    df["source"] = "S&P Capital IQ Pro"
    df.to_csv(out / "sp_valuation.csv", index=False)

    fl = pd.DataFrame(flags, columns=["firm_id", "year", "variable", "value", "check"])
    fl.to_csv(out / "blockA_audit.csv", index=False)

    print("\nFlags by check:")
    print(fl["check"].value_counts().to_string() if len(fl) else "  none")
    if neg_bv:
        print(f"\nNote: {neg_bv} firm-years have NEGATIVE book equity. That is possible for")
        print("distressed firms and is not an error, but those firm-years need handling")
        print("in the price-level model - report how many and how they are treated.")

    print(f"\nSaved -> {out/'sp_valuation.csv'}")
    print(f"         {out/'blockA_audit.csv'}   ({len(fl)} flags)")
    print(f"         {out/'blockA_gaps.csv'}    ({len(gaps)} firm-years with gaps)")


if __name__ == "__main__":
    main()
