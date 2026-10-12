#!/usr/bin/env python3
"""
Step 18 - Build the analysis dataset.

Joins the disclosure panel, the valuation block, the controls block, the auditor
coding and the extracted reserve tonnage, then constructs every variable the
specification needs. Writes CSV for Python and .dta for Stata.

Inputs (all keyed on firm_id x year)
    out/panel.csv                   RDQI, dimensions, DQUANT
    out/sp_valuation.csv            Block A: market cap, shares, book equity, EPS
    SP_Blocks_BC.xlsx               Block B: total assets, total debt, capex, auditor
    out/auditor.csv                 Big 4 coding from the annual reports
    out/reserves_firm_year.csv      reserve tonnage, where extractable (optional)
    revised_frame.csv               group_id for clustering, if present (optional)

Usage:
    python 18_build_analysis_dataset.py
    python 18_build_analysis_dataset.py --blockb SP_Blocks_BC.xlsx --out out
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def key(d):
    d = d.rename(columns={"entity_id": "firm_id"}).copy()
    d["firm_id"] = pd.to_numeric(d["firm_id"], errors="coerce")
    d = d[d["firm_id"].notna()]
    d["firm_id"] = d["firm_id"].astype(int)
    if "year" in d.columns:
        d["year"] = pd.to_numeric(d["year"], errors="coerce")
        d = d[d["year"].notna()]
        d["year"] = d["year"].astype(int)
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--blockb", default="SP_Blocks_BC.xlsx")
    ap.add_argument("--frame", default="revised_frame.csv")
    a = ap.parse_args()
    out = Path(a.out)

    panel = key(pd.read_csv(out / "panel.csv"))
    val = key(pd.read_csv(out / "sp_valuation.csv"))
    ctrl = key(pd.read_excel(a.blockb, sheet_name="Block B Controls",
                             header=0, skiprows=[1]))
    print(f"panel {len(panel)}  valuation {len(val)}  controls {len(ctrl)}")

    df = panel.merge(val.drop(columns=[c for c in ["company"] if c in val.columns]),
                     on=["firm_id", "year"], how="left", suffixes=("", "_v"))
    df = df.merge(ctrl[["firm_id", "year", "total_assets", "total_debt",
                        "capex", "auditor_sp"]],
                  on=["firm_id", "year"], how="left")

    # ---- Big 4: annual-report coding first, S&P as the fallback ----
    BIG4 = ("deloitte", "ernst", "kpmg", "pricewaterhouse", "pwc", "ey ")
    ap_file = out / "auditor.csv"
    if ap_file.exists():
        aud = key(pd.read_csv(ap_file))
        df = df.merge(aud[["firm_id", "year", "auditor", "big4"]],
                      on=["firm_id", "year"], how="left")
    else:
        df["auditor"], df["big4"] = np.nan, np.nan
    def big4_from_name(s):
        s = "" if s is None else str(s).strip().lower()
        if s in ("", "nan", "none"):
            return np.nan
        return 1 if any(b in s for b in BIG4) else 0

    sp_b4 = df["auditor_sp"].map(big4_from_name)
    filled = df["big4"].isna() & sp_b4.notna()
    df["big4_source"] = np.where(df["big4"].notna(), "annual report",
                          np.where(filled, "S&P", "missing"))
    df["big4"] = df["big4"].fillna(sp_b4)
    print(f"Big 4 coded: {int(df['big4'].notna().sum())} of {len(df)} "
          f"({int(filled.sum())} filled from S&P)")
    dis = df[df["big4"].notna() & sp_b4.notna() & (df["big4"] != sp_b4)]
    if len(dis):
        print(f"  NOTE: {len(dis)} firm-years where the annual report and S&P disagree "
              f"on Big 4 status - listed in out/big4_disagreements.csv")
        dis[["firm_id", "year", "auditor", "auditor_sp"]].to_csv(
            out / "big4_disagreements.csv", index=False)

    # ---- reserve tonnage, where it could be extracted ----
    rf = out / "reserves_firm_year.csv"
    if rf.exists():
        res = key(pd.read_csv(rf))
        cols = ["firm_id", "year", "reserve_tonnes_primary", "primary_commodity_fixed",
                "multi_commodity"]
        df = df.merge(res[[c for c in cols if c in res.columns]],
                      on=["firm_id", "year"], how="left")

    # ---- group id for clustering ----
    fr = Path(a.frame)
    if fr.exists():
        f = pd.read_csv(fr)
        idc = "entity_id" if "entity_id" in f.columns else "firm_id"
        if "group_id" in f.columns:
            g = f[[idc, "group_id"]].rename(columns={idc: "firm_id"})
            g["firm_id"] = pd.to_numeric(g["firm_id"], errors="coerce")
            df = df.merge(g, on="firm_id", how="left")
    if "group_id" not in df.columns or df["group_id"].isna().all():
        df["group_id"] = df["firm_id"]
        print("  group_id not found - clustering will fall back to the firm. Add a")
        print("  group_id column to revised_frame.csv so related firms cluster together.")
    else:
        df["group_id"] = df["group_id"].fillna(df["firm_id"])

    # ---------------- construct the variables ----------------
    num = ["market_cap", "shares_outstanding", "book_equity", "eps_before_xo",
           "share_price", "total_assets", "total_debt", "capex"]
    for c in num:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # market cap and book equity are USD millions; shares are millions
    df["MVPS"] = df["market_cap"] / df["shares_outstanding"]
    df["BVPS"] = df["book_equity"] / df["shares_outstanding"]
    df["EPS"] = df["eps_before_xo"]
    df["LOSS"] = (df["EPS"] < 0).astype(int)
    df["LOSS_EPS"] = df["LOSS"] * df["EPS"]
    df["NEGBV"] = (df["book_equity"] < 0).astype(int)

    df["SIZE"] = np.log(df["total_assets"].where(df["total_assets"] > 0))
    df["LEV"] = df["total_debt"] / df["total_assets"]
    # S&P reports capex as a cash OUTFLOW, so it is negative; use the magnitude
    df["CAPEXI"] = df["capex"].abs() / df["total_assets"]

    if "reserve_tonnes_primary" in df.columns:
        df["RESTON"] = np.log((df["reserve_tonnes_primary"] /
                               df["shares_outstanding"]).where(
                                   df["reserve_tonnes_primary"] > 0))

    # lags, within firm
    df = df.sort_values(["firm_id", "year"])
    for src, dst in [("RDQI_pct", "RDQI_L1"), ("DQUANT", "DQUANT_L1"),
                     ("RDQI_resid", "RDQIRES_L1")]:
        if src in df.columns:
            df[dst] = df.groupby("firm_id")[src].shift(1)

    # winsorise the financial variables at 1% / 99%
    for v in ["MVPS", "BVPS", "EPS", "LEV", "CAPEXI"]:
        if v in df.columns:
            lo, hi = df[v].quantile([0.01, 0.99])
            df[v + "_w"] = df[v].clip(lo, hi)

    # ---------------- report and save ----------------
    core = ["MVPS", "BVPS", "EPS", "RDQI_L1", "DQUANT_L1", "SIZE", "LEV", "big4"]
    print("\nCoverage of the estimation variables:")
    for c in core:
        if c in df.columns:
            print(f"  {c:<12}{int(df[c].notna().sum()):>5} of {len(df)}"
                  f"   ({df[c].notna().mean():.0%})")
    est = df.dropna(subset=[c for c in core if c in df.columns])
    print(f"\nComplete cases for the main model: {len(est)} firm-years, "
          f"{est.firm_id.nunique()} firms")
    if "RESTON" in df.columns:
        print(f"With a reserve control (robustness subsample): "
              f"{int(df['RESTON'].notna().sum())}")

    print("\nDescriptives:")
    show = [c for c in ["MVPS", "BVPS", "EPS", "RDQI_pct", "DQUANT", "SIZE",
                        "LEV", "CAPEXI"] if c in df.columns]
    print(df[show].describe().round(3).T.to_string())

    df.to_csv(out / "analysis.csv", index=False)
    try:
        slim = df.copy()
        slim.columns = [re_ok(c) for c in slim.columns]
        slim.to_stata(out / "analysis.dta", write_index=False, version=118)
        print(f"\nSaved -> {out/'analysis.csv'} and {out/'analysis.dta'}")
    except Exception as e:
        print(f"\nSaved -> {out/'analysis.csv'}")
        print(f"(Stata export skipped: {e})")


def re_ok(c):
    """Stata variable names: letters, digits and underscore, max 32 characters."""
    import re
    s = re.sub(r"[^0-9a-zA-Z_]", "_", str(c))
    if s and s[0].isdigit():
        s = "v" + s
    return s[:32]


if __name__ == "__main__":
    main()
