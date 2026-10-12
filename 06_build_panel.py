#!/usr/bin/env python3
"""
Step 6 - Build the analysis panel.

Creates:
  RDQI            total quality score, 0-22, and RDQI_pct (0-1)
  RDQI_<dim>      subscale per quality dimension
  DQUANT          disclosure quantity (item count) ; ln_words
  RDQI_resid      quality orthogonal to quantity  <-- the "distinct from quantity" test
  L1_*            one-year lags, ready to merge with S&P financials
Also reports Cronbach's alpha for the index and the quality-quantity correlation.

Usage:
    python 06_build_panel.py --out out
    python 06_build_panel.py --out out --sp data/sp_financials.csv
"""
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd


def cronbach_alpha(df):
    k = df.shape[1]
    v = df.var(axis=0, ddof=1)
    t = df.sum(axis=1).var(ddof=1)
    return (k / (k - 1)) * (1 - v.sum() / t) if k > 1 and t > 0 else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--rubric", default="rubric_v2.json")
    ap.add_argument("--min-firm-rdqi", type=float, default=None,
                    help="drop firms whose BEST year never reaches this RDQI score. "
                         "These are non-reserve-reporting firms: the construct does "
                         "not exist for them, so they leave the analysis sample "
                         "(domain exclusion) while remaining in the population.")
    ap.add_argument("--drop-unscored", action="store_true",
                    help="drop firm-years where no item could be scored (RDQI_max = 0)")
    ap.add_argument("--sp", default=None,
                    help="S&P financials CSV with firm_id,year,... to merge")
    args = ap.parse_args()

    out = Path(args.out)
    rubric = json.load(open(args.rubric, encoding="utf-8"))
    items = rubric["items"]

    q = pd.read_csv(out / "quality_raw.csv")
    q = q[q["run"] == 1].drop(columns=["run"])
    qty = pd.read_csv(out / "quantity.csv")
    df = q.merge(qty, on=["firm_id", "year"], how="outer")

    cols = ["q_" + it["id"] for it in items if "q_" + it["id"] in df.columns]
    num = df[cols].apply(pd.to_numeric, errors="coerce")      # "NA" becomes NaN
    maxes = pd.Series({"q_" + it["id"]: it["max"] for it in items})[cols]
    # NA items leave BOTH numerator and denominator, per the rubric NA rule
    df["RDQI"] = num.sum(axis=1, skipna=True)
    df["RDQI_max"] = num.notna().mul(maxes, axis=1).sum(axis=1)
    df["RDQI_pct"] = df["RDQI"] / df["RDQI_max"]

    for dim in sorted({it["dimension"] for it in items}):
        dcols = ["q_" + it["id"] for it in items
                 if it["dimension"] == dim and "q_" + it["id"] in df.columns]
        df["RDQI_" + dim.split()[0].lower()] = num[dcols].sum(axis=1, skipna=True)

    # ---------- domain exclusion ----------
    if args.drop_unscored:
        n0 = len(df)
        df = df[df["RDQI_max"].fillna(0) > 0]
        print(f"Dropped {n0-len(df)} firm-years with no scoreable items "
              f"(missing, not zero)\n")

    if args.min_firm_rdqi is not None:
        best = df.groupby("firm_id")["RDQI"].max()
        keep = best[best >= args.min_firm_rdqi].index
        gone = best[best < args.min_firm_rdqi]
        n0 = len(df)
        df = df[df["firm_id"].isin(keep)]
        print(f"DOMAIN EXCLUSION at RDQI < {args.min_firm_rdqi:g} in every year")
        print(f"  firms removed      : {len(gone)}")
        print(f"  firm-years removed : {n0-len(df)}")
        if len(gone):
            print("  " + ", ".join(f"{int(f)} (max {int(v)})"
                                   for f, v in gone.items()))
        print("  These firms declare no mineral reserves. Record them as a domain")
        print("  exclusion: they stay in the population, so N_h and the stratum")
        print("  weights are unchanged and only the achieved precision falls.\n")

        df["DQUANT"] = df["ln_pages"] if "ln_pages" in df.columns else df["ln_words"]

    # quality net of quantity: residual from RDQI on quantity measures
    sub = df.dropna(subset=["RDQI", "DQUANT", "ln_words"])
    X = np.column_stack([np.ones(len(sub)), sub["DQUANT"], sub["ln_words"]])
    beta, *_ = np.linalg.lstsq(X, sub["RDQI"].values, rcond=None)
    df.loc[sub.index, "RDQI_resid"] = sub["RDQI"].values - X @ beta

    df = df.sort_values(["firm_id", "year"])
    for c in ["RDQI", "RDQI_pct", "RDQI_resid", "DQUANT", "ln_words"]:
        df["L1_" + c] = df.groupby("firm_id")[c].shift(1)

    if args.sp:
        sp = pd.read_csv(args.sp)
        df = df.merge(sp, on=["firm_id", "year"], how="left")
        print(f"Merged S&P financials: {sp.shape[1]-2} extra columns")

    path = out / "panel.csv"
    df.to_csv(path, index=False)

    print(f"Panel: {len(df)} firm-years, {df['firm_id'].nunique()} firms, "
          f"{int(df['year'].min())}-{int(df['year'].max())}\n")
    print(df[["RDQI", "RDQI_max", "RDQI_pct", "DQUANT",
              "ln_words"]].describe().round(3).to_string())
    alpha = cronbach_alpha(num.dropna())
    r = df[["RDQI", "DQUANT"]].corr().iloc[0, 1]
    print(f"\nCronbach's alpha (index reliability): {alpha:.3f}")
    print(f"Correlation quality vs quantity:      {r:.3f}")
    if r > 0.85:
        print("  ** High: the two are not empirically distinct. Tighten the 0/1/2")
        print("     anchors so quality rewards specificity, not presence.")
    print(f"\nSaved -> {path}")


if __name__ == "__main__":
    main()
