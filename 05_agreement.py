#!/usr/bin/env python3
"""
Step 5 - Validation statistics: machine vs human agreement (and machine self-consistency).

Reports, overall and per item:
  exact agreement %, Cohen's kappa, quadratic-weighted kappa (ordinal-appropriate),
  and the confusion matrix. Weighted kappa is the one to report for a 0/1/2 scale.

Usage:
    python 05_agreement.py --out out
    python 05_agreement.py --out out --self-consistency
"""
import argparse
from pathlib import Path
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix


def stats(a, b, name):
    exact = (a == b).mean()
    k = cohen_kappa_score(a, b, labels=[0, 1, 2])
    kw = cohen_kappa_score(a, b, labels=[0, 1, 2], weights="quadratic")
    within1 = (abs(a - b) <= 1).mean()
    return dict(unit=name, n=len(a), exact=round(exact, 3), within_1=round(within1, 3),
                kappa=round(k, 3), kappa_weighted=round(kw, 3))


def interpret(k):
    if pd.isna(k):
        return "no variation"   # all scores identical: kappa undefined, read 'exact'
    return ("poor" if k < .20 else "fair" if k < .40 else "moderate" if k < .60
            else "substantial" if k < .80 else "almost perfect")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--self-consistency", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)

    ev = pd.read_csv(out / "quality_evidence.csv")

    if args.self_consistency:
        w = ev.pivot_table(index=["firm_id", "year", "item"], columns="run",
                           values="score")
        w = w.dropna()
        if w.shape[1] < 2:
            raise SystemExit("Need --runs 2 in 03_score_quality.py first.")
        a, b = w[1].astype(int), w[2].astype(int)
        print("MACHINE SELF-CONSISTENCY (run 1 vs run 2)")
        print(pd.DataFrame([stats(a, b, "all items")]).to_string(index=False))
        return

    hc = pd.read_csv(out / "handcoding_sheet.csv", keep_default_na=False)
    hc = hc[hc["human_score"].astype(str).str.strip() != ""]
    if hc.empty:
        raise SystemExit("handcoding_sheet.csv has no human_score values yet.")
    if "tier" not in hc.columns:
        hc["tier"] = 1

    # NA is a valid human code on the NA-permitted items, but kappa is defined on
    # the 0/1/2 scale, so NA rows are counted and reported separately rather than
    # forced into the numeric comparison.
    hc["hs"] = pd.to_numeric(hc["human_score"], errors="coerce")
    na_rows = hc[hc["hs"].isna()]
    hc = hc[hc["hs"].notna()].copy()
    hc["human_score"] = hc["hs"].astype(int)

    m = ev[ev["run"] == 1][["firm_id", "year", "item", "score"]].copy()
    m["score"] = pd.to_numeric(m["score"], errors="coerce")
    m = m[m["score"].notna()]
    m["score"] = m["score"].astype(int)

    all_df = hc.merge(m, on=["firm_id", "year", "item"], how="inner")
    if all_df.empty:
        raise SystemExit("No overlap between hand-coded rows and machine scores.")

    # TIER 3 is a targeted review queue selected on suspected error. Including it
    # would bias the reliability statistic downward, so it is excluded from kappa
    # and reported separately as a correction exercise.
    df = all_df[all_df["tier"].astype(int) != 3]
    t3 = all_df[all_df["tier"].astype(int) == 3]
    if df.empty:
        raise SystemExit("Only tier 3 rows are coded; code tiers 1 and 2 for the kappa.")
    print(f"Coded rows: tier 1-2 = {len(df)}, tier 3 = {len(t3)}, "
          f"NA-coded = {len(na_rows)}\n")

    rows = [stats(df["human_score"].values, df["score"].values, "ALL ITEMS")]
    for item, g in df.groupby("item"):
        rows.append(stats(g["human_score"].values, g["score"].values, item))
    res = pd.DataFrame(rows)
    res["reading"] = res["kappa_weighted"].apply(interpret)

    print(f"MACHINE vs HUMAN  ({df[['firm_id','year']].drop_duplicates().shape[0]} "
          f"firm-years, {len(df)} item scores)\n")
    print(res.to_string(index=False))
    res.to_csv(out / "agreement.csv", index=False)

    print("\nConfusion matrix (rows = human, cols = machine), labels 0/1/2:")
    print(pd.DataFrame(confusion_matrix(df["human_score"], df["score"], labels=[0, 1, 2]),
                       index=["h0", "h1", "h2"], columns=["m0", "m1", "m2"]))

    worst = res[res["unit"] != "ALL ITEMS"].nsmallest(3, "kappa_weighted")
    print("\nWeakest items - discuss these in the reliability appendix:")
    print(worst[["unit", "kappa_weighted", "exact"]].to_string(index=False))

    # ---- tier 1 vs tier 2, and the direction of disagreement ----
    for t in (1, 2):
        g = df[df["tier"].astype(int) == t]
        if len(g) > 20:
            st = stats(g["human_score"].values, g["score"].values, f"tier {t}")
            print(f"\nTier {t} only: n={st['n']}, exact={st['exact']}, "
                  f"weighted kappa={st['kappa_weighted']}")

    hi = (df["score"] > df["human_score"]).sum()
    lo = (df["score"] < df["human_score"]).sum()
    print(f"\nDirection of disagreement: machine HIGHER than coder {hi}, "
          f"LOWER {lo}, equal {len(df)-hi-lo}")
    if hi > lo * 1.5:
        print("  The machine scores systematically high. Report this; it means any")
        print("  measurement error in RDQI is upward, not random, and the bias in the")
        print("  main coefficient is toward zero only if the error is uncorrelated.")
    elif lo > hi * 1.5:
        print("  The machine scores systematically low. Report this.")
    else:
        print("  Disagreement is roughly symmetric, consistent with random coder error.")

    # ---- tier 3: corrections, reported separately ----
    if len(t3):
        agree = (t3["human_score"] == t3["score"]).sum()
        down = (t3["human_score"] < t3["score"]).sum()
        up = (t3["human_score"] > t3["score"]).sum()
        print(f"\nTIER 3 REVIEW QUEUE (untraced evidence, {len(t3)} cells)")
        print(f"  machine score confirmed : {agree}")
        print(f"  corrected DOWNWARD      : {down}")
        print(f"  corrected UPWARD        : {up}")
        t3[["firm_id", "year", "item", "score", "human_score"]].rename(
            columns={"score": "machine", "human_score": "corrected"}).to_csv(
            out / "tier3_corrections.csv", index=False)
        print(f"  -> {out/'tier3_corrections.csv'}")
        print("  Apply these corrections to quality_raw.csv before building the panel,")
        print("  and report the count in the appendix. Do NOT fold them into the kappa.")

    print(f"\nSaved -> {out/'agreement.csv'}")


if __name__ == "__main__":
    main()
