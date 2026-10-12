#!/usr/bin/env python3
"""
Step 14 - List the machine/human disagreements so they can be diagnosed.

Writes out/disagreements.csv: one row per cell where your hand code differs from
the machine score, with the machine's evidence quote (if any) and its quote
verification status alongside. Sorted worst-gap first.

The purpose is diagnostic, not corrective. If the disagreements cluster on items
whose evidence lives in tables (I2, I8, I18, I19), the cause is the table-stripping
filter in 03_score_quality.py rather than coder judgement, and the fix is to loosen
that filter and re-score.

Usage:
    python 14_disagreements.py --out out
    python 14_disagreements.py --out out --sample 10     # a random ten to check by hand
"""
import argparse
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--sample", type=int, default=None,
                    help="print a random subset for manual checking")
    ap.add_argument("--seed", type=int, default=20260912)
    a = ap.parse_args()
    out = Path(a.out)

    hc = pd.read_csv(out / "handcoding_sheet.csv", keep_default_na=False)
    hc = hc[hc["human_score"].astype(str).str.strip() != ""]
    if "tier" not in hc.columns:
        hc["tier"] = 1
    hc["human"] = pd.to_numeric(hc["human_score"], errors="coerce")
    hc = hc[hc["human"].notna()]

    ev = pd.read_csv(out / "quality_evidence.csv", keep_default_na=False)
    ev = ev[ev["run"].astype(str) == "1"]
    ev["machine"] = pd.to_numeric(ev["score"], errors="coerce")

    # the hand-coding sheet may itself carry an "evidence" or "note" column, so
    # name the machine's quote explicitly rather than letting the merge suffix it
    ev = ev.rename(columns={"evidence": "machine_evidence"})
    df = hc.merge(ev[["firm_id", "year", "item", "machine", "machine_evidence"]],
                  on=["firm_id", "year", "item"], how="inner")
    if "evidence" in df.columns and "note" not in df.columns:
        df = df.rename(columns={"evidence": "note"})   # your own coding note
    df = df[df["machine"].notna()]

    qv = out / "quote_verification.csv"
    if qv.exists():
        v = pd.read_csv(qv, keep_default_na=False)[
            ["firm_id", "year", "item", "status"]]
        df = df.merge(v, on=["firm_id", "year", "item"], how="left")
    else:
        df["status"] = ""

    dis = df[df["human"] != df["machine"]].copy()
    dis["gap"] = (dis["human"] - dis["machine"]).abs()
    dis["direction"] = ["machine lower" if h > m else "machine higher"
                        for h, m in zip(dis["human"], dis["machine"])]
    cols = ["tier", "firm_id", "year", "item", "label", "human", "machine",
            "gap", "direction", "status", "note", "machine_evidence"]
    dis = dis[[c for c in cols if c in dis.columns]].sort_values(
        ["gap", "item", "firm_id"], ascending=[False, True, True])
    dis.to_csv(out / "disagreements.csv", index=False)

    print(f"Disagreements: {len(dis)} of {len(df)} coded cells "
          f"({len(dis)/max(len(df),1):.1%})\n")
    print("By item:")
    t = dis.groupby("item").agg(n=("gap", "size"), mean_gap=("gap", "mean"))
    t["share_of_coded"] = t["n"] / df.groupby("item").size()
    print(t.sort_values("n", ascending=False).round(3).to_string())

    print("\nBy direction:")
    print(dis["direction"].value_counts().to_string())

    if "status" in dis.columns and dis["status"].astype(str).str.len().sum():
        print("\nQuote verification status of the machine score in disagreements:")
        print(dis["status"].value_counts().to_string())

    blank = (dis["machine_evidence"].astype(str).str.strip() == "").sum()
    print(f"\nDisagreements where the machine quoted NOTHING: {blank} "
          f"({blank/max(len(dis),1):.0%})")
    print("  A high share here means the machine did not SEE the evidence, which")
    print("  points at the text it was sent rather than at its judgement. Check")
    print("  whether that evidence sits in a numeric table in the source PDF.")

    if a.sample:
        s = dis.sample(min(a.sample, len(dis)), random_state=a.seed)
        print(f"\n--- {len(s)} cells to check by hand ---")
        for _, r in s.iterrows():
            print(f"\n{r['firm_id']}_{r['year']}.pdf   {r['item']}  "
                  f"(you {int(r['human'])}, machine {int(r['machine'])})")
            print(f"   item     : {r.get('label','')}")
            print(f"   machine  : {str(r['machine_evidence'])[:150] or '(no quote)'}")
            print(f"   your note: {str(r.get('note',''))[:120]}")
        print("\nFor each: open the PDF, find the evidence you used, and record")
        print("whether it sits inside a numeric table. That answers the question.")

    print(f"\nSaved -> {out/'disagreements.csv'}")


if __name__ == "__main__":
    main()
