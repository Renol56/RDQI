#!/usr/bin/env python3
"""
Step 4 - Draw a stratified validation subsample and write a blank hand-coding sheet.

Draw the sample BEFORE you look at the machine scores. Stratify by year block and
document type so the validation covers the range of disclosure practice, and keep
the seed in your methodology section so the draw is reproducible.

Usage:
    python 04_sample_for_handcoding.py --out out --frac 0.12 --seed 20260912
"""
import argparse, json
from pathlib import Path
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--rubric", default="rubric.json")
    ap.add_argument("--frac", type=float, default=0.10,
                    help="share of firm-years coded on ALL items (tier 1)")
    ap.add_argument("--focus-frac", type=float, default=0.10,
                    help="further share coded on the judgement items only (tier 2)")
    ap.add_argument("--focus-items", default="I10,I12,I13,I14,I17,I21",
                    help="items where coder judgement is hardest; tier 2 covers only these")
    ap.add_argument("--seed", type=int, default=20260912)
    ap.add_argument("--include-unverified", action="store_true",
                    help="add a tier 3 covering every non-zero score whose evidence "
                         "quote could not be traced to the source (from 13_verify_quotes)")
    ap.add_argument("--unverified-min-score", type=int, default=1,
                    help="only queue unverified scores at or above this value")
    args = ap.parse_args()

    out = Path(args.out)
    rubric = json.load(open(args.rubric, encoding="utf-8"))
    q = pd.read_csv(out / "quantity.csv")

    q["year_block"] = pd.cut(q["year"], [2011, 2015, 2019, 2022, 2026],
                             labels=["2012-15", "2016-19", "2020-22", "2023-25"])
    parts = []
    for _, g in q.groupby(["year_block", "doc_type"], observed=True):
        n = max(1, round(len(g) * args.frac))
        parts.append(g.sample(min(n, len(g)), random_state=args.seed))
    samp = pd.concat(parts)[["firm_id", "year", "file", "doc_type", "year_block"]]
    samp = samp.sort_values(["firm_id", "year"]).reset_index(drop=True)

    # ---- tier 2: a further draw, coded on the judgement-heavy items only ----
    focus = [x.strip() for x in args.focus_items.split(",") if x.strip()]
    remaining = q[~q.set_index(["firm_id", "year"]).index.isin(
        samp.set_index(["firm_id", "year"]).index)]
    parts2 = []
    for _, g in remaining.groupby(["year_block", "doc_type"], observed=True):
        n = max(1, round(len(g) * args.focus_frac))
        parts2.append(g.sample(min(n, len(g)), random_state=args.seed + 1))
    samp2 = pd.concat(parts2)[["firm_id", "year", "file", "doc_type", "year_block"]] \
        if parts2 else samp.iloc[0:0]
    samp2 = samp2.sort_values(["firm_id", "year"]).reset_index(drop=True)

    rows = []
    for _, r in samp.iterrows():                       # tier 1: every item
        for it in rubric["items"]:
            rows.append(dict(tier=1, firm_id=r.firm_id, year=r.year, file=r.file,
                             item=it["id"], label=it["label"], max=it["max"],
                             human_score="", note="", coder=""))
    for _, r in samp2.iterrows():                      # tier 2: judgement items only
        for it in rubric["items"]:
            if it["id"] not in focus:
                continue
            rows.append(dict(tier=2, firm_id=r.firm_id, year=r.year, file=r.file,
                             item=it["id"], label=it["label"], max=it["max"],
                             human_score="", note="", coder=""))
    # ---- tier 3: scores whose evidence could not be traced to the source ----
    n3 = 0
    qv = out / "quote_verification.csv"
    if args.include_unverified and qv.exists():
        v = pd.read_csv(qv, keep_default_na=False)
        bad = v[v["status"].isin(["NOT FOUND in source",
                                  "MISSING QUOTE on a non-zero score"])].copy()
        bad["sc"] = pd.to_numeric(bad["score"], errors="coerce")
        bad = bad[bad["sc"] >= args.unverified_min_score]
        already = {(int(r["firm_id"]), int(r["year"]), r["item"]) for r in rows}
        labels = {it["id"]: (it["label"], it["max"]) for it in rubric["items"]}
        for _, b in bad.iterrows():
            k = (int(b["firm_id"]), int(b["year"]), b["item"])
            if k in already:
                continue
            lab, mx = labels.get(b["item"], ("", ""))
            rows.append(dict(tier=3, firm_id=k[0], year=k[1], file="",
                             item=k[2], label=lab, max=mx,
                             human_score="", note=f"machine={b['score']}; evidence not traced",
                             coder=""))
            already.add(k)
            n3 += 1
    elif args.include_unverified:
        print("NOTE: quote_verification.csv not found - run 13_verify_quotes.py first.\n")

    sheet = pd.DataFrame(rows)
    sheet.to_csv(out / "handcoding_sheet.csv", index=False)
    pd.concat([samp.assign(tier=1), samp2.assign(tier=2)]).to_csv(
        out / "validation_sample.csv", index=False)

    t1_cells = len(samp) * len(rubric["items"])
    t2_cells = len(samp2) * len(focus)
    print(f"TIER 1  all {len(rubric['items'])} items : {len(samp):>3} firm-years "
          f"({len(samp)/len(q):.1%})  -> {t1_cells:>5} cells")
    print(f"TIER 2  {len(focus)} judgement items : {len(samp2):>3} firm-years "
          f"({len(samp2)/len(q):.1%})  -> {t2_cells:>5} cells")
    print(f"        items: {', '.join(focus)}")
    if n3:
        print(f"TIER 3  untraced evidence   : {n3:>5} cells "
              f"(scores \u2265 {args.unverified_min_score} whose quote could not be located)")
    print(f"TOTAL                       : {len(samp)+len(samp2):>3} firm-years, "
          f"{t1_cells+t2_cells+n3:>5} cells   seed={args.seed}")
    single = len(q) * len(rubric["items"])
    print(f"\nAgainst coding every document in full ({single:,} cells), this is "
          f"{(t1_cells+t2_cells)/single:.1%} of the work.")
    print(f"\nBlank sheet -> {out/'handcoding_sheet.csv'}")
    print("Fill human_score (0/1/2, or NA where permitted) WITHOUT looking at the")
    print("machine scores, then run 05_agreement.py. Tier 1 gives the overall kappa;")
    print("tier 2 adds precision on the items where coders actually disagree.")
    if n3:
        print("Tier 3 is a targeted review queue, not a random sample: code these to")
        print("correct the scores, and report them SEPARATELY from the kappa, since")
        print("selecting on suspected error would bias the reliability statistic.")
    print("A second coder on ~15 tier-1 firm-years also gives you inter-human kappa.")


if __name__ == "__main__":
    main()
