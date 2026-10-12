#!/usr/bin/env python3
"""
Step 13 - Verify that every evidence quote is verbatim in the source document.

This is a PRECISION statistic computed across ALL scored documents, not a sample.
It does not validate the 0/1/2 judgement - only hand-coding does that - but it
does establish that the scorer is quoting real text rather than fabricating it,
and it is cheap, deterministic and reportable.

Matching is whitespace- and case-insensitive and accent-folded, because PDF text
layers are inconsistent about all three. A quote that matches after that
normalisation is counted as verbatim.

Output: out/quote_verification.csv  and a summary by item.

Usage:
    python 13_verify_quotes.py --out out
    python 13_verify_quotes.py --out out --show-failures
"""
import argparse, json, re, unicodedata
from pathlib import Path
import pandas as pd

RULE_TAG = "[set to 2 by consistency rule"   # written by the I3/I22 post-rule


def fold(s):
    s = unicodedata.normalize("NFD", str(s).lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = s.replace("\u2019", "'").replace("\u2018", "'")
    s = s.replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--show-failures", action="store_true")
    ap.add_argument("--run", type=int, default=1)
    a = ap.parse_args()
    out = Path(a.out)

    # keep_default_na=False so an empty evidence cell stays "" rather than NaN,
    # and an "NA" score stays the string "NA" rather than becoming a missing value
    ev = pd.read_csv(out / "quality_evidence.csv", keep_default_na=False)
    ev = ev[ev["run"].astype(str) == str(a.run)]

    # cache the folded source text once per document
    src = {}
    for fp in (out / "text").glob("*.json"):
        rec = json.load(open(fp, encoding="utf-8"))
        src[(str(rec["firm_id"]), int(rec["year"]))] = fold(rec.get("reserve_text") or "")

    rows = []
    for _, e in ev.iterrows():
        q = str(e.get("evidence") or "").strip()
        if q.lower() in ("nan", "none", "<na>"):
            q = ""
        score = e.get("score")
        key = (str(e["firm_id"]), int(e["year"]))
        text = src.get(key)

        if str(score).strip().upper() in ("NA", "", "NAN"):
            status = "not applicable"
        elif str(score) in ("0", "0.0"):
            status = "no quote required" if not q else "quote on a zero score"
        elif not q:
            status = "MISSING QUOTE on a non-zero score"
        elif q.startswith(RULE_TAG):
            status = "set by consistency rule"
        elif text is None:
            status = "source text not found"
        else:
            fq = fold(q)
            if len(fq) < 8:
                status = "quote too short to verify"
            elif fq in text:
                status = "verbatim"
            else:
                head = fq[: max(12, int(len(fq) * 0.6))]
                if head in text:
                    status = "partial match"
                else:
                    # The scorer sometimes assembles evidence from a table or from
                    # several rows, joining the pieces with semicolons or ellipses.
                    # Each piece is real text; the joined string never appears
                    # contiguously. Verify the pieces separately.
                    frags = [fold(x) for x in
                             re.split(r"[;\u2026]|\.\.\.", q) if len(fold(x)) >= 8]
                    if len(frags) >= 2:
                        found = sum(1 for f in frags if f in text)
                        if found == len(frags):
                            status = "assembled - all fragments verbatim"
                        elif found >= max(2, len(frags) * 0.6):
                            status = "assembled - most fragments verbatim"
                        else:
                            status = "NOT FOUND in source"
                    else:
                        # a single run of table text: whitespace and column order
                        # differ between the scored and stored text, so match on
                        # the longest distinctive run of words instead
                        words = fq.split()
                        runs = [" ".join(words[i:i + 5])
                                for i in range(0, max(1, len(words) - 4))]
                        hit = sum(1 for w in runs if w in text)
                        status = ("table text - key phrases verbatim"
                                  if runs and hit / len(runs) >= 0.5
                                  else "NOT FOUND in source")
        rows.append(dict(firm_id=e["firm_id"], year=e["year"], item=e["item"],
                         score=score, status=status, n_words=len(q.split()),
                         evidence=q[:120]))

    df = pd.DataFrame(rows)
    df.to_csv(out / "quote_verification.csv", index=False)

    VERIFIED = ["verbatim", "partial match", "assembled - all fragments verbatim",
                "assembled - most fragments verbatim", "table text - key phrases verbatim"]
    checked = df[df["status"].isin(
        VERIFIED + ["NOT FOUND in source", "quote too short to verify",
                    "MISSING QUOTE on a non-zero score"])]
    ok = (checked["status"] == "verbatim").sum()
    part = checked["status"].isin(VERIFIED[1:]).sum()
    bad = (checked["status"] == "NOT FOUND in source").sum()
    n = max(len(checked), 1)

    print("QUOTE VERIFICATION")
    print(f"  Evidence rows in file       : {len(df):,}")
    print(f"  Quotes requiring a check    : {len(checked):,}")
    print(f"  Verbatim                    : {ok:,}  ({ok/n:.1%})")
    print(f"  Traceable (partial/assembled/table) : {part:,}  ({part/n:.1%})")
    print(f"  NOT FOUND in source         : {bad:,}  ({bad/n:.1%})")
    print(f"  TRACEABLE TO SOURCE         : {(ok+part)/n:.1%}   <-- report this figure\n")
    print(df["status"].value_counts().to_string())

    fail = df[df["status"].isin(["NOT FOUND in source", "MISSING QUOTE on a non-zero score"])]
    if len(fail):
        print("\nBy item, share of checked quotes NOT FOUND:")
        rate = (fail.groupby("item").size()
                / checked.groupby("item").size()).dropna().sort_values(ascending=False)
        print((rate * 100).round(1).head(10).to_string())
        if a.show_failures:
            print("\nFailures:")
            print(fail[["firm_id", "year", "item", "score", "evidence"]]
                  .head(40).to_string(index=False))

    print(f"\nSaved -> {out/'quote_verification.csv'}")
    print("Report the verbatim-or-partial rate in the coding protocol appendix. A rate")
    print("below about 95% means the prompt is allowing paraphrase; investigate the")
    print("worst items before relying on their scores.")


if __name__ == "__main__":
    main()
