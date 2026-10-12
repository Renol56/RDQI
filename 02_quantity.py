#!/usr/bin/env python3
"""
Step 2 - Disclosure QUANTITY measures + keyword screen.

Produces, per firm-year:
  n_words, ln_words, n_tables_approx, n_numbers, doc_type, standalone dummy
  kw_<item> = 1 if the item's keywords appear at all  (presence only)
  kw_items  = count of the 11 items detected          <-- QUANTITY control
This is deliberately separate from Step 3, which measures QUALITY on 0/1/2.

Usage:
    python 02_quantity.py --out out
"""
import argparse, json, re, unicodedata
from pathlib import Path
import math
import pandas as pd

NUM_RE = re.compile(r"\b\d[\d,\.]*\b")
ROW_RE = re.compile(r"^.*?\d[\d,\.]*\s+\d[\d,\.]*\s+\d[\d,\.]*.*$", re.M)  # table-ish rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--rubric", default="rubric.json")
    args = ap.parse_args()

    rubric = json.load(open(args.rubric, encoding="utf-8"))
    items = rubric["items"]
    out = Path(args.out)

    rows = []
    for fp in sorted((out / "text").glob("*.json")):
        rec = json.load(open(fp, encoding="utf-8"))
        text = rec["reserve_text"] or ""
        low = "".join(c for c in unicodedata.normalize("NFD", text.lower())
                      if unicodedata.category(c) != "Mn")
        words = len(text.split())

        row = dict(firm_id=rec["firm_id"], year=rec["year"], file=rec["file"],
                   doc_type=rec["doc_type"],
                   standalone=int(rec["doc_type"] == "standalone"),
                   section_found=int(rec["section_found"]),
                                      n_pages_doc=rec["n_pages"],
                   n_section_pages=(rec["section_end_page"] - rec["section_start_page"] + 1)
                                   if rec["section_found"] else 0,
                   ln_pages=math.log((rec["section_end_page"] - rec["section_start_page"] + 2)
                                     if rec["section_found"] else 1),
                   n_words=words,
                   ln_words=math.log(words + 1),
                   n_numbers=len(NUM_RE.findall(text)),
                   n_table_rows=len(ROW_RE.findall(text)))

        hits = 0
        for it in items:
            found = any(
                "".join(c for c in unicodedata.normalize("NFD", k.lower())
                        if unicodedata.category(c) != "Mn") in low
                for k in it["keywords"])
            row[f"kw_{it['id']}"] = int(found)
            hits += int(found)
        row["kw_items"] = hits
        rows.append(row)

    df = pd.DataFrame(rows).sort_values(["firm_id", "year"])
    path = out / "quantity.csv"
    df.to_csv(path, index=False)
    print(df[["firm_id", "year", "n_words", "kw_items", "standalone"]].to_string(index=False))
    print(f"\nSaved {len(df)} firm-years -> {path}")


if __name__ == "__main__":
    main()
