#!/usr/bin/env python3
"""
Step 8 - Extract financial statement variables from annual report PDFs.

USE THIS AS A GAP-FILLER, NOT A PRIMARY SOURCE.
S&P Capital IQ figures are standardised and restated; annual reports are
as-reported. Mixing them silently makes Tobin's Q incomparable across firms.
Extract from PDFs only where S&P has no value, and keep the source flag.

What it does
  * skips documents with no financial statements (standalone reserve statements)
  * finds the statement pages, sends only those
  * captures currency AND units, and normalises to absolute units
  * requires a verbatim evidence quote for every figure
  * caches on the prompt hash, so re-runs are free

Variables: total_assets, total_debt, gross_ppe, ebit, goodwill,
           cash_and_equivalents, shares_outstanding, revenue

Usage:
    python 08_extract_financials.py --out out --mock          # test, no API
    python 08_extract_financials.py --out out --limit 3       # pilot
    python 08_extract_financials.py --out out
    python 08_extract_financials.py --out out --gaps gaps.csv # only listed firm-years
"""
import argparse, csv, hashlib, json, os, re, sys, time, unicodedata
from pathlib import Path
import pandas as pd

MAX_CHARS = 90_000

# Pages that look like primary statements or the notes we need
STATEMENT_RE = re.compile(
    r"statements?\s+of\s+financial\s+position|balance\s+sheet|"
    r"statements?\s+of\s+(comprehensive\s+)?(income|profit)|income\s+statement|"
    r"statements?\s+of\s+cash\s+flows?|property,?\s+plant\s+and\s+equipment|"
    r"goodwill|borrowings|issued\s+capital|share\s+capital|"
    r"[ée]tat\s+de\s+la\s+situation\s+financi[èe]re|bilan|compte\s+de\s+r[ée]sultat|"
    r"immobilisations\s+corporelles|[ée]carts?\s+d[’']acquisition|emprunts",
    re.I)
# A document with no statements at all - typical standalone reserve report
HAS_STATEMENTS_RE = re.compile(
    r"total\s+assets|total\s+equity|balance\s+sheet|financial\s+position|"
    r"total\s+de\s+l[’']actif|capitaux\s+propres", re.I)

VARS = [
    ("revenue", "Revenue / turnover from continuing operations. NOT other income, "
                "NOT gross profit."),
    ("ebit", "Operating profit or EBIT, BEFORE finance costs and tax. If only profit "
             "before tax is shown, report that and say so in units_note."),
    ("total_assets", "Total assets from the statement of financial position."),
    ("total_debt", "Interest-bearing borrowings, current PLUS non-current. EXCLUDE lease "
                   "liabilities unless the company reports them within borrowings; if "
                   "leases are included say so in units_note. NOT total liabilities."),
    ("gross_ppe", "Property, plant and equipment at COST or gross carrying amount, "
                  "BEFORE accumulated depreciation. Usually in the PPE note, not the "
                  "balance sheet face. If only the net figure is available, return null."),
    ("goodwill", "Goodwill only. NOT total intangible assets. Return 0 if the company "
                 "states it has no goodwill; null if simply not mentioned."),
    ("cash_and_equivalents", "Cash and cash equivalents from the balance sheet. EXCLUDE "
                             "restricted cash held separately."),
    ("shares_outstanding", "Ordinary shares in issue at the reporting date, EXCLUDING "
                           "treasury shares. A share COUNT, not a value."),
]

SYSTEM = """You extract financial statement figures for an accounting research dataset.
You are literal and conservative: report only figures you can see in the text provided.
Never estimate, never derive a figure that is not stated unless the instruction says to
add two stated components. If a figure is not present, return null.

Return ONLY a JSON object, no preamble, no markdown fences."""

USER = """Extract the figures below from this annual report extract.

COMPANY {firm_id}, FISCAL YEAR {year}

VARIABLES
{varblock}

RULES
- Report the figure AS PRINTED, without rescaling. Put the scale in "units".
- "currency" is the ISO code printed in the statements (ZAR, USD, GBP, AUD, CAD, XOF...).
- "units" is one of: ones, thousands, millions, billions - as the column heading states
  (e.g. "R'000" is thousands, "US$m" is millions).
- Every non-null figure needs a verbatim quote of 3-12 words copied from the text,
  containing the line-item label. If you cannot quote it, return null.
- Use the CURRENT year column, not the comparative. If you cannot tell which column is
  current, return null for that figure rather than guessing.
- Negative figures: keep the sign (a loss is negative).
- Do not convert currencies. Do not annualise. Do not sum across segments.

TEXT
<<<
{text}
>>>

Return JSON exactly:
{{"currency": "XXX", "units": "thousands", "fiscal_year_end": "YYYY-MM-DD or null",
  "statements_present": true,
  "figures": {{ {shape} }} }}
each figure: {{"v": <number or null>, "e": "<verbatim quote or empty>", "note": ""}}
"""

SCALE = {"ones": 1, "unit": 1, "units": 1, "thousands": 1_000, "thousand": 1_000,
         "millions": 1_000_000, "million": 1_000_000,
         "billions": 1_000_000_000, "billion": 1_000_000_000}


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def statement_pages(full_text):
    """Keep pages that look like primary statements or the relevant notes."""
    pages = full_text.split("\n\n")
    keep = [p for p in pages if STATEMENT_RE.search(p)]
    out = "\n\n[...]\n\n".join(keep)
    return out[:MAX_CHARS]


def parse_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1])


def call_model(client, model, prompt, tries=4):
    for a in range(tries):
        try:
            kw = dict(model=model, max_tokens=2000, system=SYSTEM,
                      messages=[{"role": "user", "content": prompt}])
            try:
                msg = client.messages.create(temperature=0, **kw)
            except TypeError:
                msg = client.messages.create(**kw)
            return parse_json("".join(b.text for b in msg.content if b.type == "text"))
        except Exception:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))


def mock(text):
    figs = {}
    for name, _ in VARS:
        m = re.search(r"\b\d[\d, ]{4,}\b", text)
        figs[name] = {"v": float(m.group(0).replace(",", "").replace(" ", "")) if m else None,
                      "e": "mock", "note": ""}
    return {"currency": "ZAR", "units": "thousands", "fiscal_year_end": None,
            "statements_present": True, "figures": figs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--gaps", default=None,
                    help="CSV with firm_id,year - extract only these firm-years")
    args = ap.parse_args()

    out = Path(args.out)
    cache = out / "cache_fin"; cache.mkdir(parents=True, exist_ok=True)

    wanted = None
    if args.gaps:
        wanted = set()
        with open(args.gaps, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                wanted.add((str(row["firm_id"]).strip(), int(float(row["year"]))))

    client = None
    if not args.mock:
        try:
            import anthropic
        except ImportError:
            sys.exit("pip install anthropic  (or run with --mock)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first")
        client = anthropic.Anthropic()

    varblock = "\n".join(f"- {n}: {d}" for n, d in VARS)
    shape = ", ".join(f'"{n}": {{"v":..,"e":".."}}' for n, _ in VARS)

    rows, skipped = [], 0
    files = sorted((out / "text").glob("*.json"))
    done = 0
    for fp in files:
        if args.limit and done >= args.limit:
            break
        rec = json.load(open(fp, encoding="utf-8"))
        key = (str(rec["firm_id"]), int(rec["year"]))
        if wanted is not None and key not in wanted:
            continue

        full = rec.get("full_text") or ""
        if not HAS_STATEMENTS_RE.search(full):
            skipped += 1
            rows.append(dict(firm_id=rec["firm_id"], year=rec["year"],
                             status="no financial statements in document"))
            print(f"{rec['firm_id']} {rec['year']}: no statements - skipped")
            continue

        text = statement_pages(full)
        if len(text) < 500:
            skipped += 1
            rows.append(dict(firm_id=rec["firm_id"], year=rec["year"],
                             status="statement pages not located"))
            continue

        prompt = USER.format(firm_id=rec["firm_id"], year=rec["year"],
                             varblock=varblock, text=text, shape=shape)
        h = hashlib.sha256(f"{args.model}|{prompt}".encode()).hexdigest()[:24]
        cf = cache / f"{h}.json"
        if cf.exists():
            res = json.load(open(cf, encoding="utf-8"))
        else:
            res = mock(text) if args.mock else call_model(client, args.model, prompt)
            json.dump(res, open(cf, "w", encoding="utf-8"))

        mult = SCALE.get(str(res.get("units", "")).lower(), None)
        row = dict(firm_id=rec["firm_id"], year=rec["year"], status="ok",
                   currency=res.get("currency"), units=res.get("units"),
                   scale_multiplier=mult, fiscal_year_end=res.get("fiscal_year_end"),
                   source="pdf")
        for name, _ in VARS:
            f_ = res.get("figures", {}).get(name, {}) or {}
            v = f_.get("v")
            row[name] = v
            # shares outstanding is a count, never rescaled by the statement units
            row[name + "_abs"] = (v * mult if v is not None and mult
                                  and name != "shares_outstanding" else v)
            row[name + "_evidence"] = f_.get("e", "")
        rows.append(row)
        done += 1
        print(f"{rec['firm_id']} {rec['year']}: {row['currency']} {row['units']} "
              f"assets={row.get('total_assets')}")

    df = pd.DataFrame(rows)
    path = out / "financials_pdf.csv"
    df.to_csv(path, index=False)
    print(f"\nExtracted {done}, skipped {skipped}  ->  {path}")
    print("Check 'currency' and 'units' before using ANY figure. A wrong scale is the")
    print("single most damaging error in this file and it looks entirely plausible.")


if __name__ == "__main__":
    main()
