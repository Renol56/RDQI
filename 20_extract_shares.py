#!/usr/bin/env python3
"""
Step 20 - Extract ordinary shares in issue from the annual reports.

For firms where the S&P share count cannot be reconciled with the company's own
share register, this reads the share capital note out of the text already
extracted in out/text and returns the count at each fiscal year end.

It asks only for ORDINARY shares, excluding deferred, preference and treasury
shares, and requires a verbatim quote containing the figure.

Usage:
    python 20_extract_shares.py --firms 4347817 --mock
    python 20_extract_shares.py --firms 4347817
    python 20_extract_shares.py --firms 4347817,4349189 --out out

Output: out/shares_extracted.csv
"""
import argparse, hashlib, json, os, re, sys, time
from pathlib import Path
import pandas as pd

MAX_CHARS = 60_000

SHARE_RE = re.compile(
    r"share\s+capital|shares?\s+in\s+issue|ordinary\s+shares?|issued\s+share|"
    r"number\s+of\s+shares|no\.?\s+of\s+shares|allotted|nominal\s+value|"
    r"issued\s+and\s+fully\s+paid", re.I)

SYSTEM = """You read share capital notes from annual reports for an accounting research
dataset. You are literal: report only figures printed in the text. Never estimate and
never convert. If the figure is not present, return null.

Return ONLY a JSON object, no preamble, no markdown fences."""

USER = """From the annual report extract below, find the number of ORDINARY SHARES IN
ISSUE at the reporting date.

COMPANY {firm_id}, FISCAL YEAR {year}

RULES
- ORDINARY shares only. EXCLUDE deferred shares, preference shares and treasury
  shares. Many companies list several classes in one table; take the ordinary line.
- Take the figure AT THE REPORTING DATE (the closing balance), not the opening
  balance and not shares issued during the year.
- Report the number AS PRINTED and state the unit: "units" for a full count such as
  7,945,171,311, "thousands" or "millions" if the column heading says so.
- Give the reporting date exactly as printed, since a company may change its year end.
- Provide a verbatim quote of 4-20 words containing the figure.
- If the note reports a share consolidation or subdivision during the year, say so in
  "note" and give the ratio if printed. This matters: counts before and after a
  consolidation are not comparable.

TEXT
<<<
{text}
>>>

Return JSON exactly:
{{"shares": <number or null>,
  "unit": "units" | "thousands" | "millions" | null,
  "reporting_date": "<as printed, or null>",
  "share_class": "<label of the line taken, e.g. Ordinary 0.1p>",
  "consolidation": false,
  "note": "",
  "evidence": "<verbatim quote containing the figure>"}}
"""


def trim(text, cap=MAX_CHARS):
    """Keep the passages around share capital wording; the note is usually short
    and buried deep in a long report."""
    if len(text) <= cap:
        return text
    parts = re.split(r"\n", text)
    keep, out = [], 0
    for i, p in enumerate(parts):
        if SHARE_RE.search(p):
            lo, hi = max(0, i - 3), min(len(parts), i + 12)
            keep.append("\n".join(parts[lo:hi]))
            out += len(keep[-1])
            if out > cap:
                break
    joined = "\n[...]\n".join(keep)
    return (joined if len(joined) > 500 else text)[:cap]


def parse_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0:
        raise ValueError("no JSON in response")
    return json.loads(raw[i:j + 1] if j > i else raw[i:])


def call_model(client, model, prompt, tries=4):
    for a in range(tries):
        try:
            kw = dict(model=model, max_tokens=1200, system=SYSTEM,
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
    m = re.search(r"([\d][\d,\. ]{6,})\s*(?:ordinary)?", text)
    return {"shares": float(re.sub(r"[ ,]", "", m.group(1))) if m else None,
            "unit": "units", "reporting_date": None, "share_class": "mock",
            "consolidation": False, "note": "mock", "evidence": m.group(0) if m else ""}


SCALE = {"units": 1.0, "thousands": 1e3, "millions": 1e6}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--firms", required=True, help="comma-separated entity ids")
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--mock", action="store_true")
    a = ap.parse_args()

    out = Path(a.out)
    cache = out / "cache_shares"; cache.mkdir(parents=True, exist_ok=True)
    want = {int(x.strip()) for x in a.firms.split(",") if x.strip()}

    client = None
    if not a.mock:
        try:
            import anthropic
        except ImportError:
            sys.exit("pip install anthropic  (or run with --mock)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first")
        client = anthropic.Anthropic()

    rows = []
    for fp in sorted((out / "text").glob("*.json")):
        rec = json.load(open(fp, encoding="utf-8"))
        if int(rec["firm_id"]) not in want:
            continue
        text = rec.get("full_text") or rec.get("reserve_text") or ""
        if len(text.split()) < 50:
            continue
        prompt = USER.format(firm_id=rec["firm_id"], year=rec["year"], text=trim(text))
        h = hashlib.sha256(f"{a.model}|{prompt}".encode()).hexdigest()[:24]
        cf = cache / f"{h}.json"
        if cf.exists():
            res = json.load(open(cf, encoding="utf-8"))
        else:
            try:
                res = mock(text) if a.mock else call_model(client, a.model, prompt)
            except Exception as e:
                print(f"{rec['firm_id']} {rec['year']}: FAILED {type(e).__name__}")
                continue
            json.dump(res, open(cf, "w", encoding="utf-8"))

        n = res.get("shares")
        mult = SCALE.get(str(res.get("unit")).lower(), None)
        rows.append(dict(firm_id=int(rec["firm_id"]), year=int(rec["year"]),
                         shares_as_printed=n, unit=res.get("unit"),
                         shares_absolute=(n * mult) if (n and mult) else None,
                         shares_millions=(n * mult / 1e6) if (n and mult) else None,
                         reporting_date=res.get("reporting_date"),
                         share_class=res.get("share_class"),
                         consolidation=res.get("consolidation"),
                         note=res.get("note", ""), evidence=res.get("evidence", "")))
        print(f"{rec['firm_id']} {rec['year']}: {n} {res.get('unit')} "
              f"({res.get('share_class')})")

    df = pd.DataFrame(rows).sort_values(["firm_id", "year"])
    df.to_csv(out / "shares_extracted.csv", index=False)

    print(f"\nExtracted {len(df)} firm-years")
    if len(df):
        print(df[["firm_id", "year", "shares_as_printed", "unit",
                  "shares_millions", "share_class"]].to_string(index=False))
        if df["consolidation"].astype(str).str.lower().eq("true").any():
            print("\nCONSOLIDATION FLAGGED in at least one year. Counts before and")
            print("after a consolidation are not comparable: restate the earlier")
            print("years onto the new basis before computing per-share values.")
        for fid, g in df.groupby("firm_id"):
            s = g.dropna(subset=["shares_millions"]).sort_values("year")
            if len(s) > 1:
                r = s["shares_millions"].max() / max(s["shares_millions"].min(), 1e-9)
                print(f"\n  firm {fid}: {s.shares_millions.min():,.1f}m to "
                      f"{s.shares_millions.max():,.1f}m shares "
                      f"({r:,.1f}x over the panel)")
    print(f"\nSaved -> {out/'shares_extracted.csv'}")
    print("Check the evidence column before using these figures, then apply them")
    print("with 19_apply_corrections.py or by editing the Block A sheet.")


if __name__ == "__main__":
    main()
