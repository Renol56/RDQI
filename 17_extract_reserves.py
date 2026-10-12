#!/usr/bin/env python3
"""
Step 17 - Extract group reserve tonnage per firm-year from the reserve disclosures.

Replaces the S&P property-level route, which returned each firm's peripheral
assets while missing its core operations.

For every firm-year it asks for the TOTAL PROVED PLUS PROBABLE MINERAL RESERVE
TONNAGE reported for the group, on an attributable basis where the firm gives one,
together with the unit as printed, the commodity the tonnage refers to, and a
verbatim quote. Where a firm reports several commodities separately it returns one
line per commodity, so the primary commodity can be derived rather than judged.

Same discipline as the RDQI scoring: fixed prompt, temperature 0, cached on a hash
of the prompt, verbatim evidence required, and nothing inferred that is not printed.

Output
    out/reserves_raw.csv        one row per firm-year per commodity, as reported
    out/reserves_firm_year.csv  aggregated: RESTON inputs, primary commodity, MULTICOM

Usage:
    python 17_extract_reserves.py --out out --mock          # test, no API
    python 17_extract_reserves.py --out out --limit 5       # pilot
    python 17_extract_reserves.py --out out
"""
import argparse, hashlib, json, os, re, sys, time, unicodedata
from pathlib import Path
import pandas as pd

MAX_CHARS = 150_000          # a 380k-character statement was being cut to 49k,
                             # dropping the very table that holds the group total

SYSTEM = """You extract reported mineral reserve figures for an accounting research
dataset. You are literal: report only figures printed in the text. Never estimate,
never convert units, never sum figures yourself unless the instruction says to.
If a figure is not present, return null.

Return ONLY a JSON object, no preamble, no markdown fences."""

USER = """From the mineral reserve disclosure below, extract the TOTAL MINERAL RESERVE
for the group.

COMPANY {firm_id}, FISCAL YEAR {year}

WHAT TO EXTRACT
- Total PROVED (or Proven) plus PROBABLE Mineral Reserves for the whole group.
- Prefer the ATTRIBUTABLE total where the company reports both attributable and
  100% figures. Record which basis you used.
- Report the TONNAGE of ore where it is given. Many companies also report contained
  metal (ounces, kilograms, contained tonnes); capture that separately, do not
  substitute it for ore tonnage.
- If the company reports separate totals by commodity (for example PGMs, iron ore
  and coal), return ONE ENTRY PER COMMODITY. Do not add different commodities together.
- If only individual mines are listed with no group total, add the mine figures for
  the SAME commodity and set "summed_by_you": true.

RULES
- Copy the unit exactly as printed: Mt, kt, tonnes, Moz, koz, kg, Mlb.
- Every entry needs a verbatim quote of 4-20 words copied from the text, containing
  the figure. Without a quotable figure, return no entry.
- Reserves only. Never report Measured, Indicated or Inferred RESOURCES as reserves.
- If the disclosure reports resources but declares no reserves, return an empty list
  and set "no_reserves_declared": true.
- Do not convert between units or commodities.

DISCLOSURE
<<<
{text}
>>>

Return JSON exactly:
{{"no_reserves_declared": false,
  "basis": "attributable" | "100%" | "not stated",
  "entries": [
    {{"commodity": "<as printed>",
      "ore_tonnage": <number or null>, "ore_unit": "<Mt|kt|tonnes|null>",
      "contained": <number or null>, "contained_unit": "<Moz|koz|kg|Mlb|t|null>",
      "summed_by_you": false,
      "evidence": "<verbatim quote containing the figure>"}}
  ]}}
"""

SHORT_TON = 0.907185          # 1 short ton in tonnes

TONNE_FACTOR = {
    # millions
    "mt": 1e6, "million tonnes": 1e6, "million tons": 1e6 * SHORT_TON,
    "mmt": 1e6, "m tonnes": 1e6, "mt ore": 1e6,
    # thousands - the form used in North American filings
    "kt": 1e3, "thousand tonnes": 1e3, "kilotonnes": 1e3, "000t": 1e3,
    "x1000 tonnes": 1e3, "x1000 tonne": 1e3, "'000 tonnes": 1e3,
    "000 tonnes": 1e3, "kilo tonnes": 1e3, "thousands of tonnes": 1e3,
    "x1000 tons": 1e3 * SHORT_TON, "'000 tons": 1e3 * SHORT_TON,
    "thousand tons": 1e3 * SHORT_TON, "000 tons": 1e3 * SHORT_TON,
    # units
    "t": 1.0, "tonnes": 1.0, "tonne": 1.0, "metric tonnes": 1.0,
    "tons": SHORT_TON, "ton": SHORT_TON, "short tons": SHORT_TON,
    # spelled-out and dry-metric variants seen in the corpus
    "millions of tonnes": 1e6, "millions of tons": 1e6 * SHORT_TON,
    "million": 1e6, "millions": 1e6,
    "mdmt": 1e6, "million dmt": 1e6, "millions of dmt": 1e6,
    "dmt": 1.0, "bdmt": 1.0, "million bdmt": 1e6, "mbdmt": 1e6,
    "million wmt": 1e6, "wmt": 1.0, "mwmt": 1e6,
    "000s": 1e3, "'000s": 1e3, "thousands": 1e3,
}

# Units of CONTAINED METAL, not ore. If one of these turns up in the ore_unit
# field the model has put contained metal in the wrong place: the row carries no
# ore tonnage and must not be converted, or a firm's ore base is silently wrong.
CONTAINED_UNITS = re.compile(
    r"\b(oz|ozs|ounce|ounces|moz|koz|carat|carats|mct|lb|lbs|pound|pounds|mlb|"
    r"klb|kg|kilogram|g|grams|oz/t|g/t)\b", re.I)

SECTION_RE = re.compile(
    r"total|group|proved|proven|probable|reserve|attributable|tonnes|tonnage|mt\b",
    re.I)


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


TOTAL_RE = re.compile(
    r"total\s+(group\s+)?(proved|proven|probable|mineral\s+reserve|ore\s+reserve)|"
    r"(group|company|consolidated|attributable)\s+total|"
    r"total\s+(attributable|worldwide|reserves)|grand\s+total|"
    r"total\s+mineral\s+reserves", re.I)


def trim(text, cap=MAX_CHARS):
    """Keep the passages likely to carry a group reserve total. Unlike the RDQI
    filter this KEEPS numeric table rows, because the figure itself lives there.

    Where a document is longer than the cap, the passages containing an explicit
    TOTAL are preserved first and the rest fills the remaining budget. Truncating
    from the front dropped exactly the totals table in several large statements."""
    if len(text) <= cap:
        return text
    parts = re.split(r"(?<=[.;])\s+|\n", text)
    totals = [p for p in parts if TOTAL_RE.search(p)]
    rest = [p for p in parts if SECTION_RE.search(p) and not TOTAL_RE.search(p)]

    head = "\n".join(totals)
    if len(head) > cap:
        return head[:cap]
    out = head + "\n" + "\n".join(rest)
    if len(out.strip()) < 2000:
        # too little matched the filters: keep any totals line at the FRONT and
        # fill the remaining budget with the raw text, rather than discarding the
        # total by falling back to the head of the document
        out = (head + "\n" + text) if head.strip() else text
    return out[:cap]


def parse_json(raw):
    """Parse the model's JSON. A long reply can be cut off at the token limit,
    leaving an unterminated object; rather than lose the whole document, close the
    structure after the last complete entry and keep what did arrive."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i, j = raw.find("{"), raw.rfind("}")
    if i < 0:
        raise ValueError("no JSON object in response")
    body = raw[i:j + 1] if j > i else raw[i:]
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        pass
    # trim back to the last complete entry and close the brackets
    cut = body.rfind("}")
    while cut > 0:
        trial = body[:cut + 1]
        if trial.count("[") > trial.count("]"):
            trial += "]"
        while trial.count("{") > trial.count("}"):
            trial += "}"
        try:
            out = json.loads(trial)
            out["_truncated"] = True
            return out
        except json.JSONDecodeError:
            cut = body.rfind("}", 0, cut)
    raise ValueError("could not salvage JSON from response")


def call_model(client, model, prompt, tries=4):
    for a in range(tries):
        try:
            kw = dict(model=model, max_tokens=8000, system=SYSTEM,
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


def _num(s):
    """Parse a printed figure, or return None. Real documents contain strings like
    '4.2.2' (a section number) that look numeric to a regex but are not."""
    # South African and French filings write "121 500" with a space separator
    t = re.sub(r"[\s,\u00a0\u2009]", "", str(s)).strip()
    try:
        return float(t)
    except ValueError:
        return None


def mock(text):
    """Crude stand-in for the model: finds '<Commodity> Mineral Reserves N Unit'."""
    pat = re.compile(r"Total\s+([A-Za-z ]+?)\s+Mineral Reserves[: ]+([\d ,\.]+)\s*"
                     r"(Mt|kt|tonnes)", re.I)
    ents = []
    for m in pat.finditer(text):
        v = _num(m.group(2))
        if v is None:
            continue
        ents.append({"commodity": m.group(1).strip().title(),
                     "ore_tonnage": v,
                     "ore_unit": m.group(3), "contained": None,
                     "contained_unit": None, "summed_by_you": False,
                     "evidence": m.group(0)[:60]})
    if not ents:
        for m in re.finditer(r"([\d][\d ,\.]{2,})\s*(Mt|kt|tonnes)", text, re.I):
            v = _num(m.group(1))
            if v is None or "no mineral reserves" in text.lower():
                continue
            ents = [{"commodity": "Gold", "ore_tonnage": v,
                     "ore_unit": m.group(2), "contained": None,
                     "contained_unit": None, "summed_by_you": False,
                     "evidence": m.group(0)}]
            break
    if not ents:
        return {"no_reserves_declared": True, "basis": "not stated", "entries": []}
    return {"no_reserves_declared": False, "basis": "attributable", "entries": ents}


def to_tonnes(v, unit):
    """Convert a printed ore tonnage to tonnes. Returns None on an unrecognised
    unit rather than guessing - an unconverted row is visible, a wrongly
    converted one is not."""
    if isinstance(v, str):
        v = _num(v)
    if v is None or unit is None:
        return None
    u = re.sub(r"[\s\u00a0]+", " ", str(unit)).strip().lower().rstrip(".")
    u = u.replace("×", "x").replace("’", "'")
    if CONTAINED_UNITS.search(u) and not re.search(r"tonne|ton\b|dmt|wmt", u):
        return None          # contained metal, not ore tonnage
    if u in TONNE_FACTOR:
        return v * TONNE_FACTOR[u]
    # tolerate wording variants: "x1,000 tonnes", "in thousands of tonnes"
    m = re.search(r"(x ?1[, ]?000|'000|000)\s*(tonnes?|tons?)", u)
    if m:
        return v * (1e3 if m.group(2).startswith("tonne") else 1e3 * SHORT_TON)
    if re.fullmatch(r"m(illion)? ?(tonnes?|tons?)", u):
        return v * (1e6 if "tonne" in u else 1e6 * SHORT_TON)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    out = Path(a.out)
    cache = out / "cache_resv"; cache.mkdir(parents=True, exist_ok=True)

    client = None
    if not a.mock:
        try:
            import anthropic
        except ImportError:
            sys.exit("pip install anthropic  (or run with --mock)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first")
        client = anthropic.Anthropic()

    # restrict to the analysis sample where a panel exists
    keep = None
    if (out / "panel.csv").exists():
        p = pd.read_csv(out / "panel.csv")
        keep = set(zip(p.firm_id.astype(int), p.year.astype(int)))

    rows, done, skipped, failed = [], 0, 0, 0
    for fp in sorted((out / "text").glob("*.json")):
        if a.limit and done >= a.limit:
            break
        rec = json.load(open(fp, encoding="utf-8"))
        key = (int(rec["firm_id"]), int(rec["year"]))
        if keep is not None and key not in keep:
            continue
        text = rec.get("reserve_text") or ""
        if len(text.split()) < 20:
            skipped += 1
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
                print(f"  {key[0]} {key[1]}: FAILED ({type(e).__name__}) - skipped")
                rows.append(dict(firm_id=key[0], year=key[1], commodity=None,
                                 ore_tonnage=None, ore_unit=None, tonnes=None,
                                 contained=None, contained_unit=None, basis=None,
                                 summed_by_you=None, evidence="",
                                 note=f"extraction failed: {type(e).__name__}"))
                failed += 1
                continue
            json.dump(res, open(cf, "w", encoding="utf-8"))

        if res.get("no_reserves_declared") or not res.get("entries"):
            rows.append(dict(firm_id=key[0], year=key[1], commodity=None,
                             ore_tonnage=None, ore_unit=None, tonnes=None,
                             contained=None, contained_unit=None,
                             basis=res.get("basis"), summed_by_you=None,
                             evidence="", note="no reserves declared"))
        else:
            for e in res["entries"]:
                rows.append(dict(
                    firm_id=key[0], year=key[1], commodity=e.get("commodity"),
                    ore_tonnage=e.get("ore_tonnage"), ore_unit=e.get("ore_unit"),
                    tonnes=to_tonnes(e.get("ore_tonnage"), e.get("ore_unit")),
                    contained=e.get("contained"), contained_unit=e.get("contained_unit"),
                    basis=res.get("basis"), summed_by_you=e.get("summed_by_you"),
                    evidence=e.get("evidence", ""), note=""))
        done += 1
        n = len([r for r in rows if r["firm_id"] == key[0] and r["year"] == key[1]])
        print(f"{key[0]} {key[1]}: {n} entry(ies)")

    raw = pd.DataFrame(rows)
    raw.to_csv(out / "reserves_raw.csv", index=False)

    # ---------- aggregate to firm-year ----------
    ok = raw[raw["tonnes"].notna()].copy()

    # normalise commodity labels: the same metal is printed many ways
    def canon(c):
        t = re.sub(r"\(.*?\)", "", str(c)).strip().lower()
        t = re.sub(r"[^a-z ]+", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        for key, name in [
            ("pgm", "PGM"), ("platinum", "PGM"), ("pge", "PGM"),
            ("gold", "Gold"), ("silver", "Silver"), ("copper", "Copper"),
            ("iron", "Iron ore"), ("manganese", "Manganese"),
            ("chrom", "Chrome"), ("coal", "Coal"), ("graphite", "Graphite"),
            ("lithium", "Lithium"), ("nickel", "Nickel"), ("zinc", "Zinc"),
            ("bauxite", "Bauxite"), ("uranium", "Uranium"),
            ("ilmenite", "Mineral sands"), ("mineral sand", "Mineral sands"),
            ("diamond", "Diamonds"), ("phosphate", "Phosphate"),
        ]:
            if key in t:
                return name
        return str(c).strip().title()
    ok["commodity"] = ok["commodity"].map(canon)

    # A company that reports the same table in both tonnes and short tons produces
    # two rows for one figure. Keep one per firm-year-commodity, preferring the
    # metric row, then the larger tonnage.
    ok["_metric"] = ok["ore_unit"].astype(str).str.contains("tonne", case=False).astype(int)
    ok = (ok.sort_values(["_metric", "tonnes"], ascending=[False, False])
            .drop_duplicates(["firm_id", "year", "commodity"], keep="first"))

    agg = []
    for (fid, yr), g in ok.groupby(["firm_id", "year"]):
        byc = g.groupby("commodity")["tonnes"].sum().sort_values(ascending=False)
        agg.append(dict(firm_id=fid, year=yr,
                        primary_commodity=byc.index[0],
                        # RESTON uses the PRIMARY commodity only. Summing across
                        # commodities would double-count: a polymetallic orebody is
                        # reported once per metal, so the same rock appears several
                        # times. reserve_tonnes_max_commodity is the defensible
                        # measure of the firm's ore base.
                        reserve_tonnes_primary=byc.iloc[0],
                        reserve_tonnes_sum_all=byc.sum(),
                        n_commodities=len(byc),
                        multi_commodity=int(len(byc) > 1),
                        basis=g["basis"].iloc[0]))
    fy = pd.DataFrame(agg)
    if len(fy):
        # hold the primary commodity constant per firm, taken from its latest year
        last = fy.sort_values("year").groupby("firm_id")["primary_commodity"].last()
        fy["primary_commodity_fixed"] = fy["firm_id"].map(last)
    fy.to_csv(out / "reserves_firm_year.csv", index=False)

    print(f"\nDocuments processed : {done}   (skipped {skipped} too short, "
          f"{failed} failed)")
    print(f"Firm-years with a usable tonnage : {len(fy)}")
    if len(fy):
        print(f"Firms                            : {fy.firm_id.nunique()}")
        print(f"Multi-commodity firm-years       : {int(fy.multi_commodity.sum())}")
        print("\nBasis reported:")
        print(fy["basis"].value_counts().to_string())
        print("\nTop commodities (fixed per firm):")
        print(fy.groupby("firm_id")["primary_commodity_fixed"].first()
              .value_counts().to_string())
    nore = raw[raw["note"] == "no reserves declared"]
    if len(nore):
        print(f"\nFirm-years reporting NO reserves : {len(nore)}")
    print(f"\nSaved -> {out/'reserves_raw.csv'} and {out/'reserves_firm_year.csv'}")
    bad = raw[raw["ore_tonnage"].notna() & raw["tonnes"].isna()]
    if len(bad):
        print(f"\nUNCONVERTED ROWS: {len(bad)} - the unit was not recognised.")
        print(bad["ore_unit"].value_counts().to_string())
        print("Send me these unit strings and they will be added to the converter.")
    print("\nUse reserve_tonnes_primary for RESTON, not reserve_tonnes_sum_all:")
    print("a polymetallic orebody is reported once per metal, so summing across")
    print("commodities counts the same rock several times.")
    print("Check that a firm's tonnage moves sensibly year to year.")


if __name__ == "__main__":
    main()
