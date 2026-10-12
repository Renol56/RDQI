#!/usr/bin/env python3
"""
Step 3 - Score disclosure QUALITY 0/1/2 per item using an LLM against a fixed rubric.

Key design choices (these are what referees ask about):
  * temperature = 0 and a fixed rubric -> reproducible
  * the model must return an evidence quote and a page hint for every non-zero
    score, so any score can be audited back to the document
  * responses are cached on a hash of (text, rubric, model), so re-runs are free
  * --runs 2 scores each document twice to measure machine self-consistency

Usage:
    export ANTHROPIC_API_KEY=sk-...
    python 03_score_quality.py --out out --model claude-sonnet-4-6
    python 03_score_quality.py --out out --runs 2        # self-consistency check
    python 03_score_quality.py --out out --mock          # pipeline test, no API
"""
import argparse, hashlib, json, os, re, sys, time, unicodedata
from pathlib import Path
import pandas as pd

MAX_CHARS = 120_000          # hard ceiling after trimming
MIN_KEEP_RATIO = 0.15        # if trimming keeps less than this share of the
MIN_KEEP_CHARS = 3_000       # section, or less than this many characters,
                             # send the untrimmed section instead
WINDOW = 1_500               # chars kept around each keyword hit when condensing

SYSTEM = """You are a disclosure-quality coder for an accounting research project.
You score mineral reserve disclosures against a fixed rubric. You are strict and
literal: score only what the text actually says. Never infer, never reward intent,
never use knowledge about the company from outside the text provided.

Return ONLY a JSON object, no preamble, no markdown fences."""

USER_TEMPLATE = """Score the reserve disclosure below against the rubric.

SCALE
0 = {s0}
1 = {s1}
2 = {s2}

RUBRIC
{rubric_block}

RULES
- Score each item independently.
- Any score of 1 or 2 requires a VERBATIM quote copied word-for-word from the
  disclosure above, 5-20 words. If you cannot copy such a quote, the score is 0.
- The evidence field takes quoted text ONLY. Never write your own description,
  summary, explanation or judgement there. Never write phrases like "not
  disclosed", "no ... published", "not explicitly detailed", "disclosed at group
  level but ...". If the only thing you could write is a statement about what is
  MISSING, then the item is absent and the score is 0.
- Quote ONE representative example, not a list gathered across several assets.
- The disclosure may be in English or French. Score it identically either way and
  copy the evidence quote VERBATIM in the language of the document - never
  translate a quote. French equivalents: reserves minerales = mineral reserves,
  personne competente = competent person, teneur de coupure = cut-off grade,
  taux de recuperation = metallurgical recovery, taux de change = exchange rate,
  taux d'actualisation = discount rate, redevances = royalties,
  rapprochement = reconciliation, visite du site = site visit.
- Some items cross-reference each other. Where an anchor tells you to check
  another item first, do so, and make the two scores consistent.
- The quote must evidence THIS item's question. A quote that proves a different
  item does not count: e.g. a cut-off grade VALUE evidences I9, not I10, which
  asks for the BASIS of that cut-off. If the quote does not answer the question
  asked, the score is 0.
- A number counts as quantified only if it appears in the text with its unit.
- Some anchors require TWO components (e.g. I16 needs an exchange rate AND a
  discount rate). If your quote shows only one component, the score is 1, not 2.
- Boilerplate that repeats the code's wording without company-specific content is 1, not 2.
- Respect each item's max: an item with max 1 can only be scored 0 or 1.
- Use the string "NA" as the score ONLY for items marked NA-allowed, and only
  where the item genuinely cannot arise. Do use it: for I23, if there was no
  material downward restatement in the year, the correct score is "NA", NOT 0.
  NA and 0 are different - NA removes the item from the denominator, 0 counts
  against the company.
- If the disclosure is truncated or unreadable, use "unreadable": true.
- Before returning, re-read each evidence string. If it is not text you copied
  out of the disclosure, set that item's score to 0 and its evidence to "".

DISCLOSURE ({firm_id}, FY{year})
<<<
{text}
>>>

Return JSON exactly in this shape:
{{"unreadable": false,
  "scores": {{ {shape} }} }}
where each item maps to {{"s": 0|1|2, "e": "<quote or empty>"}}
"""


NUM_ROW = re.compile(r"^[^A-Za-z]*(?:[\d.,%\u2013\-]+\s+){3,}")


def fold(s):
    """Lowercase and strip accents so 'reserves' matches 'réserves' either way.
    PDF text layers are inconsistent about accents, so match on the folded form
    and keep the original text for the quote."""
    return "".join(c for c in unicodedata.normalize("NFD", s.lower())
                   if unicodedata.category(c) != "Mn")


def trim(text, rubric, cap=MAX_CHARS):
    """Cut a reserve section down to the sentences that can carry rubric evidence.

    Reserve reports are mostly numeric tables. Those tables are what the tonnage
    and grade figures live in, but every rubric item is judged on NARRATIVE text -
    footnotes, competence statements, explanatory notes. Dropping table rows and
    keeping only keyword-bearing sentences cuts roughly 80% of the characters and
    therefore about 80% of the API cost, with no loss of scoreable evidence.

    Returns (text_to_send, was_trimmed).
    """
    kws = sorted({fold(k) for it in rubric["items"] for k in it["keywords"]},
                 key=len, reverse=True)
    lines = [l for l in text.split("\n") if not NUM_ROW.match(l.strip())]
    flat = re.sub(r"\s+", " ", " ".join(lines))
    parts = re.split(r"(?<=[.;])\s+", flat)

    seen, keep = set(), []
    for p in parts:
        if len(p) > 900:
            continue
        low = fold(p)
        if not any(k in low for k in kws):
            continue
        head = low[:120]
        if head in seen:            # drop repeated boilerplate across properties
            continue
        seen.add(head)
        keep.append(p)

    out = " ".join(keep)

    # RETENTION FLOOR. The filter drops numeric table rows, which is right for a
    # document whose narrative sits in prose. In a heavily tabulated reserve
    # statement the narrative sits BETWEEN table rows, and stripping them removes
    # the narrative too: observed cases retained under 3% of the source, leaving
    # only footnotes, map legends and running headers. The model then scores zero
    # on everything, correctly given what it was sent and wrongly about the
    # document. Where the filter keeps too little, send the untrimmed section
    # instead and let the character cap do the limiting.
    if not out.strip() or len(out) < max(MIN_KEEP_CHARS, MIN_KEEP_RATIO * len(text)):
        out = flat
    trimmed = len(out) < len(text)
    return out[:cap], trimmed


def build_prompt(rec, rubric, text):
    lines = []
    for it in rubric["items"]:
        L = [f"- {it['id']} ({it['dimension']}), max {it['max']}: {it['question']}",
             f"    0 = {it['anchor_0']}", f"    1 = {it['anchor_1']}"]
        if it["max"] == 2:
            L.append(f"    2 = {it['anchor_2']}")
        else:
            L.append("    (scored 0 or 1 only - never award 2 for this item)")
        if it.get("na_allowed"):
            L.append("    NA = the item cannot arise for this disclosure")
        L.append("")
        lines.append("\n".join(L))
    block = "\n".join(lines)
    shape = ", ".join(f'"{it["id"]}": {{"s":..,"e":".."}}' for it in rubric["items"])
    return USER_TEMPLATE.format(
        s0=rubric["scale"]["0"], s1=rubric["scale"]["1"], s2=rubric["scale"]["2"],
        rubric_block=block, firm_id=rec["firm_id"], year=rec["year"],
        text=text, shape=shape)


def parse_json(raw):
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.M).strip()
    i, j = raw.find("{"), raw.rfind("}")
    return json.loads(raw[i:j + 1])


# anthropic SDK v1.0 (Aug 2026) removed temperature/top_p/top_k: passing them
# raises TypeError. We try with temperature=0 first (better for reproducibility,
# and it works on anthropic<1.0), then fall back once and remember the result.
_USE_TEMPERATURE = True


def call_model(client, model, prompt, tries=4):
    global _USE_TEMPERATURE
    for a in range(tries):
        try:
            kw = dict(model=model, max_tokens=2000, system=SYSTEM,
                      messages=[{"role": "user", "content": prompt}])
            if _USE_TEMPERATURE:
                kw["temperature"] = 0
            msg = client.messages.create(**kw)
            return parse_json("".join(b.text for b in msg.content if b.type == "text"))
        except TypeError as e:
            if _USE_TEMPERATURE and "temperature" in str(e):
                _USE_TEMPERATURE = False
                print("    note: this SDK rejects temperature; continuing without it.\n"
                      "    For temperature=0 reproducibility run: pip install \"anthropic<1.0\"",
                      file=sys.stderr)
                continue
            raise
        except Exception as e:
            if a == tries - 1:
                raise
            time.sleep(3 * (a + 1))
            print(f"    retry {a+1}: {type(e).__name__}", file=sys.stderr)


def mock_result(rubric, text):
    low = text.lower()
    out = {}
    for it in rubric["items"]:
        hit = [k for k in it["keywords"] if k in low]
        cap = it["max"]
        sc = 0 if not hit else (cap if re.search(r"\d", text) and len(hit) > 1 else 1)
        out[it["id"]] = {"s": sc, "e": hit[0] if hit else ""}
    return {"unreadable": False, "scores": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--rubric", default="rubric_v2.json")
    ap.add_argument("--model", default="claude-haiku-4-5-20251001")
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rubric = json.load(open(args.rubric, encoding="utf-8"))
    out = Path(args.out)
    cache_dir = out / "cache"; cache_dir.mkdir(parents=True, exist_ok=True)

    client = None
    if not args.mock:
        try:
            import anthropic
        except ImportError:
            sys.exit("pip install anthropic  (or run with --mock)")
        if not os.environ.get("ANTHROPIC_API_KEY"):
            sys.exit("Set ANTHROPIC_API_KEY first")
        client = anthropic.Anthropic()

    files = sorted((out / "text").glob("*.json"))[: args.limit]
    rows, raw_rows = [], []

    for fp in files:
        rec = json.load(open(fp, encoding="utf-8"))
        text = rec["reserve_text"] or ""
        if len(text.split()) < 50:
            print(f"{fp.stem}: section too short, flagged for manual coding")
            rows.append(dict(firm_id=rec["firm_id"], year=rec["year"], run=1,
                             flag="too_short"))
            continue

        sent, condensed = trim(text, rubric)
        prompt = build_prompt(rec, rubric, sent)

        for run in range(1, args.runs + 1):
            key = hashlib.sha256(
                f"{args.model}|{run}|{prompt}".encode()).hexdigest()[:24]
            cache = cache_dir / f"{key}.json"
            if cache.exists():
                res = json.load(open(cache, encoding="utf-8"))
            else:
                res = mock_result(rubric, sent) if args.mock \
                    else call_model(client, args.model, prompt)
                json.dump(res, open(cache, "w", encoding="utf-8"))

            row = dict(firm_id=rec["firm_id"], year=rec["year"], run=run,
                       condensed=int(condensed), unreadable=int(res.get("unreadable", False)),
                       flag="")
            for it in rubric["items"]:
                s = res["scores"].get(it["id"], {})
                raw = s.get("s", s.get("score", 0))
                row["q_" + it["id"]] = ("NA" if str(raw).upper() == "NA"
                                        else min(int(raw), it["max"]))
                raw_rows.append(dict(firm_id=rec["firm_id"], year=rec["year"], run=run,
                                     item=it["id"], score=row["q_" + it["id"]],
                                     evidence=s.get("e", s.get("evidence", ""))))
            # --- deterministic consistency rules, applied after the model ---
            # Your I3 decision: an external sign-off credits independence even
            # where the CP is an employee. I22 detects exactly that sign-off.
            # The model scores I3 before it has read I22, so enforce it here
            # rather than relying on the prompt. Documented, reproducible, and
            # applied identically to every firm-year.
            if row.get("q_I22") == 2 and row.get("q_I3") not in (2, "NA"):
                row["q_I3"] = 2
                row["rule_I3_from_I22"] = 1
            else:
                row["rule_I3_from_I22"] = 0

            rows.append(row)
            vals = [row["q_" + it["id"]] for it in rubric["items"]]
            total = sum(v for v in vals if v != "NA")
            print(f"{rec['firm_id']} {rec['year']} run{run}: total={total}")

    for r in rows:
        r["temperature_0"] = int(_USE_TEMPERATURE)
    pd.DataFrame(rows).to_csv(out / "quality_raw.csv", index=False)
    ev = pd.DataFrame(raw_rows)
    fixed = {(r["firm_id"], r["year"], r["run"]) for r in rows
             if r.get("rule_I3_from_I22")}
    if not ev.empty and fixed:
        m = ev.apply(lambda x: (x["firm_id"], x["year"], x["run"]) in fixed
                     and x["item"] == "I3", axis=1)
        ev.loc[m, "score"] = 2
        ev.loc[m, "evidence"] = "[set to 2 by consistency rule: I22 = 2]"
    ev.to_csv(out / "quality_evidence.csv", index=False)
    print(f"\nSaved -> {out/'quality_raw.csv'} and {out/'quality_evidence.csv'}")
    if args.runs > 1:
        print("Run 05_agreement.py --self-consistency to compare runs.")


if __name__ == "__main__":
    main()
