#!/usr/bin/env python3
"""
Step 11 - Identify the external auditor and code Big 4 status.

Reads the extracted full text (out/text/*.json), locates the auditor's report or
signature block, and identifies the audit firm. Regex-first: it only reports a
result where the firm name appears in an AUDITOR CONTEXT, not merely anywhere in
the document - annual reports mention Deloitte or PwC in advisory, tax and
valuation roles, and a naive keyword search misclassifies those.

Output: out/auditor.csv
    firm_id, year, auditor, big4, method, confidence, evidence

  big4 = 1  Deloitte, EY, KPMG, PwC (including local member firms)
  big4 = 0  any other identified audit firm
  big4 = blank  not identified - review by hand

Usage:
    python 11_detect_auditor.py --out out
    python 11_detect_auditor.py --out out --review        # list only what needs checking
"""
import argparse, json, re, unicodedata
from pathlib import Path
import pandas as pd

BIG4 = {
    "Deloitte": [r"deloitte(\s*&\s*touche)?", r"deloitte\s+touche\s+tohmatsu"],
    "EY": [r"ernst\s*&\s*young", r"ernst\s+and\s+young", r"\bey\s+(inc|llp|audit|s\.?a\.?)",
           r"\bey\b(?=\s*$)"],
    "KPMG": [r"\bkpmg\b"],
    "PwC": [r"pricewaterhousecoopers", r"price\s*waterhouse\s*coopers", r"\bpwc\b"],
}

NON_BIG4 = {
    "BDO": [r"\bbdo\b"], "Grant Thornton": [r"grant\s+thornton"],
    "SNG Grant Thornton": [r"sizwentsalubagobodo", r"\bsng\b\s+grant"],
    "Mazars": [r"\bmazars\b"], "Forvis Mazars": [r"forvis\s+mazars"],
    "Moore": [r"\bmoore\s+(stephens|global|za|kingston)"],
    "Nexia": [r"\bnexia\b"], "RSM": [r"\brsm\b"], "Crowe": [r"\bcrowe\b"],
    "Baker Tilly": [r"baker\s+tilly"], "PKF": [r"\bpkf\b"], "HLB": [r"\bhlb\b"],
    "MNP": [r"\bmnp\s+llp"], "Marcum": [r"\bmarcum\b"], "KNAV": [r"\bknav\b"],
    "Stantons": [r"stantons?\s+international"], "HLB Mann Judd": [r"hlb\s+mann\s+judd"],
    "William Buck": [r"william\s+buck"], "Bentleys": [r"\bbentleys\b"],
    "Pitcher Partners": [r"pitcher\s+partners"], "Hall Chadwick": [r"hall\s+chadwick"],
    "Nexia SAB&T": [r"nexia\s+sab"], "SizweNtsalubaGobodo": [r"sizwe\s*ntsaluba"],
    "Davidson": [r"davidson\s+&\s+company"], "Smythe": [r"smythe\s+llp"],
    "Dale Matheson": [r"dale\s+matheson"], "Raines": [r"raines\s+&\s+fischer"],
    "Turner Stone": [r"turner,?\s+stone"], "DNTW": [r"\bdntw\b"],
    "Mazars Mauritius": [r"mazars\s+mauritius"], "SW": [r"\bshinewing\b"],
}

# Text that signals we are in the auditor's report or the signature block
CONTEXT = re.compile(
    r"independent\s+auditor.{0,3}s?\s+report|report\s+of\s+(the\s+)?independent"
    r"(\s+registered\s+public\s+accounting\s+firm)?|auditor.{0,3}s?\s+report\s+to\s+the|"
    r"registered\s+auditor|chartered\s+accountants|"
    r"we\s+have\s+audited|in\s+our\s+opinion|report\s+on\s+the\s+audit\s+of|"
    r"appointed\s+as\s+(the\s+)?(external\s+)?auditor|"
    r"rapport\s+de\s+l.{0,3}auditeur|commissaire\s+aux\s+comptes|"
    r"nous\s+avons\s+effectu[ée]\s+l.{0,3}audit",
    re.I)
WINDOW = 2500


def fold(s):
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn").lower()


def auditor_windows(text):
    """Text around every auditor-context cue."""
    low = fold(text)
    out = []
    for m in CONTEXT.finditer(low):
        a, b = max(0, m.start() - 400), min(len(text), m.end() + WINDOW)
        out.append(text[a:b])
    return out


# Words that mark a NON-audit engagement. Annual reports name Big 4 firms as tax
# advisers, valuers, sponsors and consultants; counting those misclassifies the
# auditor, so any firm name sitting near one of these is discarded.
ADVISORY = re.compile(
    r"tax\s+(advis|servic|complian)|advisory\s+(servic|fee)|consult(ing|ancy)\s+fee|"
    r"non-?audit\s+(fee|servic)|valuation\s+(servic|of\s+the)|corporate\s+finance|"
    r"transaction\s+servic|sponsor|remuneration\s+consult|actuar|"
    r"was\s+engaged\s+to\s+(advise|assist)|assisted\s+with", re.I)
# Words that mark the signing block of the audit opinion
SIGNATURE = re.compile(
    r"registered\s+auditor|chartered\s+accountant|\bper\s*:|/s/|"
    r"partner|director\s*$|designated\s+auditor|statutory\s+auditor|"
    r"commissaire\s+aux\s+comptes|expert-?comptable", re.I)
ALL_FIRMS = {**BIG4, **NON_BIG4}


def identify(chunks):
    """Find the SIGNING auditor, not every firm mentioned nearby.

    Scoring, in order of weight:
      +5  the name appears within 400 chars of a signature marker
      +3  the name is the FIRST firm named after the auditor-report cue
      +1  any other appearance in auditor context
      -6  the name sits within 200 chars of advisory/non-audit wording
    """
    score, ev = {}, {}
    for ch in chunks:
        low = fold(ch)
        hits = []
        for name, pats in ALL_FIRMS.items():
            for p in pats:
                for m in re.finditer(p, low, re.I):
                    hits.append((m.start(), m.end(), name))
        hits.sort()
        for i, (a, b, name) in enumerate(hits):
            # A signature marker immediately around the name outranks everything:
            # "KPMG Inc. / Registered Auditor / Per: ..." is the signing block even
            # if a sentence about another firm's tax work follows shortly after.
            sig_zone = low[max(0, a - 120):min(len(low), b + 160)]
            signed = bool(SIGNATURE.search(sig_zone))
            # The advisory pattern is "tax services were provided by <FIRM>", so look
            # mostly BEHIND the name, not ahead of it.
            near = low[max(0, a - 150):min(len(low), b + 40)]
            if ADVISORY.search(near) and not signed:
                score[name] = score.get(name, 0) - 6
                continue
            pts = 1
            if signed:
                pts += 5
            if i == 0:
                pts += 3
            # take the BEST single appearance, not the sum: the signing auditor is
            # named once in the opinion block, while an adviser may be named several
            # times in the fee note, and summing would let repetition win
            if pts <= score.get(name, 0):
                continue
            score[name] = pts
            if name not in ev or pts >= 6:
                x, y = max(0, a - 70), min(len(ch), b + 70)
                ev[name] = re.sub(r"\s+", " ", ch[x:y]).strip()

    live = {k: v for k, v in score.items() if v > 0}
    if not live:
        return None, None, None, "none"
    ranked = sorted(live.items(), key=lambda x: -x[1])
    best, n = ranked[0]
    runner = ranked[1][1] if len(ranked) > 1 else 0
    conf = "high" if n >= 6 and n >= runner + 3 else "low"
    return best, int(best in BIG4), ev.get(best, ""), conf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--review", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    rows = []
    for fp in sorted((out / "text").glob("*.json")):
        rec = json.load(open(fp, encoding="utf-8"))
        full = rec.get("full_text") or ""
        chunks = auditor_windows(full)
        if not chunks:
            rows.append(dict(firm_id=rec["firm_id"], year=rec["year"], auditor="",
                             big4="", method="no auditor report in document",
                             confidence="", evidence=""))
            continue
        name, b4, evd, conf = identify(chunks)
        rows.append(dict(firm_id=rec["firm_id"], year=rec["year"],
                         auditor=name or "", big4="" if name is None else b4,
                         method="auditor context" if name else "firm not recognised",
                         confidence=conf, evidence=evd or ""))

    df = pd.DataFrame(rows).sort_values(["firm_id", "year"])
    df.to_csv(out / "auditor.csv", index=False)

    ident = df[df["auditor"] != ""]
    print(f"Documents scanned      : {len(df)}")
    print(f"Auditor identified     : {len(ident)}  ({len(ident)/max(len(df),1):.0%})")
    if len(ident):
        print(f"  Big 4                : {(ident['big4']==1).sum()}")
        print(f"  non-Big 4            : {(ident['big4']==0).sum()}")
        print(f"  low confidence       : {(ident['confidence']=='low').sum()}")
        print("\nBy firm:")
        print(ident["auditor"].value_counts().to_string())

    need = df[(df["auditor"] == "") | (df["confidence"] == "low")]
    print(f"\nNeeds manual check     : {len(need)}")
    if a.review and len(need):
        print(need[["firm_id", "year", "method"]].to_string(index=False))

    # a switch mid-panel is either real or a detection error - always worth checking
    sw = []
    for fid, g in ident.groupby("firm_id"):
        u = g.sort_values("year")["auditor"].tolist()
        changes = sum(1 for i in range(1, len(u)) if u[i] != u[i - 1])
        if changes:
            sw.append((fid, changes, " -> ".join(dict.fromkeys(u))))
    if sw:
        print(f"\nFirms with an auditor change ({len(sw)}) - verify each:")
        for fid, n, seq in sw[:20]:
            print(f"  {fid}: {n} change(s)  {seq[:90]}")

    print(f"\nSaved -> {out/'auditor.csv'}")
    print("Standalone reserve statements carry no auditor's report - those rows are")
    print("blank by design. Fill them from the matching annual report or from S&P.")


if __name__ == "__main__":
    main()
