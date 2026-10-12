#!/usr/bin/env python3
"""
Step 1 - Extract text from reserve disclosure PDFs and isolate the reserve section.

Input : a folder of PDFs named  <FIRMID>_<YEAR>.pdf   e.g.  GFI_2019.pdf
        (optionally a manifest.csv with columns firm_id,year,file,doc_type)
Output: one JSON per document in out/text/, plus out/extract_log.csv

Usage:
    python 01_extract.py --pdf-dir data/pdfs --out out
    python 01_extract.py --pdf-dir data/pdfs --out out --manifest data/manifest.csv
"""
import argparse, csv, json, os, re, sys
from pathlib import Path

import fitz  # PyMuPDF

# Headings that typically start a reserve/resource disclosure section
START_PATTERNS = [
    r"mineral\s+(resources?\s+and\s+)?mineral\s+reserves?",
    r"mineral\s+reserves?\s+and\s+mineral\s+resources?",
    r"(ore|mineral)\s+reserves?\s+statement",
    r"resources?\s+and\s+reserves?\s+(statement|summary|report)",
    r"reserves?\s+and\s+resources?\s+(statement|summary|report)",
    r"statement\s+of\s+(mineral\s+)?reserves?",
    r"summary\s+of\s+mineral\s+reserves?",
    # French
    r"r[ée]serves?\s+(min[ée]rales?|mini[èe]res?)",
    r"ressources?\s+(et\s+r[ée]serves?\s+)?min[ée]rales?",
    r"ressources?\s+min[ée]rales?\s+et\s+r[ée]serves?",
    r"[ée]tat\s+des\s+r[ée]serves?",
    r"d[ée]claration\s+des\s+(ressources?|r[ée]serves?)",
    r"rapport\s+sur\s+les\s+(ressources?|r[ée]serves?)\s+min[ée]rales?",
]
# Headings that typically mark the end of the section
END_PATTERNS = [
    r"^\s*(consolidated\s+)?(annual\s+)?financial\s+statements",
    r"^\s*directors[’'`]?\s+report",
    r"^\s*corporate\s+governance",
    r"^\s*shareholder\s+information",
    r"^\s*notice\s+of\s+annual\s+general\s+meeting",
    r"^\s*independent\s+auditor[’'`]?s\s+report",
    r"^\s*sustainability\s+(report|review)",
    # French
    r"^\s*[ée]tats\s+financiers",
    r"^\s*rapport\s+(annuel\s+)?des?\s+administrateurs",
    r"^\s*gouvernance\s+d.?entreprise",
    r"^\s*informations?\s+(aux\s+)?actionnaires",
    r"^\s*avis\s+de\s+convocation",
    r"^\s*rapport\s+de\s+l.?auditeur\s+ind[ée]pendant",
    r"^\s*rapport\s+(de\s+)?d[ée]veloppement\s+durable",
]
START_RE = re.compile("|".join(START_PATTERNS), re.I)
END_RE = re.compile("|".join(END_PATTERNS), re.I | re.M)

# A page must contain at least this many reserve-ish cues to count as in-section
CUE_RE = re.compile(
    # English
    r"mineral\s+reserve|ore\s+reserve|mineral\s+resource|proved|proven|probable|"
    r"measured|indicated|inferred|competent\s+person|qualified\s+person|"
    # French
    r"r[ée]serves?\s+min[ée]rale|r[ée]serves?\s+mini[èe]re|ressources?\s+min[ée]rale|"
    r"prouv[ée]e|mesur[ée]e|indiqu[ée]e|pr[ée]sum[ée]e|inf[ée]r[ée]e|"
    r"personne\s+comp[ée]tente|personne\s+qualifi[ée]e|teneur\s+de\s+coupure",
    re.I,
)

MIN_CHARS_PER_PAGE = 60  # below this a page is probably a scan needing OCR


def clean(text: str) -> str:
    text = text.replace("\u00ad", "")                  # soft hyphens
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)       # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def page_texts(path: Path):
    doc = fitz.open(path)
    pages = []
    for pg in doc:
        # "text" mode keeps reading order; use "blocks" if columns are jumbled
        pages.append(clean(pg.get_text("text")))
    doc.close()
    return pages


MIN_SECTION_CHARS = 3000     # below this a heading match is treated as a false start


def _cue_density(pages):
    """Longest run of pages carrying several reserve cues."""
    flags = [len(CUE_RE.findall(t)) >= 3 for t in pages]
    if not any(flags):
        return None, None
    best, cur = (0, 0, 0), None
    for i, f in enumerate(flags + [False]):
        if f and cur is None:
            cur = i
        elif not f and cur is not None:
            if i - cur > best[0]:
                best = (i - cur, cur, i)
            cur = None
    return best[1], best[2]


def find_section(pages):
    """Return (start_idx, end_idx_exclusive, method)."""
    n = len(pages)
    # 1. heading match near the top of a page
    starts = [i for i, t in enumerate(pages) if START_RE.search(t[:600])]
    for s in starts:                      # try each heading hit, not just the first
        end = n
        for j in range(s + 1, n):
            head = pages[j][:400]
            if END_RE.search(head) and not CUE_RE.search(head):
                end = j
                break
        if len("".join(pages[s:end])) >= MIN_SECTION_CHARS:
            return s, end, "heading"
        # else: this heading was a contents-page entry or a one-line cross
        # reference, and the next page tripped an end-marker. Keep looking.

    if starts:
        # every heading hit produced a stub: fall back to cue density, and if
        # that fails too, take from the first heading to the end of the document
        a, b = _cue_density(pages)
        if a is not None and len("".join(pages[a:b])) >= MIN_SECTION_CHARS:
            return a, b, "cue_density_after_stub"
        return starts[0], n, "heading_to_end"

    # 2. no heading anywhere: fall back to the densest run of reserve cues
    a, b = _cue_density(pages)
    if a is None:
        return None, None, "not_found"
    return a, b, "cue_density"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf-dir", required=True)
    ap.add_argument("--out", default="out")
    ap.add_argument("--manifest", default=None,
                    help="optional CSV: firm_id,year,file,doc_type(standalone|embedded)")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "text").mkdir(parents=True, exist_ok=True)

    meta = {}
    if args.manifest:
        with open(args.manifest, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                meta[row["file"]] = row

    log = []
    pdfs = sorted(Path(args.pdf_dir).glob("*.pdf"))
    if not pdfs:
        sys.exit(f"No PDFs found in {args.pdf_dir}")

    for p in pdfs:
        m = meta.get(p.name, {})
        stem = re.match(r"(?P<firm>[A-Za-z0-9\.\-]+)_(?P<year>\d{4})", p.stem)
        firm = m.get("firm_id") or (stem.group("firm") if stem else p.stem)
        year = int(m.get("year") or (stem.group("year") if stem else 0))

        try:
            pages = page_texts(p)
        except Exception as e:                       # corrupt / encrypted file
            log.append(dict(file=p.name, firm_id=firm, year=year, status=f"ERROR: {e}"))
            continue

        n_pages = len(pages)
        empty = sum(1 for t in pages if len(t) < MIN_CHARS_PER_PAGE)
        needs_ocr = n_pages > 0 and empty / n_pages > 0.5

        s, e, method = find_section(pages)
        section = "\n\n".join(pages[s:e]) if s is not None else ""

        doc_type = m.get("doc_type")
        if not doc_type:
            doc_type = "standalone" if (s == 0 and (e or 0) >= n_pages - 1) else "embedded"

        rec = dict(firm_id=firm, year=year, file=p.name, n_pages=n_pages,
                   section_found=s is not None, section_method=method,
                   section_start_page=(s + 1) if s is not None else None,
                   section_end_page=e if e is not None else None,
                   needs_ocr=needs_ocr, doc_type=doc_type,
                   reserve_text=section, full_text="\n\n".join(pages))

        with open(out / "text" / f"{firm}_{year}.json", "w", encoding="utf-8") as f:
            json.dump(rec, f, ensure_ascii=False)

        log.append({k: v for k, v in rec.items() if k not in ("reserve_text", "full_text")}
                   | {"section_chars": len(section), "status": "ok"})
        print(f"{p.name:40s} pages={n_pages:4d} section={method:12s} chars={len(section):7d}"
              + ("  << OCR NEEDED" if needs_ocr else "")
              + ("  << SECTION NOT FOUND" if s is None else "")
              + ("  << SHORT - CHECK THIS" if s is not None
                 and len(section) < MIN_SECTION_CHARS else ""))

    with open(out / "extract_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
        w.writeheader()
        w.writerows(log)

    bad = [r for r in log if r.get("status") != "ok" or not r.get("section_found")
           or r.get("needs_ocr")]
    print(f"\nDone: {len(log)} documents. Review manually: {len(bad)}")
    print(f"Log: {out/'extract_log.csv'}")


if __name__ == "__main__":
    main()
