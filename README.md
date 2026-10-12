# RDQI
Mineral Reserves Disclosure Index
# RDQI coding pipeline

Builds a Reserve Disclosure Quality Index (RDQI) and separate disclosure quantity
measures from mining reserve disclosure PDFs, for the firm-value study on listed
companies with material African operations, 2012-2025.

Quality is scored 0/1/2 against a fixed rubric (`rubric.json`). Quantity is measured
separately as item counts and word counts, so the two constructs are not the same
variable by construction.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...                        # Windows: set ANTHROPIC_API_KEY=...
```

## Folder layout

```
rdqi/
  rubric.json            <- the 11 items, dimensions, 0/1/2 anchors, keywords
  scripts/               <- the six steps
  data/pdfs/             <- your PDFs, named FIRMID_YEAR.pdf  e.g. GFI_2019.pdf
  data/manifest.csv      <- optional: firm_id,year,file,doc_type
  data/sp_financials.csv <- optional: your S&P extract, keyed firm_id,year
  out/                   <- everything the pipeline produces
```

`doc_type` is `standalone` (separate reserve statement) or `embedded` (section inside
an annual/integrated report). The script guesses it if you don't supply a manifest,
but supplying it is safer because it becomes a control variable.

## The six steps

### Step 0 â€” Collect the PDFs

Download standalone Mineral Resources and Mineral Reserves statements where they
exist: JSE company websites and SENS, EDGAR full-text search for Exhibit 96 (S-K
1300), SEDAR+ for NI 43-101, ASX announcements for JORC. Only fall back to the full
annual report when there is no standalone statement. Name every file `FIRMID_YEAR.pdf`.

### Step 1 â€” Extract text and isolate the reserve section

```bash
python scripts/01_extract.py --pdf-dir data/pdfs --out out --manifest data/manifest.csv
```

Uses PyMuPDF. Finds the reserve section by heading match, falling back to the densest
run of pages containing reserve cues. Writes one JSON per document plus
`out/extract_log.csv`.

**Check this log before going further.** Three flags need your eyes:
`section_found=False` (fix by adding the firm's heading wording to `START_PATTERNS`),
`needs_ocr=True` (scanned PDF â€” run `ocrmypdf in.pdf out.pdf` and re-extract), and
`section_method=cue_density` (heading not matched, so verify the page range is right).
Expect to hand-fix 5-10% of documents; older filings from 2012-2015 are the worst.

### Step 2 â€” Quantity measures

```bash
python scripts/02_quantity.py --out out
```

Produces `out/quantity.csv`: `n_words`, `ln_words`, `n_numbers`, `n_table_rows`,
`standalone`, and a keyword presence screen `kw_items` (0-11). `kw_items` is your
DQUANT control. Cheap, deterministic, no API needed.

### Step 3 â€” Quality scoring against the rubric

```bash
python scripts/03_score_quality.py --out out --mock          # dry run, no API cost
python scripts/03_score_quality.py --out out --limit 20      # pilot on 20 documents
python scripts/03_score_quality.py --out out                 # full run
python scripts/03_score_quality.py --out out --runs 2        # self-consistency
```

Temperature 0, fixed rubric, one API call per document. Every non-zero score must
carry a verbatim evidence quote, written to `out/quality_evidence.csv` so any score
can be audited back to the text. Results are cached on a hash of the prompt, so
re-running costs nothing for documents already scored.

**Pilot first.** Run `--limit 20`, read `quality_evidence.csv` line by line, and fix
the rubric anchors wherever the evidence does not justify the score. Rubric revisions
are cheap now and expensive after you have scored 1,000 documents.

### Step 4 â€” Draw the validation sample

```bash
python scripts/04_sample_for_handcoding.py --out out --frac 0.12 --seed 20260912
```

Stratified by year block and document type. Draw it **before** looking at machine
scores and report the seed in your methodology section. Writes a blank
`out/handcoding_sheet.csv`; fill in `human_score` (0/1/2) yourself from the PDFs
without consulting the machine scores.

### Step 5 â€” Agreement statistics

```bash
python scripts/05_agreement.py --out out
python scripts/05_agreement.py --out out --self-consistency
```

Reports exact agreement, Cohen's kappa and quadratic-weighted kappa overall and per
item, plus the confusion matrix. **Report weighted kappa** â€” the scale is ordinal, so
a 1-versus-2 disagreement should not count the same as 0-versus-2.

Benchmarks: weighted kappa above 0.80 is publishable as is; 0.60-0.80 is acceptable if
you discuss it; below 0.60 on an item means the anchors are ambiguous â€” rewrite them
and re-score that item. The script lists the three weakest items for you.

### Step 6 â€” Build the panel

```bash
python scripts/06_build_panel.py --out out --sp data/sp_financials.csv
```

Produces `out/panel.csv` with RDQI (0-22), RDQI_pct, the four dimension subscales,
DQUANT, RDQI_resid (quality orthogonal to quantity, for the robustness test), and
one-year lags ready to regress on firm value. Also prints Cronbach's alpha and the
quality-quantity correlation.

If that correlation exceeds about 0.85, quality and quantity are not empirically
distinct and your headline claim is in trouble. The fix is in the rubric: the 0/1/2
anchors need to reward quantified, company-specific content rather than presence.

## Things reviewers will ask

- **Why an LLM?** Answer with the validation statistics, not with the method. Report
  the kappa, the subsample size, the seed, and the confusion matrix.
- **Is it reproducible?** Temperature 0, fixed rubric, cached prompts, pinned model
  version. State the model version and the date range of scoring in the paper.
- **Did the model hallucinate evidence?** Spot-check `quality_evidence.csv`: search
  each quote in the source text. Add a column reporting the share of quotes verified.
- **Boilerplate drift.** Firms copy their own prior-year wording. A within-firm
  disclosure change is therefore meaningful; report how often RDQI changes at all.

## Cost and runtime

About 1,000 firm-years at roughly 10-20k input tokens each is a few hours of wall
time and a modest API bill; the cache means a re-run after a rubric fix only re-scores
what changed. Steps 1, 2, 4, 5 and 6 are free and run in minutes.

