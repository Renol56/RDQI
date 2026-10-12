#!/usr/bin/env python3
"""
Step 21 - Apply re-pulled Block B figures for a corrected firm.

Replaces total_assets, total_debt and capex in the Block B sheet for the firm-years
present in the correction file, and writes a new workbook. The original is untouched.

Checks before writing:
  * leverage (total debt / total assets) must be plausible, under 3
  * capex must be a cash outflow, i.e. negative, consistent with the rest of Block B
  * the firm-years must exist in the sheet being corrected

Usage:
    python 21_apply_blockb_correction.py --xlsx SP_Blocks_BC.xlsx \\
        --correction "Endeavor_re-pulled.xlsx" --firm 4349981 --dry-run
    python 21_apply_blockb_correction.py --xlsx SP_Blocks_BC.xlsx \\
        --correction "Endeavor_re-pulled.xlsx" --firm 4349981
"""
import argparse, shutil
from pathlib import Path

import pandas as pd
import openpyxl

FIELDS = ["total_assets", "total_debt", "capex"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="SP_Blocks_BC.xlsx")
    ap.add_argument("--sheet", default="Block B Controls")
    ap.add_argument("--correction", required=True)
    ap.add_argument("--firm", type=int, required=True)
    ap.add_argument("--header-row", type=int, default=2,
                    help="0-based row in the correction file holding the headers")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    # ---- read the correction ----
    # S&P exports carry a hidden "__snloffice" sheet first; take the first real one
    xl = pd.ExcelFile(a.correction)
    sheet = next((s for s in xl.sheet_names if not str(s).startswith("__")),
                 xl.sheet_names[0])
    new = pd.read_excel(xl, sheet_name=sheet, header=a.header_row)
    new.columns = [str(c).strip() for c in new.columns]
    ycol = "year" if "year" in new.columns else new.columns[1]
    new = new.rename(columns={ycol: "year"})
    yr = pd.to_datetime(new["year"], errors="coerce")
    new["year"] = yr.dt.year.fillna(pd.to_numeric(new["year"], errors="coerce"))
    new = new[new["year"].notna()].copy()
    new["year"] = new["year"].astype(int)
    for c in FIELDS:
        if c not in new.columns:
            raise SystemExit(f"'{c}' not found in the correction file. "
                             f"Columns: {list(new.columns)}")
        new[c] = pd.to_numeric(new[c], errors="coerce")

    lev = new["total_debt"] / new["total_assets"]
    print(f"Correction rows: {len(new)}  years {new.year.min()}-{new.year.max()}")
    print(f"Leverage: min {lev.min():.3f}  median {lev.median():.3f}  max {lev.max():.3f}")
    if lev.max() > 3:
        raise SystemExit("Leverage above 3 in the correction file - check the scale "
                         "before writing.")
    neg = (new["capex"] < 0).mean()
    print(f"Capex negative in {neg:.0%} of rows "
          f"({'consistent with Block B' if neg > 0.8 else 'CHECK THE SIGN'})")

    # ---- locate the columns in the Block B sheet ----
    wb = openpyxl.load_workbook(a.xlsx)
    ws = wb[a.sheet]
    hdr = {str(ws.cell(1, c).value).strip(): c for c in range(1, ws.max_column + 1)
           if ws.cell(1, c).value}
    need = ["entity_id", "year"] + FIELDS
    missing = [c for c in need if c not in hdr]
    if missing:
        raise SystemExit(f"Columns not found in '{a.sheet}': {missing}")

    lookup = {int(r["year"]): r for _, r in new.iterrows()}
    written, unmatched = 0, []
    for row in range(3, ws.max_row + 1):
        eid = ws.cell(row, hdr["entity_id"]).value
        yv = ws.cell(row, hdr["year"]).value
        if eid is None or yv is None:
            continue
        try:
            if int(float(eid)) != a.firm:
                continue
            y = int(float(yv))
        except (TypeError, ValueError):
            continue
        rec = lookup.get(y)
        if rec is None:
            unmatched.append(y)
            continue
        if not a.dry_run:
            for c in FIELDS:
                ws.cell(row, hdr[c]).value = float(rec[c])
        written += 1

    print(f"\nFirm {a.firm}: {written} rows updated")
    if unmatched:
        print(f"  years in the sheet with no correction: {sorted(unmatched)}")
    extra = sorted(set(lookup) - {int(float(ws.cell(r, hdr['year']).value))
                                  for r in range(3, ws.max_row + 1)
                                  if ws.cell(r, hdr['entity_id']).value is not None
                                  and str(int(float(ws.cell(r, hdr['entity_id']).value)))
                                  == str(a.firm)
                                  and ws.cell(r, hdr['year']).value is not None})
    if extra:
        print(f"  years in the correction not in the sheet: {extra}")

    if a.dry_run:
        print("\nDRY RUN - nothing written.")
        return

    dst = Path(a.xlsx).with_name(Path(a.xlsx).stem + "_corrected.xlsx")
    wb.save(dst)
    print(f"\nSaved -> {dst}")
    print("Rebuild with:")
    print(f"  python scripts/18_build_analysis_dataset.py --blockb \"{dst.name}\"")


if __name__ == "__main__":
    main()
