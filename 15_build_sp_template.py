#!/usr/bin/env python3
"""
Step 15 - Build the S&P collection template, keyed to the analysis sample.

Reads out/panel.csv and writes an Excel workbook with one pre-keyed row per
firm-year, so the extract is bounded to the firms and years actually in the
sample rather than the full frame. Blank input cells are shaded; everything else
is either a key or a note.

Usage:
    python 15_build_sp_template.py --out out                      # Block A only
    python 15_build_sp_template.py --out out --blocks a,b,c,d     # all blocks
    python 15_build_sp_template.py --out out --names frame.csv    # add company names
"""
import argparse
from pathlib import Path

import pandas as pd
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

NAVY, GOLD = "1F3864", "C9A227"
INPUT = PatternFill("solid", fgColor="FFF9E6")     # you type here
KEY = PatternFill("solid", fgColor="E7E6E6")       # pre-filled, do not edit
THIN = Side(style="thin", color="B8C2D6")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

BLOCKS = {
 "a": ("Block A Valuation", [
   ("market_cap", "Market value of common equity, 3 months after fiscal year end", 34),
   ("shares_outstanding", "Common shares in issue at reporting date, excl. treasury", 34),
   ("book_equity", "Total common equity", 26),
   ("eps_before_xo", "Diluted EPS before extraordinary items", 30),
   ("share_price", "Closing price at the measurement date", 28),
   ("currency", "Reporting currency (ISO code)", 22),
   ("units", "ones / thousands / millions as reported", 26),
   ("source_note", "Anything unusual about this row", 30)]),
 "b": ("Block B Controls", [
   ("total_assets", "Balance sheet total -> SIZE (log) and the Tobin's Q denominator", 30),
   ("total_debt", "Short-term plus long-term borrowings -> LEV. Note in source_note "
                  "whether lease liabilities are included (IFRS 16 changed this mid-panel)", 38),
   ("capex", "Purchase of property, plant and equipment -> capex intensity", 32),
   ("auditor_sp", "External auditor per S&P -> cross-check on the BIG4 variable, "
                  "which carries H2", 36),
   ("source_note", "Anything unusual about this row", 28)]),
 "c": ("Block C Reserves", [
   ("reserve_tonnes_attrib", "Proven + probable, attributable, firm total", 34),
   ("primary_commodity", "Commodity with the largest share of reserve tonnage", 34),
   ("multi_commodity", "1 if reserves in more than one commodity group", 32),
   ("production_tonnes", "Annual production, same basis as reserves", 30),
   ("tonnage_unit", "Mt / kt / oz / lb as reported", 24),
   ("source_note", "Property-level detail or aggregation notes", 34)]),
 "d": ("Block D Firm static", [
   ("ultimate_parent", "Ultimate parent -> group-level clustering, since several sampled "
                       "firms are majority-held subsidiaries of others in the sample", 42),
   ("producer_status_note", "Year production commenced, if within the window -> PRODUCER", 40)]),
}


def header(ws, cols, row=1):
    for c, (name, note, width) in enumerate(cols, 1):
        cell = ws.cell(row, c, name)
        cell.font = Font(bold=True, color="FFFFFF", size=10)
        cell.fill = PatternFill("solid", fgColor=NAVY)
        cell.alignment = Alignment(wrap_text=True, vertical="center")
        cell.border = BORDER
        ws.column_dimensions[get_column_letter(c)].width = width
        n = ws.cell(row + 1, c, note)
        n.font = Font(italic=True, size=8, color="595959")
        n.alignment = Alignment(wrap_text=True, vertical="top")
        n.border = BORDER
    ws.row_dimensions[row].height = 30
    ws.row_dimensions[row + 1].height = 34
    ws.freeze_panes = ws.cell(row + 2, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--blocks", default="a")
    ap.add_argument("--names", default=None,
                    help="CSV with entity_id and company, to label the rows")
    ap.add_argument("--file", default="SP_Collection_Template.xlsx")
    a = ap.parse_args()

    panel = pd.read_csv(Path(a.out) / "panel.csv")
    keys = panel[["firm_id", "year"]].drop_duplicates().sort_values(["firm_id", "year"])
    keys["firm_id"] = keys["firm_id"].astype(int)
    keys["year"] = keys["year"].astype(int)

    names = {}
    if a.names and Path(a.names).exists():
        nf = pd.read_csv(a.names)
        idc = "entity_id" if "entity_id" in nf.columns else "firm_id"
        nmc = "company" if "company" in nf.columns else nf.columns[1]
        names = {int(float(r[idc])): r[nmc] for _, r in nf.iterrows()
                 if pd.notna(r[idc])}

    wb = openpyxl.Workbook()
    rd = wb.active
    rd.title = "README"
    rd.column_dimensions["A"].width = 110
    lines = [
     ("S&P CAPITAL IQ COLLECTION TEMPLATE", 15, True, NAVY),
     (f"{len(keys)} firm-years, {keys.firm_id.nunique()} firms, "
      f"{keys.year.min()}\u2013{keys.year.max()}", 11, False, "404040"),
     ("", 10, False, "000000"),
     ("HOW TO USE", 12, True, NAVY),
     ("Rows are pre-keyed to the analysis sample. Do not add, delete or reorder rows: the "
      "merge is on entity_id and year, and reordering breaks nothing but inserting does.", 10, False, "000000"),
     ("Shaded input columns are for you to fill. Grey key columns are fixed.", 10, False, "000000"),
     ("Leave a cell EMPTY where S&P has no value. Do not enter 0, NA, n/a or a dash: "
      "zero is a real value and would be read as one.", 10, False, "C00000"),
     ("Record currency and units for every row. A misread scale is the most damaging "
      "error in this file and it looks entirely plausible in a regression.", 10, False, "C00000"),
     ("", 10, False, "000000"),
     ("EXTRACTION DISCIPLINE", 12, True, NAVY),
     ("Extract each block in ONE pass for all firms and years, then save as a dated CSV. "
      "Do not refresh incrementally into a populated sheet \u2014 that was the cause of the "
      "earlier duplication incident.", 10, False, "000000"),
     ("After each block, export to CSV and run:  python scripts/09_reconcile_financials.py "
      "--sp <file>.csv --out out --gaps-only", 10, False, "000000"),
     ("That flags values repeated across consecutive years, reports missing cells by "
      "variable, and writes a gap list for the annual-report extractor.", 10, False, "000000"),
     ("", 10, False, "000000"),
     ("WHAT IS AND IS NOT COLLECTED", 12, True, NAVY),
     ("Every field here is used in the specification. Revenue, EBIT and cash were dropped: "
      "in a price-level residual income model profitability enters through EPS, so ROA "
      "would be largely redundant, and producer status comes from S&P directly rather "
      "than being inferred from revenue.", 10, False, "000000"),
     ("Primary exchange and country of incorporation are not collected again \u2014 they are "
      "already in the revised frame, and firm fixed effects absorb them.", 10, False, "000000"),
     ("", 10, False, "000000"),
     ("MEASUREMENT POINTS", 12, True, NAVY),
     ("Market capitalisation: three months after fiscal year end, applied consistently. "
      "Fiscal years differ across the sample (June for several South African firms, "
      "December for others), so use each firm's own year end.", 10, False, "000000"),
     ("Shares outstanding: at the reporting date, excluding treasury shares.", 10, False, "000000"),
     ("Total debt: short-term plus long-term borrowings. Note in source_note whether "
      "lease liabilities are included.", 10, False, "000000"),
    ]
    rr = 1
    for text, size, bold, colour in lines:
        c = rd.cell(rr, 1, text)
        c.font = Font(size=size, bold=bold, color=colour)
        c.alignment = Alignment(wrap_text=True, vertical="top")
        rd.row_dimensions[rr].height = 34 if len(text) > 95 else (20 if text else 8)
        rr += 1

    want = [b.strip().lower() for b in a.blocks.split(",") if b.strip()]
    for b in want:
        if b not in BLOCKS:
            continue
        title, fields = BLOCKS[b]
        ws = wb.create_sheet(title[:31])
        keycols = [("entity_id", "S&P company id", 12), ("company", "name", 30)]
        if b != "d":
            keycols.append(("year", "fiscal year", 10))
        header(ws, keycols + fields)
        rows = (keys if b != "d"
                else keys[["firm_id"]].drop_duplicates().assign(year=None))
        rnum = 3
        for _, k in rows.iterrows():
            ws.cell(rnum, 1, int(k["firm_id"])).fill = KEY
            ws.cell(rnum, 2, names.get(int(k["firm_id"]), "")).fill = KEY
            col0 = 3
            if b != "d":
                ws.cell(rnum, 3, int(k["year"])).fill = KEY
                col0 = 4
            for c in range(1, col0):
                ws.cell(rnum, c).border = BORDER
            for c in range(col0, col0 + len(fields)):
                cell = ws.cell(rnum, c)
                cell.fill = INPUT
                cell.border = BORDER
            rnum += 1
        if b == "c":
            col = get_column_letter(col0 + 2)      # multi_commodity
            dv2 = DataValidation(type="list", formula1='"0,1"', allow_blank=True)
            ws.add_data_validation(dv2)
            dv2.add(f"{col}3:{col}{rnum-1}")

    wb.save(a.file)
    print(f"{len(keys)} firm-years, {keys.firm_id.nunique()} firms")
    print("Blocks written:", ", ".join(BLOCKS[b][0] for b in want if b in BLOCKS))
    print(f"Saved -> {a.file}")


if __name__ == "__main__":
    main()
