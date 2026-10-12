#!/usr/bin/env python3
"""
Step 19 - Apply the re-pulled S&P valuation figures for corrected firms.

The validation extract is reported on a different scale from the main Block A
sheet, so the figures are converted before they replace the existing rows:

    market_cap          USD millions   -> unchanged
    shares_outstanding  absolute units -> divided by 1,000,000 (millions)
    book_equity         USD thousands  -> divided by 1,000 (millions)
    eps_before_xo       USD per share  -> unchanged
    share_price         USD per share  -> unchanged

The conversion is verified before writing: market cap divided by shares must
reproduce the share price. Any firm-year failing that check is reported and the
run stops, rather than silently replacing good rows with mis-scaled ones.

Usage:
    python 19_apply_corrections.py --xlsx Data_Validation_for_3_companies.xlsx
    python 19_apply_corrections.py --xlsx <file> --dry-run
"""
import argparse, shutil
from pathlib import Path

import numpy as np
import pandas as pd

NUM = ["market_cap", "shares_outstanding", "book_equity", "eps_before_xo", "share_price"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xlsx", default="Data_Validation_for_3_companies.xlsx")
    ap.add_argument("--sheet", default="Sheet1")
    ap.add_argument("--out", default="out")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)

    new = pd.read_excel(a.xlsx, sheet_name=a.sheet, header=0, skiprows=[1])
    new.columns = [str(c).strip() for c in new.columns]
    new = new.rename(columns={"entity_id": "firm_id"})

    # the year column comes back as a fiscal year-end date
    yr = pd.to_datetime(new["year"], errors="coerce")
    new["year"] = np.where(yr.notna(), yr.dt.year,
                           pd.to_numeric(new["year"], errors="coerce"))
    new = new[new["year"].notna()]
    new["year"] = new["year"].astype(int)
    new["firm_id"] = pd.to_numeric(new["firm_id"], errors="coerce").astype("Int64")
    for c in NUM:
        new[c] = pd.to_numeric(new[c], errors="coerce")

    # ---- convert to the panel's units ----
    new["shares_outstanding"] = new["shares_outstanding"] / 1e6   # -> millions
    new["book_equity"] = new["book_equity"] / 1e3                 # -> USD millions

    # ---- verify: market cap / shares must reproduce the share price ----
    implied = new["market_cap"] / new["shares_outstanding"]
    ratio = implied / new["share_price"]
    bad = new[(ratio < 0.98) | (ratio > 1.02)]
    print(f"Rows in the correction file : {len(new)}")
    print(f"Firms                       : {sorted(new.firm_id.dropna().unique().tolist())}")
    print(f"Identity check (market cap / shares vs share price):")
    print(f"  median ratio {ratio.median():.4f}, outside 2% band: {len(bad)}")
    if len(bad):
        print(bad[["firm_id", "year", "market_cap", "shares_outstanding",
                   "share_price"]].to_string(index=False))
        raise SystemExit("Conversion does not reconcile - not writing.")

    old = pd.read_csv(out / "sp_valuation.csv")
    old["firm_id"] = pd.to_numeric(old["firm_id"], errors="coerce").astype("Int64")
    old["year"] = pd.to_numeric(old["year"], errors="coerce").astype("Int64")

    firms = set(new.firm_id.dropna().tolist())
    before = old[old.firm_id.isin(firms)]
    print(f"\nExisting rows for these firms: {len(before)}  -> replaced by {len(new)}")

    keep_cols = [c for c in old.columns if c in new.columns or c in
                 ("firm_id", "year", "company", "currency", "units_monetary",
                  "units_per_share", "source")]
    meta = old[["firm_id", "year", "company"]] if "company" in old.columns else None
    merged = old[~old.firm_id.isin(firms)].copy()

    add = new[["firm_id", "year"] + NUM].copy()
    if meta is not None:
        add = add.merge(meta, on=["firm_id", "year"], how="left")
    add["currency"] = "USD"
    add["units_monetary"] = "millions"
    add["units_per_share"] = "absolute"
    add["source"] = "S&P Capital IQ Pro (re-pulled)"

    result = pd.concat([merged, add], ignore_index=True).sort_values(
        ["firm_id", "year"])

    print(f"Result: {len(result)} rows, {result.firm_id.nunique()} firms")
    for fid in sorted(firms):
        g = result[result.firm_id == fid]
        print(f"\n  {fid}: {len(g)} rows, "
              f"market cap {g.market_cap.min():,.1f} to {g.market_cap.max():,.1f} USD m")

    if a.dry_run:
        print("\nDRY RUN - nothing written.")
        return

    shutil.copyfile(out / "sp_valuation.csv", out / "sp_valuation_before_corrections.csv")
    result.to_csv(out / "sp_valuation.csv", index=False)
    print(f"\nSaved -> {out/'sp_valuation.csv'}")
    print(f"Previous version kept as {out/'sp_valuation_before_corrections.csv'}")
    print("\nNow rerun: python scripts/18_build_analysis_dataset.py")


if __name__ == "__main__":
    main()
