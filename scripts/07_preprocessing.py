from pathlib import Path
import numpy as np
import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def snake(df):
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return df

sales = snake(pd.read_csv(RAW / "retail_sales_ml_apl.csv", low_memory=False))
inv = snake(pd.read_csv(RAW / "retail_inventory_ml_apl.csv", low_memory=False))
print(f"Loaded sales: {len(sales):,} rows | inventory: {len(inv):,} rows")

# ------------------------------------------------------------------ A
section("A. PRODUCT KEY CONSISTENCY ACROSS FILES")
s_first = sales.groupby("product_no")[["product_description", "product_category"]].first()
i_first = inv.groupby("product_no")[["product_description", "product_category"]].first()
both = s_first.join(i_first, how="inner", lsuffix="_s", rsuffix="_i")
print(f"Product No in sales: {len(s_first):,} | in inventory: {len(i_first):,} | in both: {len(both):,}")
print(f"Same description in both files: {(both.product_description_s == both.product_description_i).mean() * 100:.2f}%")
print(f"Same category in both files:    {(both.product_category_s == both.product_category_i).mean() * 100:.2f}%")
multi = (inv.groupby("product_no")["product_description"].nunique() > 1).sum()
print(f"Inventory Product No with more than 1 description: {multi:,}")
print("-> Both match rates should be 100%. If not, STOP and send me this output.")

# ------------------------------------------------------------------ B
section("B. CLEAN SALES")
sales["transaction_date"] = pd.to_datetime(sales["transaction_date"])

zero_rows = (sales["qty_sold"] == 0).sum()
print(f"Rows with zero quantity removed: {zero_rows}")
sales = sales[sales["qty_sold"] != 0].copy()

print("\nIs Return vs negative quantity (should line up exactly):")
print(pd.crosstab(sales["is_return"], sales["qty_sold"] < 0).to_string())

sales = sales.drop(columns=["reason_of_return", "sales_channel"])
sales["gross_qty"] = sales["qty_sold"].clip(lower=0)
sales["return_qty"] = (-sales["qty_sold"]).clip(lower=0)

dups = sales.duplicated().sum()
print(f"\nFully identical rows (kept, they can be real repeat sales): {dups:,}")
print(f"Clean sales rows: {len(sales):,}")
print(f"Gross units: {sales['gross_qty'].sum():,.0f} | returned units: {sales['return_qty'].sum():,.0f}")

sales_cols = ["transaction_date", "store", "product_no", "product_description",
              "product_division", "product_category", "product_subcategory",
              "product_segment", "supplier", "sales_type", "is_return",
              "qty_sold", "gross_qty", "return_qty", "sales_amount", "cogs",
              "number_of_transactions"]
sales = sales[sales_cols].sort_values(["transaction_date", "store", "product_no"])
sales.to_csv(OUT / "sales_clean.csv", index=False)
print(f"Saved: {OUT / 'sales_clean.csv'}")

# ------------------------------------------------------------------ C
section("C. CLEAN INVENTORY")
cutoff = sales["transaction_date"].max()
open_mask = inv["end_date"].astype(str).str.startswith("9999")
inv["end_is_open"] = open_mask.astype(int)
inv["start_date"] = pd.to_datetime(inv["start_date"])
inv["end_date"] = pd.to_datetime(inv["end_date"].where(~open_mask), errors="coerce")
inv.loc[open_mask, "end_date"] = cutoff
inv["days_valid"] = (inv["end_date"] - inv["start_date"]).dt.days + 1
print(f"Open-ended records (9999-12-31) set to {cutoff.date()}: {open_mask.sum():,}")
print(f"Records with end before start: {(inv['end_date'] < inv['start_date']).sum():,}")

inv = inv.sort_values(["product_no", "store", "start_date"]).reset_index(drop=True)
pair = ["product_no", "store"]

def fill_price(series_name, source_col):
    inv[series_name] = inv[source_col].where(inv[source_col] > 0)
    blank_before = inv[series_name].isna().sum()
    inv[series_name] = inv.groupby(pair)[series_name].ffill()
    inv[series_name] = inv.groupby(pair)[series_name].bfill()
    inv[series_name] = inv[series_name].fillna(
        inv.groupby("product_no")[series_name].transform("median"))
    print(f"{series_name}: blank/invalid before = {blank_before:,} | "
          f"still missing after filling = {inv[series_name].isna().sum():,}")

fill_price("unit_price", "stock_unit_selling_price")
fill_price("unit_cost", "stock_unit_cost_price")

inv["out_of_stock"] = (inv["qty_on_hand"] <= 0).astype(int)
inv["negative_stock"] = (inv["qty_on_hand"] < 0).astype(int)
print(f"\nout_of_stock records: {inv['out_of_stock'].sum():,} | "
      f"negative_stock records: {inv['negative_stock'].sum():,}")

drop_cols = ["stocks_selling_amount", "cost_of_stocks", "stock_unit_selling_price",
             "stock_unit_cost_price", "sales_channel", "store_type"]
inv = inv.drop(columns=drop_cols)
print(f"Dropped columns: {drop_cols}")

prev_end = inv.groupby(pair)["end_date"].shift()
print(f"Overlapping records after cleaning: {(inv['start_date'] <= prev_end).sum():,}")
print(f"Clean inventory rows: {len(inv):,}")

inv.to_csv(OUT / "inventory_intervals_clean.csv", index=False)
print(f"Saved: {OUT / 'inventory_intervals_clean.csv'}")

section("D. FINAL COLUMNS")
print("sales_clean:", list(sales.columns))
print("inventory_intervals_clean:", list(inv.columns))
