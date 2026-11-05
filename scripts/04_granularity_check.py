from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def find_file(keyword):
    for f in sorted(RAW_DIR.glob("*.csv")):
        if keyword in f.stem.lower():
            return f
    return None

def find_col(df, keywords, exclude=()):
    for c in df.columns:
        low = c.lower()
        if all(k in low for k in keywords) and not any(x in low for x in exclude):
            return c
    return None

sales_file, inv_file = find_file("sales"), find_file("inventory")
sales = pd.read_csv(sales_file, low_memory=False)
inv = pd.read_csv(inv_file, low_memory=False)

# ---------------------------------------------------------------- SALES
section(f"A. SALES GRANULARITY  ({sales_file.name})")
s_prod = find_col(sales, ["product", "id"]) or find_col(sales, ["product"])
s_store = find_col(sales, ["store", "id"]) or find_col(sales, ["store"])
s_date = find_col(sales, ["date"])
s_qty = find_col(sales, ["quantity"]) or find_col(sales, ["qty"])
s_txn = (find_col(sales, ["transaction"]) or find_col(sales, ["order"])
         or find_col(sales, ["invoice"]))
print(f"Detected -> product: {s_prod} | store: {s_store} | date: {s_date} "
      f"| quantity: {s_qty} | transaction id: {s_txn}")

sales[s_date] = pd.to_datetime(sales[s_date], errors="coerce")
print(f"Rows: {len(sales):,}")
print(f"Date range: {sales[s_date].min()}  to  {sales[s_date].max()}")
print(f"Unparseable dates: {sales[s_date].isna().sum():,}")

if s_txn:
    print(f"Unique {s_txn}: {sales[s_txn].nunique():,} "
          f"(rows per transaction: {len(sales) / sales[s_txn].nunique():.2f})")

key = [s_date, s_prod, s_store]
per_key = sales.groupby(key).size()
print(f"\nUnique (date, product, store) combinations: {len(per_key):,}")
print("Rows per (date, product, store):")
print(per_key.describe().round(2).to_string())
print(f"Keys with more than 1 row: {(per_key > 1).sum():,}")
print("-> If this is large, several rows exist per key and must be SUMMED "
      "before modeling.")

if s_qty:
    print(f"\nQuantity summary:\n{sales[s_qty].describe().round(2).to_string()}")

# Sparsity
section("B. SPARSITY: HOW OFTEN DOES EACH PRODUCT-STORE PAIR SELL?")
n_days = (sales[s_date].max() - sales[s_date].min()).days + 1
pair_days = sales.groupby([s_prod, s_store])[s_date].nunique()
print(f"Days in the period: {n_days}")
print(f"Product-store pairs with at least one sale: {len(pair_days):,}")
print("\nSelling days per pair:")
print(pair_days.describe().round(2).to_string())
print("\nPairs by number of selling days:")
bins = [0, 4, 14, 29, 59, 119, n_days]
labels = ["1-4", "5-14", "15-29", "30-59", "60-119", "120+"]
dist = pd.cut(pair_days, bins=bins, labels=labels).value_counts().reindex(labels)
print(pd.DataFrame({"pairs": dist, "percent": (dist / len(pair_days) * 100).round(1)})
      .to_string())

density_day = len(per_key) / (len(pair_days) * n_days) * 100
print(f"\nDaily density: {density_day:.2f}% of all possible "
      f"(pair, day) cells have a sale")

weekly = sales.assign(week=sales[s_date].dt.to_period("W")) \
              .groupby([s_prod, s_store, "week"]).size()
n_weeks = sales[s_date].dt.to_period("W").nunique()
density_week = len(weekly) / (len(pair_days) * n_weeks) * 100
print(f"Weekly density: {density_week:.2f}% of all possible (pair, week) cells have a sale")
print("-> The higher the density, the more stable that grain is for forecasting.")

# Product-level (all stores) for comparison
prod_days = sales.groupby(s_prod)[s_date].nunique()
print(f"\nProduct-level (all stores combined): median selling days per product = "
      f"{prod_days.median():.0f} of {n_days}")

# ------------------------------------------------------------ INVENTORY
section(f"C. INVENTORY GRANULARITY  ({inv_file.name})")
i_prod = find_col(inv, ["product", "id"]) or find_col(inv, ["product"])
i_store = find_col(inv, ["store", "id"]) or find_col(inv, ["store"])
i_start = find_col(inv, ["start"]) or find_col(inv, ["from"]) or find_col(inv, ["effective"])
i_end = find_col(inv, ["end"]) or find_col(inv, ["to_date"]) or find_col(inv, ["expir"])
date_like = [c for c in inv.columns if any(k in c.lower() for k in ["date", "_on", "_at"])]
print(f"Detected -> product: {i_prod} | store: {i_store} | start: {i_start} | end: {i_end}")
print(f"All date-like columns: {date_like}")
print(f"Rows: {len(inv):,}")

per_pair = inv.groupby([i_prod, i_store]).size()
print(f"\nUnique product-store pairs in inventory: {len(per_pair):,}")
print("Inventory rows per product-store pair:")
print(per_pair.describe().round(2).to_string())

if i_start and i_end:
    inv["_start"] = pd.to_datetime(inv[i_start], errors="coerce")
    inv["_end_raw"] = inv[i_end].astype(str)
    open_mask = inv["_end_raw"].str.startswith("9999")
    inv["_end"] = pd.to_datetime(inv[i_end].where(~open_mask), errors="coerce")
    cutoff = sales[s_date].max()
    inv.loc[open_mask, "_end"] = cutoff
    print(f"\nRecords with open end date (9999): {open_mask.sum():,} "
          f"-> replaced by {cutoff.date()} for these checks")
    print(f"Start range: {inv['_start'].min()}  to  {inv['_start'].max()}")

    inv["_days"] = (inv["_end"] - inv["_start"]).dt.days + 1
    print("\nDays each record is valid:")
    print(inv["_days"].describe().round(2).to_string())
    print(f"Records with end before start: {(inv['_end'] < inv['_start']).sum():,}")

    inv = inv.sort_values([i_prod, i_store, "_start"])
    prev_end = inv.groupby([i_prod, i_store])["_end"].shift()
    overlap = (inv["_start"] <= prev_end)
    print(f"Records overlapping the previous record of the same pair: {overlap.sum():,}")
    print("-> Interval style (open end dates, no overlaps) means each row is a "
          "version of stock info that holds until the next one. Daily stock "
          "can be rebuilt by expanding the intervals.")
elif date_like:
    d = date_like[0]
    inv["_d"] = pd.to_datetime(inv[d], errors="coerce")
    print(f"\nSingle date column '{d}' -> treating as snapshots")
    print(f"Distinct snapshot dates: {inv['_d'].nunique():,}")
    print(f"Range: {inv['_d'].min()} to {inv['_d'].max()}")
    dup = inv.duplicated([i_prod, i_store, "_d"]).sum()
    print(f"Duplicate (product, store, date) rows: {dup:,}")
    gaps = inv.sort_values("_d").groupby([i_prod, i_store])["_d"].diff().dt.days
    print("Days between consecutive snapshots of the same pair:")
    print(gaps.describe().round(2).to_string())

# Stock columns (needed for stockout labels)
section("D. STOCK QUANTITY COLUMNS (needed for stockout labels)")
stock_cols = [c for c in inv.select_dtypes(include="number").columns
              if any(k in c.lower() for k in ["qty", "quantity", "stock", "on_hand", "onhand"])]
print(f"Stock-like numeric columns: {stock_cols}")
for c in stock_cols:
    print(f"\n{c}:")
    print(inv[c].describe().round(2).to_string())
    print(f"  zero: {(inv[c] == 0).sum():,} ({(inv[c] == 0).mean() * 100:.2f}%)")
    print(f"  negative: {(inv[c] < 0).sum():,} ({(inv[c] < 0).mean() * 100:.2f}%)")
print("-> The share of zero or negative stock shows whether a stockout label is "
      "realistic or too rare.")

# ----------------------------------------------------- SALES vs INVENTORY
section("E. SALES vs INVENTORY PAIR COVERAGE")
s_pairs = set(zip(sales[s_prod], sales[s_store]))
i_pairs = set(zip(inv[i_prod], inv[i_store]))
both = len(s_pairs & i_pairs)
print(f"Pairs in sales: {len(s_pairs):,} | in inventory: {len(i_pairs):,} | in both: {both:,}")
print(f"Share of sales pairs that have inventory data: {both / len(s_pairs) * 100:.1f}%")
print("-> Only pairs in BOTH files can be used for stockout modeling.")
