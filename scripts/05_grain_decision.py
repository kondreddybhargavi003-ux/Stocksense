from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

S_PROD, S_STORE, S_DATE, S_QTY = "Product No", "Store", "Transaction Date", "Qty Sold"
I_PROD, I_STORE, I_START, I_END, I_QTY = "Product No", "Store", "Start Date", "End Date", "Qty on hand"
I_PRICE, I_COST = "Stock Unit Selling Price", "Cost of Stocks"

sales = pd.read_csv(RAW_DIR / "retail_sales_ml_apl.csv", low_memory=False)
inv = pd.read_csv(RAW_DIR / "retail_inventory_ml_apl.csv", low_memory=False)
sales[S_DATE] = pd.to_datetime(sales[S_DATE], errors="coerce")

# ------------------------------------------------------------------ A
section("A. COLUMNS OF THE TWO RETAIL FILES")
print("SALES:")
print(sales.dtypes.to_string())
print("\nINVENTORY:")
print(inv.dtypes.to_string())
print("\nSales sample:")
print(sales.head(2).T.to_string())

# ------------------------------------------------------------------ B
section("B. SALES: MISSING DAYS AND RETURNS")
all_days = pd.date_range(sales[S_DATE].min(), sales[S_DATE].max())
missing_days = all_days.difference(sales[S_DATE].unique())
print(f"Days with no sales at all ({len(missing_days)}):")
print([d.strftime("%Y-%m-%d (%a)") for d in missing_days])

q = sales[S_QTY]
print(f"\nRows with negative Qty Sold (returns): {(q < 0).sum():,} ({(q < 0).mean() * 100:.2f}%)")
print(f"Rows with zero Qty Sold: {(q == 0).sum():,}")
print(f"Gross units sold: {q[q > 0].sum():,} | units returned: {-q[q < 0].sum():,}")

for c in sales.columns:
    if "return" in c.lower():
        n = sales[c].nunique(dropna=True)
        if 0 < n <= 12:
            print(f"\nValue counts for '{c}':")
            print(sales[c].value_counts(dropna=False).to_string())

# ------------------------------------------------------------------ C
section("C. CANDIDATE MODELING GRAINS (weekly, partial first/last weeks removed)")
sales["week"] = sales[S_DATE].dt.to_period("W")
weeks = sorted(sales["week"].unique())
inner = weeks[1:-1]
df = sales[sales["week"].isin(inner)]
n_weeks = len(inner)
print(f"Full weeks used: {n_weeks}")

def evaluate(name, keys):
    g = df.groupby(keys + ["week"])[S_QTY].sum().reset_index()
    per = g.groupby(keys)[S_QTY].agg(["count", "sum"])
    n_series = len(per)
    nonzero = (g[S_QTY] != 0).sum()
    density = nonzero / (n_series * n_weeks) * 100
    print(f"\n{name}")
    print(f"  series: {n_series:,}")
    print(f"  weekly density (share of weeks with demand): {density:.1f}%")
    print(f"  median selling weeks per series: {per['count'].median():.0f} of {n_weeks}")
    print(f"  series with 20+ selling weeks: {(per['count'] >= 20).mean() * 100:.1f}%")
    print(f"  median total units per series: {per['sum'].median():.0f}")
    wk = g.groupby("week")[S_QTY].sum()
    print(f"  total units per week: mean={wk.mean():.0f} min={wk.min():.0f} max={wk.max():.0f}")

evaluate("Product (all stores)", [S_PROD])
evaluate("Store (all products)", [S_STORE])
evaluate("Product-Store", [S_PROD, S_STORE])

hints = ["categor", "dept", "department", "class", "group", "brand", "division", "section", "type"]
cand = [c for c in sales.columns
        if c not in (S_PROD, S_STORE, S_DATE, "week")
        and sales[c].dtype == object
        and 2 <= sales[c].nunique() <= 300
        and any(h in c.lower() for h in hints)]
print(f"\nCategory-like columns detected: {cand if cand else 'None'}")
for c in cand:
    evaluate(f"{c} (all stores)", [c])
    evaluate(f"{c} x Store", [c, S_STORE])

# ------------------------------------------------------------------ D
section("D. INVENTORY: DAY-WEIGHTED STOCKOUT RATE AND ANOMALIES")
cutoff = sales[S_DATE].max()
inv["_start"] = pd.to_datetime(inv[I_START], errors="coerce")
open_mask = inv[I_END].astype(str).str.startswith("9999")
inv["_end"] = pd.to_datetime(inv[I_END].where(~open_mask), errors="coerce")
inv.loc[open_mask, "_end"] = cutoff
inv["_days"] = (inv["_end"] - inv["_start"]).dt.days + 1

total_days = inv["_days"].sum()
out_mask = inv[I_QTY] <= 0
print(f"Record-weighted share with Qty on hand <= 0: {out_mask.mean() * 100:.2f}%")
print(f"Day-weighted share with Qty on hand <= 0:    {inv.loc[out_mask, '_days'].sum() / total_days * 100:.2f}%")
print("\nHow long do zero/negative-stock records last (days)?")
print(inv.loc[out_mask, "_days"].describe().round(2).to_string())

pairs_total = inv.groupby([I_PROD, I_STORE]).ngroups
pairs_out = inv[out_mask].groupby([I_PROD, I_STORE]).ngroups
print(f"\nPairs that hit zero/negative stock at least once: {pairs_out:,} of {pairs_total:,} "
      f"({pairs_out / pairs_total * 100:.1f}%)")

inv = inv.sort_values([I_PROD, I_STORE, "_start"])
inv["_next_qty"] = inv.groupby([I_PROD, I_STORE])[I_QTY].shift(-1)
has_next = inv[(inv[I_QTY] > 0) & inv["_next_qty"].notna()]
print(f"\nOf records with stock > 0 that have a next record: "
      f"{(has_next['_next_qty'] <= 0).mean() * 100:.2f}% are followed by a zero/negative record")

print("\nPrice vs quantity:")
print(f"  price == 0 and qty == 0: {((inv[I_PRICE] == 0) & (inv[I_QTY] == 0)).sum():,}")
print(f"  price == 0 and qty != 0: {((inv[I_PRICE] == 0) & (inv[I_QTY] != 0)).sum():,}")
print(f"  price != 0 and qty == 0: {((inv[I_PRICE] != 0) & (inv[I_QTY] == 0)).sum():,}")

print("\nCost anomalies:")
print(f"  Cost of Stocks < 0 with Qty on hand > 0: {((inv[I_COST] < 0) & (inv[I_QTY] > 0)).sum():,}")
print(f"  Cost of Stocks < 0 with Qty on hand <= 0: {((inv[I_COST] < 0) & (inv[I_QTY] <= 0)).sum():,}")
print(f"  Cost of Stocks == 0 with Qty on hand > 0: {((inv[I_COST] == 0) & (inv[I_QTY] > 0)).sum():,}")
