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

sales = pd.read_csv(RAW_DIR / "retail_sales_ml_apl.csv", low_memory=False)
inv = pd.read_csv(RAW_DIR / "retail_inventory_ml_apl.csv", low_memory=False)
sales[S_DATE] = pd.to_datetime(sales[S_DATE], errors="coerce")

# ------------------------------------------------------------------ A
section("A. PRODUCT NO vs ID INSIDE PRODUCT DESCRIPTION")
sales["_desc_id"] = sales["Product Description"].str.extract(r"^(PROD-\d+)")[0]
print(f"Rows where description starts with a PROD id: {sales['_desc_id'].notna().sum():,} of {len(sales):,}")
same = (sales["_desc_id"] == sales[S_PROD])
print(f"Rows where Product No == description id: {same.sum():,} ({same.mean() * 100:.1f}%)")
print(f"Unique Product No: {sales[S_PROD].nunique():,} | unique description ids: {sales['_desc_id'].nunique():,}")
print(f"Unique Product Description values: {sales['Product Description'].nunique():,}")
per_prod = sales.groupby(S_PROD)["Product Description"].nunique()
print(f"Product No with more than 1 distinct description: {(per_prod > 1).sum():,}")
per_desc = sales.groupby("Product Description")[S_PROD].nunique()
print(f"Descriptions linked to more than 1 Product No: {(per_desc > 1).sum():,}")
print("\nFirst 8 mismatching examples:")
print(sales.loc[~same, [S_PROD, "_desc_id", "Product Category"]].drop_duplicates().head(8).to_string())

# ------------------------------------------------------------------ B
section("B. HIERARCHY SIZE AND CONSISTENCY")
levels = ["Product Division", "Product Category", "Product Subcategory", "Product Segment"]
for c in levels + ["Supplier", "Sales Channel", "Sales Type"]:
    print(f"{c}: {sales[c].nunique():,} unique values")
print("\nDoes each Product No belong to exactly one value at each level?")
for c in levels:
    multi = (sales.groupby(S_PROD)[c].nunique() > 1).sum()
    print(f"  {c}: Product No with more than 1 value = {multi:,}")

# ------------------------------------------------------------------ C
section("C. CANDIDATE GRAINS ON GROSS DEMAND (positive quantities only, 46 full weeks)")
sales["week"] = sales[S_DATE].dt.to_period("W")
weeks = sorted(sales["week"].unique())
inner = weeks[1:-1]
df = sales[sales["week"].isin(inner) & (sales[S_QTY] > 0)]
n_weeks = len(inner)

def evaluate(name, keys):
    g = df.groupby(keys + ["week"])[S_QTY].sum().reset_index()
    per = g.groupby(keys)[S_QTY].agg(["count", "sum"])
    density = len(g) / (len(per) * n_weeks) * 100
    print(f"\n{name}")
    print(f"  series: {len(per):,} | weekly density: {density:.1f}%")
    print(f"  median selling weeks: {per['count'].median():.0f} of {n_weeks} | "
          f"series with 20+ selling weeks: {(per['count'] >= 20).mean() * 100:.1f}%")
    print(f"  median total units per series: {per['sum'].median():.0f}")

evaluate("Product No", [S_PROD])
for c in levels:
    evaluate(c, [c])
evaluate("Product Category x Store", ["Product Category", S_STORE])
evaluate("Product Subcategory x Store", ["Product Subcategory", S_STORE])

# ------------------------------------------------------------------ D
section("D. SALES TYPE, CHANNEL AND WEEKLY SPIKES")
print("Sales Type (rows and units):")
print(sales.groupby("Sales Type")[S_QTY].agg(["count", "sum"]).to_string())
print("\nSales Channel (rows and units):")
print(sales.groupby("Sales Channel")[S_QTY].agg(["count", "sum"]).to_string())

wk = df.groupby("week")[S_QTY].sum()
print("\nTop 6 weeks by units:")
print(wk.sort_values(ascending=False).head(6).to_string())
print("\nBottom 6 weeks by units:")
print(wk.sort_values().head(6).to_string())

top_weeks = wk.sort_values(ascending=False).head(6).index
mix = df[df["week"].isin(top_weeks)].groupby("Sales Type")[S_QTY].sum()
print("\nSales Type mix inside the top 6 weeks:")
print(mix.to_string())

# ------------------------------------------------------------------ E
section("E. INVENTORY: STOCK STATUS AND STORE TYPE")
print("Stock Status values:")
print(inv["Stock Status"].value_counts(dropna=False).to_string())
print("\nStock Status vs (Qty on hand <= 0):")
print(pd.crosstab(inv["Stock Status"], inv["Qty on hand"] <= 0).to_string())
print("\nStore Type values:")
print(inv["Store Type"].value_counts(dropna=False).to_string())
