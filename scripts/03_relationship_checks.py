from pathlib import Path
from itertools import combinations
import pandas as pd

RAW_DIR = Path("data/raw")
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

data = {f.stem: pd.read_csv(f, low_memory=False) for f in sorted(RAW_DIR.glob("*.csv"))}

# A. Column names
section("A. COLUMN NAMES PER FILE")
for name, df in data.items():
    print(f"\n{name} ({len(df):,} rows):")
    print("  ", list(df.columns))

# B. Shared columns between every pair of files
section("B. SHARED COLUMNS BETWEEN FILE PAIRS (value overlap)")
for (n1, d1), (n2, d2) in combinations(data.items(), 2):
    for c in [c for c in d1.columns if c in d2.columns]:
        s1 = set(d1[c].dropna().unique())
        s2 = set(d2[c].dropna().unique())
        if not s1 or not s2 or max(len(s1), len(s2)) <= 12:
            continue
        common = len(s1 & s2)
        print(f"{n1} <-> {n2} | '{c}': {len(s1):,} vs {len(s2):,} unique, {common:,} in common")

# C. Retail product and store coverage
section("C. RETAIL FILES: PRODUCT / STORE / DATE DETECTION")

def find_file(keyword):
    for name in data:
        if keyword in name.lower():
            return name
    return None

def find_col(df, keywords):
    for c in df.columns:
        low = c.lower()
        if all(k in low for k in keywords):
            return c
    return None

sales_name = find_file("sales")
inv_name = find_file("inventory")
print(f"Sales file detected: {sales_name}")
print(f"Inventory file detected: {inv_name}")

if sales_name and inv_name:
    s, i = data[sales_name], data[inv_name]
    s_prod = find_col(s, ["product", "id"]) or find_col(s, ["product"])
    s_store = find_col(s, ["store", "id"]) or find_col(s, ["store"])
    i_prod = find_col(i, ["product", "id"]) or find_col(i, ["product"])
    i_store = find_col(i, ["store", "id"]) or find_col(i, ["store"])
    print(f"Sales product col: {s_prod} | store col: {s_store}")
    print(f"Inventory product col: {i_prod} | store col: {i_store}")

    if all([s_prod, s_store, i_prod, i_store]):
        sp, ip = set(s[s_prod]), set(i[i_prod])
        ss, is_ = set(s[s_store]), set(i[i_store])
        print(f"\nProducts: sales={len(sp):,} inventory={len(ip):,} common={len(sp & ip):,}")
        print(f"Stores:   sales={len(ss):,} inventory={len(is_):,} common={len(ss & is_):,}")

        s_pairs = set(zip(s[s_prod], s[s_store]))
        i_pairs = set(zip(i[i_prod], i[i_store]))
        both = len(s_pairs & i_pairs)
        print(f"\nProduct-store pairs: sales={len(s_pairs):,} inventory={len(i_pairs):,}")
        print(f"Pairs in both: {both:,}")
        print(f"Pairs only in sales: {len(s_pairs - i_pairs):,}")
        print(f"Pairs only in inventory: {len(i_pairs - s_pairs):,}")
        print(f"Share of sales pairs found in inventory: {both / len(s_pairs) * 100:.1f}%")

    for label, df in [("Sales", s), ("Inventory", i)]:
        print(f"\n{label} date columns:")
        for c in df.columns:
            if any(k in c.lower() for k in ["date", "_on", "_at"]):
                parsed = pd.to_datetime(df[c], errors="coerce")
                valid = parsed[parsed.dt.year < 9999]
                print(f"  {c}: min={valid.min()}  max={valid.max()}")

# D. Procurement SKU vs retail product mapping
section("D. PROCUREMENT SKU vs RETAIL PRODUCT IDS")
if sales_name:
    retail_products = set(data[sales_name][s_prod].astype(str)) if s_prod else set()
    for name, df in data.items():
        for c in df.columns:
            if "sku" in c.lower():
                vals = set(df[c].dropna().astype(str))
                print(f"{name}.{c}: {len(vals)} unique | sample: {list(vals)[:3]}")
                print(f"   matches in retail product ids: {len(vals & retail_products)}")
    print(f"Retail product id sample: {list(retail_products)[:3]}")

# E. Missing paid_on investigation
section("E. SUPPLIER INVOICES: paid_on")
for name, df in data.items():
    if "paid_on" in df.columns:
        print(f"{name}: {df['paid_on'].isna().sum()} missing of {len(df)}")
        status_cols = [c for c in df.columns if "status" in c.lower()]
        for sc in status_cols:
            print(f"\nMissing paid_on by {sc}:")
            print(df.groupby(sc)["paid_on"].apply(lambda x: x.isna().sum()).to_string())