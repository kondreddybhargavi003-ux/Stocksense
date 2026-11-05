from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

def section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

csv_files = sorted(RAW_DIR.glob("*.csv"))

for file in csv_files:
    df = pd.read_csv(file, low_memory=False)
    section(f"FILE: {file.name}  |  {df.shape[0]:,} rows x {df.shape[1]} cols")

    # 1. Fully duplicated rows
    dup_count = df.duplicated().sum()
    print(f"\n[1] Fully duplicated rows: {dup_count:,}")

    # 2. Completely empty columns
    empty_cols = [c for c in df.columns if df[c].isna().all()]
    print(f"[2] Completely empty columns: {empty_cols if empty_cols else 'None'}")

    # 3. Missing values with percentage
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print("\n[3] Missing values:")
    if len(missing):
        out = pd.DataFrame({
            "missing": missing,
            "percent": (missing / len(df) * 100).round(2)
        })
        print(out.to_string())
    else:
        print("None")

    # 4. Columns that look like dates
    date_like = [c for c in df.columns
                 if any(k in c.lower() for k in ["date", "_on", "_at", "time"])]
    print(f"\n[4] Date-like columns: {date_like if date_like else 'None'}")
    for c in date_like:
        parsed = pd.to_datetime(df[c], errors="coerce")
        bad = parsed.isna().sum() - df[c].isna().sum()
        print(f"    {c}: min={parsed.min()}  max={parsed.max()}  "
              f"unparseable={bad}")
        year_9999 = (df[c].astype(str).str.startswith("9999")).sum()
        if year_9999:
            print(f"    {c}: rows with year 9999 = {year_9999:,}")

    # 5. Numeric sanity
    num_cols = df.select_dtypes(include="number").columns
    print("\n[5] Numeric columns summary:")
    if len(num_cols):
        summary = df[num_cols].describe().T[["min", "mean", "max"]]
        summary["negatives"] = (df[num_cols] < 0).sum()
        summary["zeros"] = (df[num_cols] == 0).sum()
        print(summary.round(2).to_string())
    else:
        print("None")

    # 6. Columns with very few unique values (categorical candidates)
    print("\n[6] Low-cardinality columns (<= 12 unique values):")
    found = False
    for c in df.columns:
        n = df[c].nunique(dropna=True)
        if 0 < n <= 12:
            vals = df[c].dropna().unique()[:12].tolist()
            print(f"    {c} ({n}): {vals}")
            found = True
    if not found:
        print("None")
