from pathlib import Path
import pandas as pd

RAW_DIR = Path("data/raw")

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)

csv_files = sorted(RAW_DIR.glob("*.csv"))
print(f"Found {len(csv_files)} CSV files\n")

for file in csv_files:
    df = pd.read_csv(file, low_memory=False)

    print("=" * 80)
    print(f"FILE: {file.name}")
    print("=" * 80)
    print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns\n")

    print("Columns and data types:")
    print(df.dtypes.to_string(), "\n")

    print("Missing values (only columns with missing data):")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    print(missing.to_string() if len(missing) else "None", "\n")

    print("First 3 rows:")
    print(df.head(3), "\n")
