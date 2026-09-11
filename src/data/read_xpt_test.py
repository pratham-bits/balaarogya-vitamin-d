import pandas as pd

# Path to the NHANES Vitamin-D laboratory dataset
file_path = "data/raw/VID_J.XPT"

# Read the SAS Transport (.XPT) file
df = pd.read_sas(
    file_path,
    format="xport"
)

print("\n========== DATASET LOADED ==========")
print(f"Shape: {df.shape}")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATA TYPES ==========")
print(df.dtypes)