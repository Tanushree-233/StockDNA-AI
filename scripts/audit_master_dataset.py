import pandas as pd

FILE = "data/final/master_dataset.csv"

df = pd.read_csv(FILE, parse_dates=["Date"])

print("=" * 60)
print("MASTER DATASET AUDIT")
print("=" * 60)

# 1. Shape
print("\n1. DATASET SHAPE")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

# 2. Columns
print("\n2. COLUMNS")
for col in df.columns:
    print(col)

# 3. Target distribution
print("\n3. TARGET DISTRIBUTION")

if "Target" in df.columns:
    print(df["Target"].value_counts())
    print("\nPercentage:")
    print(df["Target"].value_counts(normalize=True) * 100)
else:
    print("Target column NOT FOUND")

# 4. Missing values
print("\n4. MISSING VALUES")

missing = df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if len(missing) == 0:
    print("No missing values found.")
else:
    print(missing)

# 5. Duplicate rows
print("\n5. DUPLICATES")
print("Duplicate rows:", df.duplicated().sum())

# 6. Date information
print("\n6. DATE INFORMATION")

date_columns = [
    col for col in df.columns
    if "date" in col.lower()
]

print("Possible date columns:", date_columns)

for col in date_columns:
    print("\n", col)
    print("Min:", df[col].min())
    print("Max:", df[col].max())

# 7. Data types
print("\n7. DATA TYPES")
print(df.dtypes)

# 8. Basic statistics
print("\n8. NUMERICAL SUMMARY")
print(df.describe().T)

print("\n" + "=" * 60)
print("AUDIT COMPLETE")
print("=" * 60)