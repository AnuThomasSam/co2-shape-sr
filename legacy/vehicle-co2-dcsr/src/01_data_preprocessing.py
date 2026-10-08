# %%
import pandas as pd
import numpy as np

# %%
# Load the dataset
df = pd.read_csv("../data/raw/CO2 Emissions_Canada.csv")

print("Original dataset shape:", df.shape)

# %%
# Create working copy
data = df.copy()

# %%
# Rename the Columns
data.rename(columns={ 'Make': 'make',
                    'Model': 'model',
                    'Vehicle Class': 'vehicle_class',
                    'Engine Size(L)': 'engine_size',
                    'Cylinders': 'cylinders',
                    'Transmission': 'transmission',
                    'Fuel Type': 'fuel_type',
                    'Fuel Consumption City (L/100 km)': 'fuel_cons_city',
                    'Fuel Consumption Hwy (L/100 km)': 'fuel_cons_hwy',
                    'Fuel Consumption Comb (L/100 km)': 'fuel_cons_comb',
                    'Fuel Consumption Comb (mpg)': 'fuel_cons_comb_mpg',
                    'CO2 Emissions(g/km)': 'co2'
                    }, inplace=True)

# %%
# Validate numeric types for key columns
for col in ['engine_size', 'cylinders', 'co2']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    print(f"{col} dtype: {data[col].dtype}, nulls: {data[col].isnull().sum()}")

# %%
# Check missing values
print(data.isnull().sum())

# %%
# Define target
target = "co2"

# %%
# Identify leakage variables
leakage_columns = [
    "fuel_cons_city",
    "fuel_cons_hwy",
    "fuel_cons_comb",
    "fuel_cons_comb_mpg"
]

# Checking these are existing in the dataset
for column in leakage_columns:
    print(column, "→", column in data.columns)

# %%
# Remove leakage variables
data_clean = data.drop(columns=leakage_columns).copy()

print("Columns after leakage removal:")
print(data_clean.columns.tolist())

# %%
# Extract transmission gear count
data_clean["transmission_gears"] = (
    data_clean["transmission"]
    .str.extract(r"(\d+)", expand=False)
    .astype(float)
)

# %%
# Create transmission type
data_clean["transmission_type"] = (
    data_clean["transmission"]
    .str.extract(r"([A-Z]+)", expand=False)
)

# %%
# Check transmission extraction
print(
    data_clean[
        ["transmission", "transmission_type", "transmission_gears"]
    ].head(20)
)

print(
    "Missing transmission gears:",
    data_clean["transmission_gears"].isnull().sum()
)

# %%
# Checking whether extraction worked for everything
data_clean["transmission_gears"].isnull().sum()

# %%
# Handle AV transmissions with no fixed gear count
# Median imputation is used to enable K-Means clustering
median_gears = data_clean["transmission_gears"].median()
data_clean["transmission_gears"] = data_clean["transmission_gears"].fillna(median_gears)

print(f"AV transmissions imputed with median gear count: {median_gears}")
print("Missing transmission_gears after imputation:", data_clean["transmission_gears"].isnull().sum())

# %%
# Engineer interaction features
# f1 = engine_size / cylinders  (displacement per cylinder)
# f2 = engine_size * cylinders  (proxy for combustion chamber volume)
# f3 = engine_size^2            (nonlinear engine size effect)
# f4 = cylinders^2              (nonlinear cylinder effect)

data_clean["engine_per_cylinder"]   = data_clean["engine_size"] / data_clean["cylinders"]       # f1
data_clean["engine_x_cylinders"]    = data_clean["engine_size"] * data_clean["cylinders"]       # f2
data_clean["engine_size_sq"]        = data_clean["engine_size"] ** 2                            # f3
data_clean["cylinders_sq"]          = data_clean["cylinders"] ** 2                              # f4

print("Engineered features added:")
print(data_clean[["engine_per_cylinder", "engine_x_cylinders",
                   "engine_size_sq", "cylinders_sq"]].describe())

# %%
# Remove duplicates — after leakage removal and feature engineering
# This correctly removes rows that are identical in the leakage-free feature space
rows_before_dedup = len(data_clean)

data_clean = data_clean.drop_duplicates().copy()

rows_after_dedup = len(data_clean)

print("Rows before deduplication:", rows_before_dedup)
print("Rows after deduplication:", rows_after_dedup)
print("Duplicates removed:", rows_before_dedup - rows_after_dedup)
print("Total rows removed from original dataset:", len(df) - rows_after_dedup)

# %%
# Check duplicates AFTER feature engineering
duplicates_after_features = data_clean.duplicated().sum()

print(
    "Remaining duplicates after feature engineering and deduplication:",
    duplicates_after_features
)

# %%
# Define engineering features used for temporal clustering (KMeans)
engineering_features = ["engine_size", "cylinders", "transmission_gears"]

print(data_clean[engineering_features].describe())
print("Any nulls in clustering features:", data_clean[engineering_features].isnull().sum().sum())

# %%
# Final shape and column check
print("Final dataset shape:", data_clean.shape)
print("Columns:", data_clean.columns.tolist())
print("\nMissing values in final dataset:")
print(data_clean.isnull().sum()[data_clean.isnull().sum() > 0])

# %%
# Save processed data
data_clean.to_csv(
    "../data/processed/vehicle_co2_clean.csv",
    index=False
)
# %%
