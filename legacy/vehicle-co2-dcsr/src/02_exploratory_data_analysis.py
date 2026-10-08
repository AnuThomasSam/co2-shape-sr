# %%
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# %%
# Global style — consistent across all figures
plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

PRIMARY   = "#2C6E9E"   # blue for histograms / scatter
SECONDARY = "#E07B39"   # orange for accent plots

# %%
# Load the cleaned dataset
data_clean = pd.read_csv("../data/processed/vehicle_co2_clean.csv")

print("Number of rows:", len(data_clean))
print("Number of columns:", len(data_clean.columns))
data_clean.head()

# %%
# Distribution of Engine Size

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(data_clean["engine_size"], bins=30, color=PRIMARY, edgecolor="white")
ax.set_xlabel("Engine Size (L)")
ax.set_ylabel("Number of Vehicles")
ax.set_title("Distribution of Engine Size")
plt.tight_layout()

plt.savefig("../results/figures/engine_size_distribution.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Distribution of Cylinders

fig, ax = plt.subplots(figsize=(8, 5))

data_clean["cylinders"].value_counts().sort_index().plot(
    kind="bar", ax=ax, color=PRIMARY, edgecolor="white"
)
ax.set_xlabel("Number of Cylinders")
ax.set_ylabel("Number of Vehicles")
ax.set_title("Distribution of Engine Cylinders")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()

plt.savefig("../results/figures/cylinders_distribution.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Distribution of CO₂ Emissions

fig, ax = plt.subplots(figsize=(8, 5))

ax.hist(data_clean["co2"], bins=30, color=PRIMARY, edgecolor="white")
ax.set_xlabel("CO₂ Emissions (g/km)")
ax.set_ylabel("Number of Vehicles")
ax.set_title("Distribution of CO₂ Emissions")
plt.tight_layout()

plt.savefig("../results/figures/co2_distribution.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Engine Size vs CO₂ Emissions

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(data_clean["engine_size"], data_clean["co2"],
           alpha=0.3, color=PRIMARY, s=15)
ax.set_xlabel("Engine Size (L)")
ax.set_ylabel("CO₂ Emissions (g/km)")
ax.set_title("Engine Size vs CO₂ Emissions")
plt.tight_layout()

plt.savefig("../results/figures/engine_size_vs_co2.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Distribution of Transmission Types

fig, ax = plt.subplots(figsize=(9, 5))

data_clean["transmission"].value_counts().plot(
    kind="bar",
    ax=ax,
    color=PRIMARY,
    edgecolor="white"
)

ax.set_xlabel("Transmission Type")
ax.set_ylabel("Number of Vehicles")
ax.set_title("Distribution of Transmission Types")
ax.tick_params(axis="x", rotation=45)

plt.tight_layout()

plt.savefig(
    "../results/figures/transmission_distribution.pdf",
    format="pdf",
    bbox_inches="tight"
)

plt.show()
plt.close()

# %%
# Cylinders vs CO₂ Emissions

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(data_clean["cylinders"], data_clean["co2"],
           alpha=0.3, color=PRIMARY, s=15)
ax.set_xlabel("Number of Cylinders")
ax.set_ylabel("CO₂ Emissions (g/km)")
ax.set_title("Cylinders vs CO₂ Emissions")
plt.tight_layout()

plt.savefig("../results/figures/cylinders_vs_co2.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Engine × Cylinders vs CO₂ — key engineered feature

fig, ax = plt.subplots(figsize=(8, 5))

ax.scatter(data_clean["engine_x_cylinders"], data_clean["co2"],
           alpha=0.3, color=SECONDARY, s=15)
ax.set_xlabel("Engine Size × Cylinders")
ax.set_ylabel("CO₂ Emissions (g/km)")
ax.set_title("Engine × Cylinders vs CO₂ Emissions\n"
    "(Engineered Interaction Feature)")
plt.tight_layout()

plt.savefig("../results/figures/engine_x_cylinders_vs_co2.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Correlation heatmap — numerical features only

numerical_cols = [
    "engine_size",
    "cylinders",
    "transmission_gears",
    "engine_per_cylinder",
    "engine_x_cylinders",
    "engine_size_sq",
    "cylinders_sq",
    "co2"
]

corr_matrix = data_clean[numerical_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="Blues",
    ax=ax,
    linewidths=0.5,
    annot_kws={"size": 9}
)

ax.set_title("Correlation Matrix — Numerical Features and CO₂")
plt.tight_layout()

plt.savefig("../results/figures/correlation_heatmap.pdf",
            format="pdf", bbox_inches="tight")
plt.show()
plt.close()

# %%
# Summary statistics for key variables

print("=" * 50)
print("Summary Statistics — Key Features")
print("=" * 50)
print(data_clean[[
    "engine_size", "cylinders", "transmission_gears",
    "engine_x_cylinders", "co2"
]].describe().round(3))

# %%



