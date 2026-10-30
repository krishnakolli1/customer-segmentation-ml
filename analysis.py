
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Create output folders
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

print("=" * 55)
print("CUSTOMER SEGMENTATION - DATASET ANALYSIS")
print("=" * 55)

# Basic information
print("\nDataset shape:", df.shape)
print("\nFirst 5 records:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())

print("\nStatistical summary:")
print(df.describe())

# Gender distribution
print("\nGender distribution:")
print(df["Gender"].value_counts())

# Age and income analysis
print("\nAverage age:", round(df["Age"].mean(), 2))
print("Average annual income:", round(df["Annual Income (k$)"].mean(), 2))
print("Average spending score:", round(df["Spending Score (1-100)"].mean(), 2))

# Correlation analysis
numeric_df = df.select_dtypes(include="number")
correlation = numeric_df.corr()

print("\nCorrelation with Spending Score:")
print(correlation["Spending Score (1-100)"].sort_values(ascending=False))

# Save summary reports
df.describe().to_csv("reports/dataset_summary.csv")
correlation.to_csv("reports/feature_correlations.csv")

# 1. Age distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Age"], bins=20, kde=True)
plt.title("Customer Age Distribution")
plt.xlabel("Age")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/age_distribution.png")
plt.close()

# 2. Annual income distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Annual Income (k$)"], bins=20, kde=True)
plt.title("Annual Income Distribution")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/income_distribution.png")
plt.close()

# 3. Spending score distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Spending Score (1-100)"], bins=20, kde=True)
plt.title("Customer Spending Score Distribution")
plt.xlabel("Spending Score")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/spending_score_distribution.png")
plt.close()

# 4. Income vs spending score
plt.figure(figsize=(8, 5))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Gender"
)
plt.title("Annual Income vs Spending Score")
plt.tight_layout()
plt.savefig("charts/income_vs_spending.png")
plt.close()

print("\nAnalysis completed successfully!")
print("Charts saved in charts/ folder.")
print("Reports saved in reports/ folder.")
