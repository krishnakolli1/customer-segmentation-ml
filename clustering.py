
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# Create output folders
os.makedirs("charts", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load dataset
df = pd.read_csv("data/Mall_Customers.csv")

# Select clustering features
features = ["Annual Income (k$)", "Spending Score (1-100)"]
X = df[features]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train final K-Means model
kmeans = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

df["Cluster"] = kmeans.fit_predict(X_scaled)

# Evaluate final model
score = silhouette_score(X_scaled, df["Cluster"])
print("Final Silhouette Score:", round(score, 4))

# Cluster sizes
print("\nCustomers in each cluster:")
print(df["Cluster"].value_counts().sort_index())

# Cluster profiles
cluster_summary = df.groupby("Cluster").agg(
    Customer_Count=("CustomerID", "count"),
    Average_Age=("Age", "mean"),
    Average_Income=("Annual Income (k$)", "mean"),
    Average_Spending_Score=("Spending Score (1-100)", "mean")
).round(2)

print("\nCUSTOMER SEGMENT PROFILES")
print(cluster_summary)

# Save segmented customer data and profiles
df.to_csv("reports/customer_segments.csv", index=False)
cluster_summary.to_csv("reports/cluster_profiles.csv")

# 1. Customer segmentation scatterplot
plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segmentation Using K-Means")
plt.tight_layout()
plt.savefig("charts/customer_segments.png")
plt.close()

# 2. Cluster sizes
plt.figure(figsize=(8, 5))
sns.countplot(data=df, x="Cluster", hue="Cluster", palette="viridis", legend=False)
plt.title("Number of Customers in Each Segment")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.savefig("charts/cluster_sizes.png")
plt.close()

# 3. PCA visualization (two-dimensional representation)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame({
    "PCA1": pca_result[:, 0],
    "PCA2": pca_result[:, 1],
    "Cluster": df["Cluster"]
})

plt.figure(figsize=(10, 6))
sns.scatterplot(
    data=pca_df,
    x="PCA1",
    y="PCA2",
    hue="Cluster",
    palette="viridis",
    s=80
)

plt.title("Customer Segments - PCA Visualization")
plt.tight_layout()
plt.savefig("charts/pca_visualization.png")
plt.close()

print("\nFinal customer segmentation completed successfully!")
print("Charts saved in charts/ folder.")
print("Reports saved in reports/ folder.")
