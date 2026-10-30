# Customer Segmentation Using Machine Learning

## Project Overview
This project uses unsupervised machine learning to divide mall customers into meaningful groups based on their annual income and spending behavior.

K-Means clustering is used to identify five customer segments and provide insights that can help businesses improve marketing strategies and customer engagement.

## Dataset
- Dataset: Mall Customers
- Source: Kaggle – Customer Segmentation Tutorial
- Total records: 200
- Features: Customer ID, Gender, Age, Annual Income, Spending Score
- Missing values: 0
- Duplicate records: 0

## Technologies Used
- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- K-Means Clustering
- StandardScaler
- PCA (Principal Component Analysis)

## Project Workflow
1. Load and explore customer data.
2. Perform data cleaning and exploratory data analysis.
3. Analyze age, income, and spending patterns.
4. Standardize annual income and spending score.
5. Use the Elbow Method and Silhouette Score to evaluate cluster counts.
6. Train the final K-Means model with five clusters.
7. Create customer segment profiles.
8. Visualize customer groups using scatterplots and PCA.
9. Export customer segments and analytical reports.

## Model Evaluation
The project evaluated cluster counts from 2 to 10.

- Selected clusters: 5
- Silhouette Score: 0.5547

The five-cluster solution achieved the highest silhouette score among the tested configurations.

## Customer Segments

| Cluster | Customers | Average Income (k$) | Average Spending Score | Segment |
|---|---:|---:|---:|---|
| 0 | 81 | 55.30 | 49.52 | Average Customers |
| 1 | 39 | 86.54 | 82.13 | Premium Customers |
| 2 | 22 | 25.73 | 79.36 | Young High Spenders |
| 3 | 35 | 88.20 | 17.11 | High-Income Low Spenders |
| 4 | 23 | 26.30 | 20.91 | Budget-Conscious Customers |

## Business Recommendations
- Premium customers: Offer loyalty rewards and exclusive products.
- Young high spenders: Use personalized promotions and trending products.
- High-income low spenders: Develop targeted campaigns to encourage purchases.
- Budget-conscious customers: Provide discounts and value bundles.
- Average customers: Encourage repeat purchases through cross-selling.

## Visualizations
The `charts/` directory contains:
- Age distribution
- Annual income distribution
- Spending score distribution
- Income versus spending score
- Elbow method
- Silhouette scores
- Customer segmentation
- PCA visualization
- Cluster sizes

## Reports
The `reports/` directory contains:
- Dataset summary
- Feature correlations
- Cluster evaluation
- Customer segments
- Cluster profiles

## How to Run

1. Clone this repository.
2. Install dependencies