import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── STEP 1: Load & Explore ───────────────────────────────────────────
df = pd.read_csv("Mall_Customers.csv")
print("Shape:", df.shape)
print(df.head())

# ── STEP 2: Clean & Encode ───────────────────────────────────────────
df = df.drop(columns=['CustomerID'])
df = pd.get_dummies(df, columns=['Genre'], drop_first=True)

# ── STEP 3: Scale Features ───────────────────────────────────────────
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# ── STEP 4: Find Optimal K (Elbow + Silhouette) ───────────────────────
wcss = []
silhouette_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(df_scaled)
    wcss.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(df_scaled, kmeans.labels_))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(k_range, wcss, marker='o')
axes[0].set_title('Elbow Method')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('WCSS')

axes[1].plot(k_range, silhouette_scores, marker='o', color='orange')
axes[1].set_title('Silhouette Score')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Score')

plt.tight_layout()
plt.savefig('cluster_diagnostics.png')
print("Saved cluster_diagnostics.png")

# ── STEP 5: Train Final K-Means with K=5 ─────────────────────────────
optimal_k = 5
print(f"\nUsing K={optimal_k} based on elbow method")

kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
df['Cluster'] = kmeans_final.fit_predict(df_scaled)

# ── STEP 6: Analyse Cluster Profiles ─────────────────────────────────
print("\nCluster sizes:")
print(df['Cluster'].value_counts())

print("\nCluster profiles:")
print(df.groupby('Cluster').mean().round(1))

# ── STEP 7: Visualise Segments ────────────────────────────────────────
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    df['Annual Income (k$)'],
    df['Spending Score (1-100)'],
    c=df['Cluster'], cmap='viridis', alpha=0.6, s=80
)
plt.colorbar(scatter, label='Cluster')
plt.title('Customer Segments (Income vs Spending Score)')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.savefig('customer_segments.png')
print("Saved customer_segments.png")

# ── STEP 8: Assign Business Personas & Save ───────────────────────────
persona_names = {
    0: "High-Value Trendsetters",
    1: "Affluent Conservatives",
    2: "Mid-Tier Regulars",
    3: "Young Impulsive Spenders",
    4: "Mature Minimalists"
}
df['Persona'] = df['Cluster'].map(persona_names)

print("\nFinal persona distribution:")
print(df[['Cluster', 'Persona']].value_counts())

df.to_csv('Mall_Customers_Segmented.csv', index=False)
print("\nFinal dataset saved as Mall_Customers_Segmented.csv")