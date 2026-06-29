# 🛍️ Mall Customer Segmentation — Unsupervised Learning

This is my Project 3 submission for the DecodeLabs Data Science Industrial Training (Batch 2026). Unlike the previous projects, this one had no "right answers" to learn from — the data came with zero labels. The goal was to let the math discover hidden customer groups on its own, then translate those groups into real business personas a marketing team could actually use.

## 🧠 What I did

**1. Cleaned & encoded the data**
- Dropped the useless `CustomerID` column
- One-Hot Encoded the `Genre` column

**2. Scaled all features using StandardScaler**
- Without scaling, Annual Income ($15k–$137k) would completely overpower Spending Score (1–100) in the distance calculations K-Means relies on
- Scaling gives every feature equal mathematical "voting power"

**3. Found the optimal number of clusters**
- Used the Elbow Method (WCSS) and Silhouette Score across K=2 to K=10
- Elbow curve bent at K=5, confirming 5 as the optimal number of segments

**4. Trained K-Means with K=5**
- Algorithm assigned every customer to their nearest cluster centroid
- Ran with 10 different random initialisations to avoid getting stuck in a local minimum

**5. Interpreted clusters as business personas**
- Analysed average Age, Income, and Spending Score per cluster
- Translated raw math into actionable customer personas

## 👥 The 5 Customer Personas

| Cluster | Persona | Age | Income | Spending Score | Size |
|---|---|---|---|---|---|
| 0 | High-Value Trendsetters | 32.7 | $86.5k | 82.1 | 39 |
| 1 | Affluent Conservatives | 36.5 | $89.5k | 18.0 | 29 |
| 2 | Mid-Tier Regulars | 49.8 | $49.2k | 40.1 | 43 |
| 3 | Young Impulsive Spenders | 24.9 | $39.7k | 61.2 | 54 |
| 4 | Mature Minimalists | 55.7 | $53.7k | 36.8 | 35 |

**Business actions:**
- **High-Value Trendsetters** → VIP loyalty programs, early product access, premium experiences
- **Affluent Conservatives** → targeted re-engagement campaigns, value-focused messaging to unlock their spending potential
- **Mid-Tier Regulars** → bundle deals, seasonal promotions to nudge spending upward
- **Young Impulsive Spenders** → flash sales, influencer campaigns, buy-now-pay-later options
- **Mature Minimalists** → utility-focused messaging, clear value propositions, reliability guarantees

## 📊 What I learned

This project was the most interesting of the three because there was no "correct" answer to aim for — the algorithm found patterns I couldn't have spotted just by looking at a spreadsheet. The most important lesson was that math alone isn't enough: K=10 was technically "optimal" by the silhouette score, but 10 groups of 13-26 people each is useless for a marketing team. Knowing when to override the math with human judgment is a real data science skill.

## 🧰 Tech stack
Python · Pandas · NumPy · Scikit-Learn · Matplotlib

## 📁 Dataset source
[Mall Customer Segmentation Data — Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## 📂 Files
- `project3.py` — full clustering pipeline
- `Mall_Customers_Segmented.csv` — final dataset with cluster and persona labels
- `cluster_diagnostics.png` — elbow method and silhouette score charts
- `customer_segments.png` — scatter plot of customer segments