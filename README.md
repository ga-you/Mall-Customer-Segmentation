# 🛍️ Mall Customer Segmentation

A small unsupervised learning project. The dataset had no labels, so I let K-Means find the customer groups on its own, then tried to make sense of them as real customer types.

## 🤔 Why I built this

I wanted to see what happens when you give an algorithm customer data and ask it to find patterns without telling it what to look for. The interesting part wasn't the code. It was figuring out what the clusters actually meant.

## ⚙️ What I did

**1. Cleaned the data**
- Dropped `CustomerID` since it's just an identifier
- One-hot encoded `Genre`

**2. Scaled the features**
- Annual Income ranges from about $15k to $137k, while Spending Score only goes from 1 to 100. Without scaling, income would dominate the distance calculations, so I used `StandardScaler` to put everything on equal footing.

**3. Picked the number of clusters**
- Tried K from 2 to 10 using the elbow method (WCSS) and silhouette score
- The elbow bends around K=5, so I went with that

**4. Trained K-Means with K=5**
- Used 10 random initializations to avoid bad local minima

**5. Turned clusters into personas**
- Looked at average age, income, and spending score for each group and gave them names

## 👥 The 5 groups

| Cluster | Persona | Age | Income | Spending Score | Size |
|---|---|---|---|---|---|
| 0 | High-Value Trendsetters | 32.7 | $86.5k | 82.1 | 39 |
| 1 | Affluent Conservatives | 36.5 | $89.5k | 18.0 | 29 |
| 2 | Mid-Tier Regulars | 49.8 | $49.2k | 40.1 | 43 |
| 3 | Young Impulsive Spenders | 24.9 | $39.7k | 61.2 | 54 |
| 4 | Mature Minimalists | 55.7 | $53.7k | 36.8 | 35 |

**What I'd suggest a marketing team do with these:**
- **Trendsetters:** VIP programs, early access, premium experiences
- **Conservatives:** re-engagement campaigns and value-focused messaging to bring out their spending
- **Regulars:** bundle deals and seasonal promotions to nudge spending up
- **Young Impulsive Spenders:** flash sales, influencer campaigns, buy-now-pay-later options
- **Minimalists:** clear value propositions and reliability-focused messaging

## 💭 What I took away

The silhouette score actually preferred K=10, but 10 groups of 13–26 people each isn't something anyone could act on. Going with K=5 was a judgment call, and I think that's the part that matters most. The math gives you options, and you still have to decide which one is useful.

## 🧰 Tech stack
Python · Pandas · NumPy · Scikit-Learn · Matplotlib

## 📁 Dataset
[Mall Customer Segmentation Data on Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)

## 📂 Files
- `project3.py`: the full clustering pipeline
- `Mall_Customers_Segmented.csv`: the dataset with cluster and persona labels
- `cluster_diagnostics.png`: elbow and silhouette charts
- `customer_segments.png`: scatter plot of the segments
