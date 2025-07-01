import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def create_high_risk_label(df, snapshot_date='2025-07-01'):
    # Ensure transaction date column is datetime
    df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime'])
    
    # Convert snapshot_date to datetime
    snapshot_date = pd.to_datetime(snapshot_date)

    # Aggregate per customer
    rfm = df.groupby('customerid').agg({
        'transactionstarttime': lambda x: (snapshot_date - x.max()).days,
        'transactionid': 'count',      # Frequency
        'amount': 'sum'                # Monetary
    }).rename(columns={
        'transactionstarttime': 'Recency',
        'transactionid': 'Frequency',
        'amount': 'Monetary'
    }).reset_index()

    # Handle cases where monetary might be zero or missing - optional
    rfm['Monetary'] = rfm['Monetary'].fillna(0)

    # Scale RFM features
    scaler = StandardScaler()
    rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

    # Analyze cluster centers to find high-risk cluster
    cluster_centers = kmeans.cluster_centers_

    # In scaled space: High Recency = high positive value (long ago),
    # low Frequency & Monetary = low or negative values.
    # Find cluster with highest Recency and lowest Frequency & Monetary
    # We'll use a heuristic: cluster with max Recency mean and min Frequency and Monetary means.

    import numpy as np

    recency_idx = 0
    frequency_idx = 1
    monetary_idx = 2

    scores = []
    for i, center in enumerate(cluster_centers):
        # Score: high recency + low freq + low monetary
        score = center[recency_idx] - center[frequency_idx] - center[monetary_idx]
        scores.append((i, score))

    # Cluster with max score is high risk
    high_risk_cluster = max(scores, key=lambda x: x[1])[0]

    # Assign is_high_risk label
    rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)

    # Drop cluster column if you want
    rfm = rfm.drop(columns=['cluster'])

    return rfm[['customerid', 'is_high_risk']]

# Usage example: 
# rfm_labels = create_high_risk_label(df)
# Then merge back:
# df = df.merge(rfm_labels, on='customerid', how='left')
# Fill any missing is_high_risk with 0 (if some customers not present in RFM)
# df['is_high_risk'] = df['is_high_risk'].fillna(0).astype(int)
