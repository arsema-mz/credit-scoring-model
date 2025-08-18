import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import os
from sklearn.cluster import KMeans

# ===== Custom Transformers =====

class DateTimeFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, datetime_column='transactionstarttime'):
        self.datetime_column = datetime_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        df[self.datetime_column] = pd.to_datetime(df[self.datetime_column], errors='coerce')
        df['transaction_hour'] = df[self.datetime_column].dt.hour
        df['transaction_day'] = df[self.datetime_column].dt.day
        df['transaction_month'] = df[self.datetime_column].dt.month
        df['transaction_year'] = df[self.datetime_column].dt.year
        return df

class CustomerAggregateFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, customer_id_col='customerid', amount_col='amount'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy()
        agg = df.groupby(self.customer_id_col)[self.amount_col].agg([
            'sum', 'mean', 'count', 'std'
        ]).fillna(0)

        agg.columns = [
            'total_amount',
            'average_amount',
            'transaction_count',
            'amount_std'
        ]

        df = df.merge(agg, on=self.customer_id_col, how='left')
        return df

class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, one_hot_cols=None, label_encode_cols=None):
        self.one_hot_cols = one_hot_cols or []
        self.label_encode_cols = label_encode_cols or []
        self.encoders = {}

    def fit(self, X, y=None):
        for col in self.label_encode_cols:
            le = LabelEncoder()
            le.fit(X[col].astype(str))
            self.encoders[col] = le
        return self

    def transform(self, X):
        df = X.copy()

        # Label Encoding
        for col in self.label_encode_cols:
            df[col] = self.encoders[col].transform(df[col].astype(str))

        # One-Hot Encoding
        if self.one_hot_cols:
            df = pd.get_dummies(df, columns=self.one_hot_cols, drop_first=True)

        return df

class MissingValueHandler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.num_cols = []
        self.cat_cols = []

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.cat_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns
        self.num_imputer.fit(X[self.num_cols])
        self.cat_imputer.fit(X[self.cat_cols])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.num_cols] = self.num_imputer.transform(df[self.num_cols])
        df[self.cat_cols] = self.cat_imputer.transform(df[self.cat_cols])
        return df

class NumericalScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.scaler = StandardScaler()
        self.num_cols = []

    def fit(self, X, y=None):
        self.num_cols = X.select_dtypes(include=['int64', 'float64']).columns
        self.scaler.fit(X[self.num_cols])
        return self

    def transform(self, X):
        df = X.copy()
        df[self.num_cols] = self.scaler.transform(df[self.num_cols])
        return df


# ===== Main Processing Pipeline =====

def process_data(input_path='data/raw/data.csv', output_path='data/processed/processed.csv'):
    df = pd.read_csv(input_path)

    # Clean column names
    df.columns = df.columns.str.lower()

    # Define pipeline
    pipeline = Pipeline([
        ('datetime_features', DateTimeFeatures(datetime_column='transactionstarttime')),
        ('aggregate_features', CustomerAggregateFeatures(customer_id_col='customerid', amount_col='amount')),
        ('categorical_encoding', CategoricalEncoder(
            one_hot_cols=['productcategory', 'currencycode'],
            label_encode_cols=['providerid', 'channelid', 'productid']
        )),
        ('missing_value_imputation', MissingValueHandler()), 
        ('scaling', NumericalScaler())
    ])

    # Apply transformation pipeline
    df_processed = pipeline.fit_transform(df)

    # ===== High-Risk Target Engineering =====
    def create_high_risk_label(df, snapshot_date='2025-07-01'):
        df['transactionstarttime'] = pd.to_datetime(df['transactionstarttime']).dt.tz_localize(None)
        snapshot_date = pd.to_datetime(snapshot_date).tz_localize(None)

        rfm = df.groupby('customerid').agg({
            'transactionstarttime': lambda x: (snapshot_date - x.max()).days,
            'transactionid': 'count',
            'amount': 'sum'
        }).rename(columns={
            'transactionstarttime': 'Recency',
            'transactionid': 'Frequency',
            'amount': 'Monetary'
        }).reset_index()

        rfm['Monetary'] = rfm['Monetary'].fillna(0)

        scaler = StandardScaler()
        rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])

        kmeans = KMeans(n_clusters=3, random_state=42)
        rfm['cluster'] = kmeans.fit_predict(rfm_scaled)

        # Scoring: high recency, low freq/monetary = high risk
        cluster_centers = kmeans.cluster_centers_
        scores = [
            (i, center[0] - center[1] - center[2])
            for i, center in enumerate(cluster_centers)
        ]
        high_risk_cluster = max(scores, key=lambda x: x[1])[0]
        rfm['is_high_risk'] = (rfm['cluster'] == high_risk_cluster).astype(int)
        return rfm[['customerid', 'is_high_risk']]
    

# ===== API Input Processor =====

def process_input(data: dict):
    """
    Takes a single JSON-like dict (from API request) and applies preprocessing
    pipeline to return model-ready features.
    """
    df = pd.DataFrame([data])  # turn dict into one-row DataFrame

    # Clean column names to match training
    df.columns = df.columns.str.lower()

    # Define same preprocessing pipeline (without target engineering)
    pipeline = Pipeline([
        ('datetime_features', DateTimeFeatures(datetime_column='transactionstarttime')),
        ('aggregate_features', CustomerAggregateFeatures(customer_id_col='customerid', amount_col='amount')),
        ('categorical_encoding', CategoricalEncoder(
            one_hot_cols=['productcategory', 'currencycode'],
            label_encode_cols=['providerid', 'channelid', 'productid']
        )),
        ('missing_value_imputation', MissingValueHandler()), 
        ('scaling', NumericalScaler())
    ])

    processed = pipeline.fit_transform(df)

    # Drop target if accidentally created
    if "is_high_risk" in processed.columns:
        processed = processed.drop(columns=["is_high_risk"])

    return processed




    # Generate target labels and merge into processed data
    rfm_labels = create_high_risk_label(df)
    df_processed = df_processed.merge(rfm_labels, on='customerid', how='left')
    df_processed['is_high_risk'] = df_processed['is_high_risk'].fillna(0).astype(int)

    # Save processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_processed.to_csv(output_path, index=False)
    print(f"[✓] Processed data saved to: {output_path}")

if __name__ == "__main__":
    process_data()