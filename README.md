# Credit Risk Model using Alternative Data

This project implements an end-to-end credit scoring model pipeline using alternative data sources. It includes data ingestion, preprocessing, model training, deployment via API, and CI/CD automation.

## 🚀 Goals
- Build a transparent and interpretable credit scoring model
- Use alternative credit data (no traditional credit score)
- Deploy the model with FastAPI
- Automate with CI/CD and Docker


## 📘 Credit Scoring Business Understanding

### 1. Basel II and the Need for Interpretable Models

The Basel II Accord emphasizes the need for accurate, transparent, and auditable credit risk models. Financial institutions are encouraged to adopt internal rating systems that comply with regulatory standards and can be inspected by external auditors. Therefore, the models we build must be interpretable, reproducible, and explainable. Simple models such as Logistic Regression with Weight of Evidence (WoE) encoding are often favored due to their clarity and regulatory friendliness.

### 2. Why We Need a Proxy Variable for Default

In many alternative data scenarios (e.g., telecom, utility data, mobile money), the dataset may not include an explicit "default" label. To overcome this, we create a **proxy variable** that approximates the default behavior (e.g., payment delay > 90 days). While this enables us to train a supervised model, it introduces risks:
- The proxy may not generalize well to true default behavior.
- Mislabeling can bias model predictions.
- Regulatory bodies may question the validity of proxy-based conclusions.

### 3. Trade-offs: Interpretable vs. High-Performance Models

| Feature            | Simple Models (e.g., Logistic Regression) | Complex Models (e.g., XGBoost)         |
|--------------------|--------------------------------------------|-----------------------------------------|
| Interpretability   | ✅ Easy to explain                          | ❌ Hard to interpret                     |
| Regulatory Approval| ✅ Favorable                                | ⚠️ Requires explainability tools         |
| Predictive Power   | ⚠️ Limited                                  | ✅ Often better performance              |
| Auditability       | ✅ High                                     | ⚠️ More effort required                  |


In a regulated environment, interpretable models are often required for decision-making, even if performance is slightly lower. However, ensemble methods like Gradient Boosting may be used in internal risk scoring, provided proper justification and explanation (e.g., SHAP values) are available.

### 📚 References
- [Basel II and Credit Risk](https://www3.stat.sinica.edu.tw/statistica/oldpdf/A28n535.pdf)
- [HKMA: Alternative Credit Scoring](https://www.hkma.gov.hk/media/eng/doc/key-functions/financial-infrastructure/alternative_credit_scoring.pdf)
- [World Bank Credit Scoring](https://thedocs.worldbank.org/en/doc/935891585869698451-0130022020/original/CREDITSCORINGAPPROACHESGUIDELINESFINALWEB.pdf)


## 🔍 Exploratory Data Analysis (EDA)

The goal of EDA was to understand the structure, distribution, and quality of the dataset before modeling. Key steps included:

- **Data Inspection**: Loaded and reviewed data types, shape, and column names.
- **Missing Values**: Checked for null values; the dataset is mostly clean.
- **Target Distribution**: Identified extreme class imbalance in `FraudResult` (~0.2% fraud).
- **Feature Distributions**:
  - `amount` and `value` are highly right-skewed, with a few large outliers.
  - Applied log transformation to normalize `value`.
- **Datetime Parsing**: Converted `TransactionStartTime` to datetime and extracted features like hour and day.
- **Categorical Features**: Visualized frequency distributions for variables like `productcategory` and `channelid`.
- **Correlation Analysis**: Heatmap revealed weak linear relationships; no strong multicollinearity.

## Data Processing

* Raw data is loaded from `data/raw/data.csv`.
* Preprocessing includes datetime feature extraction, aggregation, categorical encoding, missing value imputation, and numerical scaling.
* Feature engineering is implemented as sklearn-compatible transformers inside `src/data_processing.py`.
* Processed data is saved in `data/processed/processed.csv`.


## Proxy Target Variable Engineering

* RFM (Recency, Frequency, Monetary) metrics are computed per customer.
* K-Means clustering (3 clusters) segments customers.
* The cluster with the highest risk profile is labeled `is_high_risk = 1`.
* This label is merged back into the dataset for supervised learning.


## Model Training and Evaluation

* Models trained: Logistic Regression, Random Forest.
* Hyperparameter tuning done via GridSearchCV.
* Evaluation metrics: Accuracy, Precision, Recall, F1-score, ROC-AUC.
* Best models saved as `.pkl` files in `models/`.


## API Deployment

* REST API built with FastAPI (`src/api/main.py`).

* Pydantic models used for input validation (`src/api/pydantic_models.py`).

* API loads the trained model with `joblib`.

* Endpoint `/predict` accepts JSON customer data and returns risk prediction and probability.
* Access API docs at: http://localhost:8000/docs

