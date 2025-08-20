# Credit Risk Probability Model using Alternative Data

## 📌 Project Overview
This project develops and deploys a **Credit Risk Probability Model** leveraging **alternative data** sources (e-commerce & bank transactions). The goal is to predict the probability of default (PD) for customers and support better credit risk assessment in line with **Basel II standards**.

The pipeline covers the **entire ML lifecycle**:
- Data preprocessing & feature engineering
- Model training & evaluation
- Explainability with SHAP
- Deployment via **FastAPI** & **Streamlit**
- CI/CD with GitHub Actions & Docker


## 🚀 Project Structure

credit-scoring-model/
│── data/                   # Raw & processed datasets
│── models/                 # Trained ML models
│── notebooks/              # EDA & experimentation
│── src/                    # Core ML pipeline
│   ├── data\_processing.py
│   ├── train.py
│   ├── predict.py
│── app/                    # FastAPI app for deployment
│   ├── main.py
│── streamlit\_app.py         # Streamlit dashboard
│── tests/                   # Unit tests
│── assets/                  # Screenshots & diagrams
│── requirements.txt
│── Dockerfile
│── README.md



## 🛠️ Tech Stack
- **Python 3.12**
- **scikit-learn, XGBoost** – Modeling
- **pandas, numpy** – Data preprocessing
- **SHAP** – Model explainability
- **FastAPI** – API deployment
- **Streamlit** – Interactive dashboard
- **Docker & GitHub Actions** – Containerization + CI/CD


## 📊 Key Features
1. **Data Pipeline** – Automated preprocessing (feature scaling, encoding, imputation).
2. **Model Training** – Logistic Regression & Random Forest (with best hyperparameters).
3. **Explainability** – SHAP summary plots for feature importance.
4. **Deployment**  
   - **FastAPI** – REST API for predictions.  
   - **Streamlit App** – Interactive UI for users to upload data & view predictions.  
5. **Automation** – CI/CD with GitHub Actions & Docker.


## ⚡ How to Run

### 1️⃣ Setup Environment
```bash
git clone https://github.com/arsema-mz/credit-scoring-model.git
cd credit-scoring-model
pip install -r requirements.txt
````

### 2️⃣ Train Models

```bash
python src/train.py
```

### 3️⃣ Run Predictions

```bash
# Run predictions on new data
python src/predict.py --model both --input data/new/new_data.csv --output data/predictions/predicted.csv
```

### 4️⃣ Start FastAPI Service

```bash
uvicorn app.main:app --reload
```

API available at 👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### 5️⃣ Launch Streamlit Dashboard

```bash
streamlit run streamlit_app.py
```


## 🖼️ Screenshots

### Streamlit Dashboard

![Streamlit App](assets/Screenshot%202025-08-20%20101118.png)



## 📈 Results

* Logistic Regression & Random Forest trained & evaluated
* SHAP explainability identified **transaction frequency & geolocation mismatch** as key fraud/risk drivers
* Interactive dashboard for real-time credit risk scoring


## 🐳 Docker Deployment

```bash
docker build -t credit-risk-model .
docker run -p 8000:8000 credit-risk-model
```
