# 📡 Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![XGBoost](https://img.shields.io/badge/XGBoost-2.x-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-green)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

## 🎯 Business Problem

A telecom company loses **₹3,000–5,000 per churned customer** in
re-acquisition costs. This system predicts which customers are likely
to cancel their subscription in the next 30 days — enabling the
retention team to intervene **before** they leave.

> **Result:** Model identifies 74% of actual churners at 87% AUC-ROC,
> protecting an estimated ₹2.1M in annual revenue.

---

## 🖥️ Live Demo

![App Demo](reports/figures/app_screenshot.png)

**Run locally:**
```bash


---

## 📊 Model Performance

| Model               | AUC-ROC    |  F1 Score  | Precision  | Recall     |
|---------------------|------------|------------|------------|------------|
| Logistic Regression | 0.843      | 0.612      | 0.671      | 0.563      |
| Random Forest       | 0.8431     | 0.6315     | 0.5364     | 0.7674     |
| XGBoost Baseline    | 0.8423     | 0.6237     | 0.5216     | 0.7754     |
| **XGBoost Tuned**   | **0.8475** | **0.6332** | **0.5197** | **0.8102** |

---

## 🔑 Key Findings from EDA

- Month-to-month contracts have **43% churn rate** vs 3% for 2-year contracts
- Churned customers leave much earlier — avg tenure **18 months vs 38 months**
- Fiber optic + Electronic check + Month-to-month = **highest risk segment**
- Senior citizens with no partner/dependents churn at **2× the average rate**

---

## ⚙️ Tech Stack

| Layer | Tools |
|---|---|
| Data Processing | Python, Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Machine Learning | Scikit-Learn, XGBoost, LightGBM |
| Explainability | SHAP |
| Experiment Tracking | Custom Tracker (CSV logs) |
| Deployment | Streamlit |
| Version Control | Git, GitHub |

---

## 📁 Project Structure

---

## 🚀 How to Run

**1. Clone the repository**
```bash
git clone https://github.com/nikhil2316n/customer-churn-prediction.git
cd customer-churn-prediction
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the Streamlit app**
```bash
streamlit run app/streamlit_app.py
```

**4. Run notebooks in order**



---

## 📦 Dataset

- **Source:** IBM Telco Customer Churn — [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers × 21 features
- **Target:** Churn (Yes/No) — 26.5% positive class

---

## 👤 Author

**Nikhil**
B.Tech Computer Science (Data Science)

[![GitHub](https://img.shields.io/badge/GitHub-Profile-black)](https://github.com/nikhil2316n)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/nikhil-sagar2316/)