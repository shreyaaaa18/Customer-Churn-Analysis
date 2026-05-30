# 📉 Customer Churn Analysis

A complete end-to-end data science project predicting customer churn for a telecom company using machine learning. Built as part of a data analytics portfolio targeting Werkstudent roles in Germany.

---

## 🎯 Business Problem

A telecom company loses approximately **26% of customers annually**, directly impacting revenue. This project answers two questions:

1. **Who is most likely to churn?**
2. **What are the key drivers of churn?**

---

## 📊 Key Findings

| Finding | Detail |
|---|---|
| 🔴 Month-to-month contracts | Churn at **42%** vs 3% for two-year contracts |
| ⚠️ Early tenure customers | First 12 months = highest risk window |
| 💸 High monthly charges | Customers paying €65+ are disproportionately at risk |
| 🌐 Fiber optic + no tech support | Highest churn segment across all service combinations |

---

## 🤖 Model Performance

| Metric | Score |
|---|---|
| AUC | **0.815** |
| Accuracy | 80% |
| Precision (churn) | 0.65 |
| Recall (churn) | 0.51 |
| F1 (churn) | 0.57 |

> Model: Random Forest Classifier (100 estimators)  
> Train/test split: 80/20 with stratification

---

## 🗂️ Project Structure

```
customer-churn-analysis/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Raw dataset (Kaggle)
│
├── notebooks/
│   └── churn_analysis.ipynb                    # Full analysis notebook
│
├── churn_dashboard.py                           # Streamlit dashboard
├── requirements.txt                             # Dependencies
└── README.md
```

---

## 🛠️ Tech Stack

- **Python 3.11**
- **pandas** — data cleaning and feature engineering
- **scikit-learn** — Random Forest model, train/test split, metrics
- **plotly** — interactive charts
- **streamlit** — dashboard deployment
- **SHAP** — model interpretability (feature importance)

---

## 🚀 How to Run

**1. Clone the repo**
```bash
git clone https://github.com/your-username/customer-churn-analysis.git
cd customer-churn-analysis
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Download the dataset**

Download from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) and place the CSV in the `data/` folder.

**4. Run the notebook**
```bash
jupyter notebook notebooks/churn_analysis.ipynb
```

**5. Launch the dashboard**
```bash
streamlit run churn_dashboard.py
```

Opens at `http://localhost:8501`

---

## 📸 Dashboard Preview

> *4 KPI cards · Churn by contract type · Churn by tenure · Top 10 churn drivers · Customer risk score table*

![Dashboard Preview]
<img width="1899" height="817" alt="image" src="https://github.com/user-attachments/assets/e400bd07-15c3-40dd-8a89-ff5a89a261f7" />
<img width="1897" height="879" alt="image" src="https://github.com/user-attachments/assets/f116076d-d171-4232-8ccd-41449c6aa9d7" />


---

## 💡 Business Recommendation

Based on the model findings, the highest ROI retention strategy is:

> **Target month-to-month customers in their first 12 months with a proactive retention offer.**

This segment covers **~42% of all churners** while representing only 22% of the customer base - making it a highly efficient targeting criterion.

---

## 📁 Dataset

- **Source:** [IBM Telco Customer Churn — Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size:** 7,043 customers · 21 features
- **Target variable:** `Churn` (Yes/No)

---

## 👩‍💻 Author

**Shreya Sonar**  
MSc Data Science  
