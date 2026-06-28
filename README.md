# 🏦 Bank Customer Churn Prediction Dashboard

A Machine Learning-powered interactive dashboard built using **Python**, **Streamlit**, and **Random Forest** to predict customer churn and provide business insights through interactive visualizations.

---

## 🚀 Live Demo

🔗 https://novqtv8k3xwxbpsvl35hqh.streamlit.app/

---

## 📌 Project Overview

Customer churn prediction is one of the most important applications of Machine Learning in the banking sector. This project predicts whether a customer is likely to leave the bank based on demographic and financial information.

The dashboard allows users to:

- Predict customer churn using Machine Learning
- Explore customer data
- Perform Exploratory Data Analysis (EDA)
- View business insights
- Download the dataset

---

## ✨ Features

- 🏠 Interactive Home Dashboard
- 🤖 Customer Churn Prediction
- 📊 Exploratory Data Analysis (EDA)
- 📈 Business Analytics Dashboard
- 📋 Dataset Search & Download
- 💡 Business Recommendations
- 🌍 Geography-wise Analysis
- 📉 Correlation Heatmap
- 📊 Interactive Plotly Visualizations

---

## 🤖 Machine Learning Models

The following Machine Learning models were trained and evaluated:

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | 80.50% |
| Decision Tree | 77.90% |
| ⭐ Random Forest | **86.25%** |

**Best Performing Model:** Random Forest

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Matplotlib
- Seaborn
- Joblib

---

## 📂 Dataset

**European Bank Customer Churn Dataset**

### Features

- Year
- Credit Score
- Geography
- Gender
- Age
- Tenure
- Balance
- Number of Products
- Credit Card Status
- Active Member Status
- Estimated Salary

### Target Variable

- Exited (Customer Churn)

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/tosif2027/Bank-Customer-Churn-Dashboard.git
```

Move into the project directory

```bash
cd Bank-Customer-Churn-Dashboard
```

Install the required dependencies

```bash
pip install -r requirements.txt
```

Generate the trained Machine Learning model

```bash
python train_model.py
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
Bank-Customer-Churn-Dashboard/
│
├── app.py
├── train_model.py
├── European_Bank.csv
├── requirements.txt
├── README.md
├── LICENSE
└── .gitignore
```

> **Note:**  
> `model.pkl` and `scaler.pkl` are intentionally not included in this repository because they are generated automatically by running `train_model.py`.

---

## 📈 Business Insights

- Germany has the highest customer churn.
- Active members are less likely to churn.
- Customers above 45 years have a higher churn risk.
- High-balance customers require better retention strategies.
- Random Forest achieved the highest prediction accuracy.

---

## 🎯 Future Improvements

- XGBoost Integration
- Deep Learning Models
- Explainable AI (SHAP)
- User Authentication
- Database Integration
- Cloud Deployment
- Real-time Prediction API

---

## 👨‍💻 Author

**Tosif Raza Mansoori**

Machine Learning & Data Analytics Project

Developed as part of the **Unified Mentor Internship Program**.

---

## 📜 License

This project is licensed under the **MIT License**.

---

⭐ If you found this project helpful, consider giving it a **Star ⭐** on GitHub.