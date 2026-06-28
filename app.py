# ==========================================================
# BANK CUSTOMER CHURN PREDICTION DASHBOARD
# ==========================================================

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Bank Customer Churn Dashboard",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.info(
"""
🏦 **Bank Customer Churn Prediction Dashboard**

This dashboard predicts customer churn using Machine Learning and provides business insights through interactive visualizations.

🎯 Best Model : Random Forest

🎯 Accuracy : 86.25%
"""
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background:#F5F7FA;
}

h1{
    color:#1E3A8A;
    font-weight:700;
}

h2,h3{
    color:#2563EB;
}

div[data-testid="metric-container"]{

    background:white;

    border-radius:18px;

    padding:20px;

    border-left:8px solid #2563EB;

    box-shadow:0px 8px 20px rgba(0,0,0,.08);

    transition:0.3s;
}

div[data-testid="metric-container"]:hover{

    transform:scale(1.03);

}

section[data-testid="stSidebar"]{

    background:#0F172A;

}

section[data-testid="stSidebar"] *{

    color:white;

}

.stButton>button{

    width:100%;

    background:#2563EB;

    color:white;

    border:none;

    border-radius:10px;

    padding:12px;

    font-weight:bold;

}

.stButton>button:hover{

    background:#1D4ED8;

}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD DATA
# ==========================================================

@st.cache_data
def load_data():

    df = pd.read_csv("European_Bank.csv")

    df.columns = df.columns.str.strip()

    df.drop_duplicates(inplace=True)

    df.drop(
        columns=[
            "CustomerId",
            "Surname"
        ],
        inplace=True
    )

    return df


df = load_data()

# ==========================================================
# LOAD MODEL (AUTO GENERATE IF MISSING)
# ==========================================================

import joblib

@st.cache_resource
def load_model():

    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

    return model, scaler


model, scaler = load_model()

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.image(
    "https://img.icons8.com/color/480/bank-building.png",
    width=120
)

st.sidebar.title("🏦 Bank Churn Dashboard")

st.sidebar.markdown("---")

page = st.sidebar.radio(

    "Navigation",

    [

        "🏠 Home",

        "📊 EDA",

        "🤖 Prediction",

        "📈 Analytics",

        "📋 Dataset",

        "ℹ About"

    ]

)

st.sidebar.markdown("---")

st.sidebar.success("Machine Learning Project")

st.sidebar.info(
"""
Random Forest Accuracy

86.25%
"""
)

st.sidebar.markdown("---")

st.sidebar.caption("Version 1.0")

st.sidebar.caption("© 2026 Tosif Raza Mansoori")

# ==========================================================
# HOME PAGE
# ==========================================================

if page=="🏠 Home":

    st.title("🏦 Bank Customer Churn Prediction Dashboard")

    st.write(
        """
        Welcome to the AI-powered Customer Churn Dashboard.

        This dashboard helps analyze customer behaviour,
        identify churn patterns,
        and predict whether a customer is likely to leave the bank.
        """
    )

    st.markdown("---")

    total_customers=len(df)

    churned=df["Exited"].sum()

    active=total_customers-churned

    churn_rate=(churned/total_customers)*100

    avg_credit=df["CreditScore"].mean()

    col1,col2,col3,col4,col5=st.columns(5)

    col1.metric(
        "Customers",
        f"{total_customers:,}"
    )

    col2.metric(
        "Churned",
        churned
    )

    col3.metric(
        "Active",
        active
    )

    col4.metric(
        "Churn Rate",
        f"{churn_rate:.2f}%"
    )

    col5.metric(
        "Avg Credit Score",
        f"{avg_credit:.0f}"
    )

    st.markdown("---")

    st.subheader("📌 Dashboard Features")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.success("""
    ### 🤖 AI Prediction

    Predict whether a customer is likely to churn using the trained Random Forest model.
    """)

    with c2:
        st.info("""
    ### 📊 Interactive Analytics

    Explore customer demographics, churn patterns, and business insights.
    """)

    with c3:
        st.warning("""
    ### 📈 Business Intelligence

    Support data-driven decision making through visual dashboards.
    """)
    

# ==========================================================
# EDA PAGE
# ==========================================================

elif page == "📊 EDA":

    st.title("📊 Exploratory Data Analysis")

    st.write("Visual analysis of customer demographics and churn behaviour.")

    st.markdown("---")

    # Helper Function
    def show_graph(fig):
        col1, col2, col3 = st.columns([1, 3, 1])

        with col2:
            st.pyplot(fig, use_container_width=True)

    # ==========================================================
    # 1. Customer Churn Distribution
    # ==========================================================

    st.subheader("1️⃣ Customer Churn Distribution")

    fig = px.histogram(
        df,
        x="Exited",
        color="Exited",
        text_auto=True,
        color_discrete_sequence=["#4CAF50", "#F44336"]
    )

    fig.update_xaxes(
        tickvals=[0, 1],
        ticktext=["Stayed", "Churned"]
    )

    fig.update_layout(
        height=500,
        title="Customer Churn Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 2. Gender Distribution
    # ==========================================================

    st.subheader("2️⃣ Gender Distribution")

    fig = px.histogram(
        df,
        x="Gender",
        color="Gender",
        text_auto=True
    )

    fig.update_layout(
        height=500,
        title="Gender Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 3. Geography Distribution
    # ==========================================================

    st.subheader("3️⃣ Geography Distribution")

    fig = px.histogram(
        df,
        x="Geography",
        color="Geography",
        text_auto=True
    )

    fig.update_layout(
        height=500,
        title="Customer Distribution by Geography",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 4. Age Distribution
    # ==========================================================

    st.subheader("4️⃣ Age Distribution")

    fig = px.histogram(
        df,
        x="Age",
        nbins=30,
        color_discrete_sequence=["royalblue"]
    )

    fig.update_layout(
        height=500,
        title="Age Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 5. Credit Score Distribution
    # ==========================================================

    st.subheader("5️⃣ Credit Score Distribution")

    fig = px.histogram(
        df,
        x="CreditScore",
        nbins=30,
        color_discrete_sequence=["green"]
    )

    fig.update_layout(
        height=500,
        title="Credit Score Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 6. Balance Distribution
    # ==========================================================

    st.subheader("6️⃣ Balance Distribution")

    fig = px.histogram(
        df,
        x="Balance",
        nbins=30,
        color_discrete_sequence=["orange"]
    )

    fig.update_layout(
        height=500,
        title="Balance Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 7. Estimated Salary Distribution
    # ==========================================================


    st.subheader("7️⃣ Estimated Salary Distribution")

    fig = px.histogram(
        df,
        x="EstimatedSalary",
        nbins=30,
        color_discrete_sequence=["purple"]
    )

    fig.update_layout(
        height=500,
        title="Estimated Salary Distribution",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 8. Churn by Gender
    # ==========================================================

    st.subheader("8️⃣ Churn by Gender")

    fig = px.histogram(
        df,
        x="Gender",
        color="Exited",
        barmode="group",
        color_discrete_sequence=["green", "red"]
    )

    fig.update_layout(
        height=500,
        title="Churn by Gender",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 9. Churn by Geography
    # ==========================================================

    st.subheader("9️⃣ Churn by Geography")

    fig = px.histogram(
        df,
        x="Geography",
        color="Exited",
        barmode="group",
        color_discrete_sequence=["green", "red"]
    )

    fig.update_layout(
        height=500,
        title="Churn by Geography",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

    # ==========================================================
    # 10. Correlation Heatmap
    # ==========================================================

    st.subheader("🔟 Correlation Heatmap")

    corr = df.select_dtypes(include="number").corr()

    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        aspect="auto"
    )

    fig.update_layout(
        height=700,
        title="Correlation Heatmap",
        title_x=0.5
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DATASET PAGE
# ==========================================================

elif page == "📋 Dataset":

    st.title("📋 Customer Dataset")

    st.markdown("Search any customer record below.")

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Rows",
        len(df)
    )

    c2.metric(
        "Columns",
        len(df.columns)
    )

    c3.metric(
        "Missing Values",
        df.isnull().sum().sum()
    )

    st.markdown("---")

    # Search Box
    search = st.text_input("🔍 Search Customer")

    if search:

        filtered = df[
            df.astype(str)
            .apply(lambda x: x.str.contains(search, case=False))
            .any(axis=1)
        ]

        st.dataframe(
            filtered,
            use_container_width=True
        )

    else:

        st.dataframe(
            df,
            use_container_width=True
        )

    st.markdown("---")

    # Download Button
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇ Download Dataset",
        data=csv,
        file_name="Bank_Customer_Data.csv",
        mime="text/csv"
    )

# ==========================================================
# PREDICTION PAGE
# ==========================================================

elif page == "🤖 Prediction":

    st.title("🤖 Customer Churn Prediction")

    st.write("Enter customer details to predict whether the customer is likely to churn.")

    col1, col2 = st.columns(2)

    with col1:

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650
        )

        geography = st.selectbox(
            "Geography",
            ["France", "Germany", "Spain"]
        )

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=100,
            value=35
        )

        tenure = st.slider(
            "Tenure",
            0,
            10,
            5
        )

    with col2:

        balance = st.number_input(
            "Balance",
            min_value=0.0,
            value=50000.0
        )

        products = st.slider(
            "Number of Products",
            1,
            4,
            1
        )

        has_card = st.selectbox(
            "Has Credit Card",
            [0, 1]
        )

        active = st.selectbox(
            "Is Active Member",
            [0, 1]
        )

        salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0
        )

    if st.button("🔍 Predict Churn"):

        # Encoding
        geo_map = {
            "France": 0,
            "Germany": 1,
            "Spain": 2
        }

        gender_map = {
            "Female": 0,
            "Male": 1
        }

        input_data = pd.DataFrame([{

            "CreditScore": credit_score,
            "Geography": geo_map[geography],
            "Gender": gender_map[gender],
            "Age": age,
            "Tenure": tenure,
            "Balance": balance,
            "NumOfProducts": products,
            "HasCrCard": has_card,
            "IsActiveMember": active,
            "EstimatedSalary": salary

        }])

        input_scaled = scaler.transform(input_data)

        prediction = model.predict(input_scaled)[0]

        probability = model.predict_proba(input_scaled)[0][1]

        st.markdown("---")

        st.markdown("---")

        st.subheader("🤖 Prediction Result")

        if prediction == 1:

            st.error("❌ Customer is likely to CHURN")

        else:

            st.success("✅ Customer is likely to STAY")

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Prediction Probability",
                f"{probability*100:.2f}%"
            )

        with col2:

            st.metric(
                "Model Used",
                "Random Forest"
            )

        st.markdown("---")

        # Risk Level

        if probability >= 0.80:

            st.error("🔴 HIGH RISK")

        elif probability >= 0.50:

            st.warning("🟡 MEDIUM RISK")

        else:

            st.success("🟢 LOW RISK")

        st.markdown("---")

        st.subheader("💡 Business Recommendation")

        if prediction == 1:

            st.warning("""

        ✅ Contact the customer immediately

        ✅ Offer personalized discounts

        ✅ Assign a Relationship Manager

        ✅ Recommend premium banking services

        ✅ Improve customer engagement

        """)

        else:

            st.success("""

        ✅ Customer is loyal

        ✅ Continue regular engagement

        ✅ Offer cross-selling opportunities

        ✅ Maintain current service quality

        """)

        st.markdown("---")

        st.subheader("📊 Probability Meter")

        st.progress(float(probability))

        st.write(f"**Prediction Confidence:** {probability*100:.2f}%")

# ==========================================================
# ANALYTICS PAGE
# ==========================================================

elif page == "📈 Analytics":

    st.title("📈 Business Analytics Dashboard")

    st.markdown("### Executive Business Summary")

    st.markdown("---")

    st.subheader("📈 Executive Summary")

    st.success("""

    ✔ Random Forest achieved 86.25% accuracy.

    ✔ Germany has the highest churn.

    ✔ Active members are less likely to leave.

    ✔ High balance customers require retention strategies.

    ✔ Predictive Analytics helps reduce customer loss.

    """)

    # ==============================
    # KPI CARDS
    # ==============================

    total_customers = len(df)

    churned = df["Exited"].sum()

    active = total_customers - churned

    churn_rate = (churned / total_customers) * 100

    avg_credit = df["CreditScore"].mean()

    avg_balance = df["Balance"].mean()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "👥 Total Customers",
        f"{total_customers:,}"
    )

    c2.metric(
        "❌ Churned Customers",
        churned
    )

    c3.metric(
        "✅ Active Customers",
        active
    )

    st.markdown("")

    c4, c5, c6 = st.columns(3)

    c4.metric(
        "📉 Churn Rate",
        f"{churn_rate:.2f}%"
    )

    c5.metric(
        "💳 Avg Credit Score",
        f"{avg_credit:.0f}"
    )

    c6.metric(
        "💰 Avg Balance",
        f"${avg_balance:,.0f}"
    )

    st.markdown("---")

    # ==============================
    # CHURN BY COUNTRY
    # ==============================

    st.subheader("🌍 Churn by Geography")

    geo = df.groupby("Geography")["Exited"].sum().reset_index()

    fig = px.bar(

        geo,

        x="Geography",

        y="Exited",

        color="Geography",

        text="Exited",

        title="Customer Churn by Country"

    )

    fig.update_layout(
        height=500,
        title_x=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==============================
    # ACTIVE MEMBERS
    # ==============================

    st.subheader("🟢 Active Members")

    active_df = df["IsActiveMember"].value_counts().reset_index()

    active_df.columns = ["Status", "Customers"]

    active_df["Status"] = active_df["Status"].replace({
        0: "Inactive",
        1: "Active"
    })

    fig = px.pie(

        active_df,

        names="Status",

        values="Customers",

        hole=0.45,

        title="Active vs Inactive Members"

    )

    fig.update_layout(
        height=500,
        title_x=0.5
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.markdown("---")

    # ==============================
    # BUSINESS INSIGHTS
    # ==============================

    st.subheader("📌 Key Business Insights")

    st.success("✔ Random Forest achieved the highest prediction accuracy (86.25%).")

    st.info("✔ Germany has the highest customer churn among all regions.")

    st.info("✔ Active members are significantly less likely to churn.")

    st.info("✔ Customers with higher balances show greater churn risk.")

    st.info("✔ Customers above 45 years are more likely to leave the bank.")

    st.info("✔ Customers with two banking products tend to stay longer.")

    st.markdown("---")

    # ==============================
    # RECOMMENDATIONS
    # ==============================

    st.subheader("💡 Business Recommendations")

    st.write("""

✅ Launch loyalty and rewards programs.

✅ Target high-risk customers with personalized offers.

✅ Improve engagement of inactive customers.

✅ Strengthen customer retention in Germany.

✅ Promote cross-selling of banking products.

✅ Use predictive analytics for proactive retention.

""")

    st.markdown("---")

    st.success("✅ Business Analytics Completed Successfully")

# ==========================================================
# ABOUT PAGE
# ==========================================================

elif page == "ℹ About":

    st.title("ℹ About Project")

    st.markdown("""
# 🏦 Bank Customer Churn Prediction Dashboard

This project predicts whether a customer is likely to leave the bank using Machine Learning and provides business insights through an interactive Streamlit dashboard.

---

## 🎯 Project Objective

To identify customers who are at high risk of churning so that banks can take proactive retention actions.

---

## 📂 Dataset

European Bank Customer Churn Dataset

- Total Records: 10,000
- Features: Customer demographics, financial information, and churn status

---

## 🤖 Machine Learning Models

- Logistic Regression
- Decision Tree
- ⭐ Random Forest (Best Model)

### Best Model Accuracy

**86.25%**

---

## 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Plotly
- Streamlit

---

## 📊 Dashboard Features

- Home Dashboard
- Exploratory Data Analysis (EDA)
- Customer Churn Prediction
- Business Analytics
- Dataset Viewer
- CSV Download

---

## 👨‍💻 Developed By

**Tosif Raza Mansoori**

Unified Mentor Internship Project
""")
    
st.markdown("---")

st.markdown(
"""
<div style='text-align:center'>

Made with ❤️ using

Python • Streamlit • Scikit-Learn • Plotly

<br>

© 2026 Tosif Raza Mansoori

</div>
""",
unsafe_allow_html=True
)