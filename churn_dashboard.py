import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# ─── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📉",
    layout="wide"
)

@st.cache_data
def load_and_train():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

    # Clean
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop('customerID', axis=1, inplace=True)

    # Encode binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService',
                   'PaperlessBilling', 'Churn']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    df_model = pd.get_dummies(df, columns=[
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ], drop_first=True)

    # Train model
    X = df_model.drop('Churn', axis=1)
    y = df_model['Churn']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    df_model['ChurnProbability'] = model.predict_proba(X)[:, 1]
    df_model['RiskLevel'] = pd.cut(
        df_model['ChurnProbability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

    original_contract = pd.get_dummies(df[['Contract']], drop_first=False)
    contract_map = df[['Contract']] if 'Contract' in df.columns else None

    return df, df_model, model, X.columns.tolist(), auc


df_raw, df_model, model, feature_cols, auc = load_and_train()

st.sidebar.image("https://img.icons8.com/color/96/combo-chart.png", width=60)
st.sidebar.title("Filters")

contract_options = df_raw['Contract'].unique().tolist()
selected_contracts = st.sidebar.multiselect(
    "Contract Type",
    options=contract_options,
    default=contract_options
)

tenure_range = st.sidebar.slider(
    "Tenure (months)",
    min_value=int(df_raw['tenure'].min()),
    max_value=int(df_raw['tenure'].max()),
    value=(0, int(df_raw['tenure'].max()))
)

risk_filter = st.sidebar.multiselect(
    "Risk Level",
    options=['Low', 'Medium', 'High'],
    default=['Low', 'Medium', 'High']
)

mask = (
    df_raw['Contract'].isin(selected_contracts) &
    df_raw['tenure'].between(tenure_range[0], tenure_range[1])
)
df_filtered = df_raw[mask].copy()
df_model_filtered = df_model[mask].copy()
df_model_filtered = df_model_filtered[
    df_model_filtered['RiskLevel'].isin(risk_filter)
]

# ─── Header ────────────────────────────────────────────────────────────────────
st.title("📉 Customer Churn Analysis Dashboard")
st.caption("Telco dataset · Random Forest model")

# ─── KPI Cards ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)

total = len(df_filtered)
churned = int(df_filtered['Churn'].map({'Yes': 1, 'No': 0}).sum()) \
    if df_filtered['Churn'].dtype == object \
    else int(df_filtered['Churn'].sum())
churn_rate = churned / total * 100 if total > 0 else 0
avg_monthly = df_filtered['MonthlyCharges'].mean()
high_risk = int((df_model_filtered['RiskLevel'] == 'High').sum())

k1.metric("Total Customers", f"{total:,}")
k2.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{churn_rate - 26.5:.1f}% vs baseline")
k3.metric("Avg Monthly Charge", f"€{avg_monthly:.0f}")
k4.metric("High Risk Customers", f"{high_risk:,}")

st.divider()

# ─── Row 1: Churn by Contract + Churn by Tenure ───────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Rate by Contract Type")
    churn_col = df_raw['Churn'].map({'Yes': 1, 'No': 0}) \
        if df_raw['Churn'].dtype == object else df_raw['Churn']
    contract_churn = df_raw.copy()
    contract_churn['ChurnNum'] = churn_col
    contract_summary = contract_churn.groupby('Contract')['ChurnNum'].mean().reset_index()
    contract_summary.columns = ['Contract', 'ChurnRate']
    contract_summary['ChurnRate'] = contract_summary['ChurnRate'] * 100

    fig1 = px.bar(
        contract_summary,
        x='Contract', y='ChurnRate',
        color='ChurnRate',
        color_continuous_scale=['#2ecc71', '#f39c12', '#e74c3c'],
        text=contract_summary['ChurnRate'].apply(lambda x: f"{x:.1f}%"),
        labels={'ChurnRate': 'Churn Rate (%)'}
    )
    fig1.update_traces(textposition='outside')
    fig1.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig1, width='stretch')
    st.caption("💡 Month-to-month customers churn at 3x the rate of annual contracts")

with col2:
    st.subheader("Churn Rate by Tenure Group")
    df_tenure = df_raw.copy()
    df_tenure['ChurnNum'] = df_tenure['Churn'].map({'Yes': 1, 'No': 0}) \
        if df_tenure['Churn'].dtype == object else df_tenure['Churn']
    df_tenure['TenureGroup'] = pd.cut(
        df_tenure['tenure'],
        bins=[0, 12, 24, 48, 72],
        labels=['0–12 mo', '13–24 mo', '25–48 mo', '49–72 mo']
    )
    tenure_summary = df_tenure.groupby('TenureGroup', observed=True)['ChurnNum'].mean().reset_index()
    tenure_summary['ChurnRate'] = tenure_summary['ChurnNum'] * 100

    fig2 = px.bar(
        tenure_summary,
        x='TenureGroup', y='ChurnRate',
        color='ChurnRate',
        color_continuous_scale=['#e74c3c', '#f39c12', '#2ecc71', '#27ae60'],
        text=tenure_summary['ChurnRate'].apply(lambda x: f"{x:.1f}%"),
        labels={'ChurnRate': 'Churn Rate (%)', 'TenureGroup': 'Tenure Group'}
    )
    fig2.update_traces(textposition='outside')
    fig2.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig2, width='stretch')
    st.caption("💡 First 12 months are the highest risk window for churn")

col3, col4 = st.columns(2)

with col3:
    st.subheader("Top 10 Churn Drivers")
    importances = pd.Series(model.feature_importances_, index=feature_cols)
    top10 = importances.sort_values(ascending=True).tail(10).reset_index()
    top10.columns = ['Feature', 'Importance']
    top10['Feature'] = top10['Feature'].str.replace('_', ' ').str.title()

    fig3 = px.bar(
        top10, x='Importance', y='Feature',
        orientation='h',
        color='Importance',
        color_continuous_scale=['#3498db', '#9b59b6'],
        labels={'Importance': 'Feature Importance'}
    )
    fig3.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20)
    )
    st.plotly_chart(fig3, width='stretch')
    st.caption(f"💡 Model AUC: {auc:.3f} — tenure and monthly charges are top predictors")

with col4:
    st.subheader("Customer Risk Distribution")
    risk_counts = df_model['RiskLevel'].value_counts().reset_index()
    risk_counts.columns = ['RiskLevel', 'Count']

    fig4 = px.pie(
        risk_counts,
        names='RiskLevel',
        values='Count',
        color='RiskLevel',
        color_discrete_map={
            'Low': '#2ecc71',
            'Medium': '#f39c12',
            'High': '#e74c3c'
        },
        hole=0.45
    )
    fig4.update_layout(
        margin=dict(t=20, b=20),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    st.plotly_chart(fig4, width='stretch')
    st.caption("💡 High risk = churn probability > 60%")

st.divider()

# ─── Risk Score Table ──────────────────────────────────────────────────────────
st.subheader("🎯 Customer Risk Score Table")
st.caption("Filtered by sidebar selections — sorted by churn probability")

display_cols = ['tenure', 'MonthlyCharges', 'TotalCharges',
                'ChurnProbability', 'RiskLevel']

display_df = df_raw[mask].copy().reset_index(drop=True)
display_df['ChurnProbability'] = df_model[mask]['ChurnProbability'].values
display_df['RiskLevel'] = df_model[mask]['RiskLevel'].values
display_df = display_df[display_df['RiskLevel'].isin(risk_filter)]

display_df = display_df[['Contract', 'tenure', 'MonthlyCharges',
                          'TotalCharges', 'ChurnProbability', 'RiskLevel']]
display_df = display_df.sort_values('ChurnProbability', ascending=False)
display_df['ChurnProbability'] = display_df['ChurnProbability'].apply(
    lambda x: f"{x:.1%}")

def color_risk(val):
    colors = {'High': 'background-color: #fde8e8; color: #c0392b',
              'Medium': 'background-color: #fef3cd; color: #856404',
              'Low': 'background-color: #d4edda; color: #155724'}
    return colors.get(val, '')

st.dataframe(
    display_df.head(50).style.map(color_risk, subset=['RiskLevel']),
    width='stretch',
    height=400
)

st.caption("Showing top 50 highest-risk customers. Use filters in the sidebar to segment.")

# ─── Footer ────────────
st.markdown(f"""
**Project:** Customer Churn Analysis  
**Model:** Random Forest  
**AUC:** {auc:.3f}  
**Dataset:** IBM Telco (Kaggle)  
**Built by:** Shreya Sonar
""")
