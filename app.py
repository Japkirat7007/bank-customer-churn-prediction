import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Bank Churn Predictor", page_icon="🏦", layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2rem;}
    .risk-card {
        padding: 1.2rem 1.5rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1rem;
    }
    .risk-high {background: linear-gradient(135deg, #c62828, #e53935);}
    .risk-medium {background: linear-gradient(135deg, #ef6c00, #fb8c00);}
    .risk-low {background: linear-gradient(135deg, #2e7d32, #43a047);}
    .risk-card h2 {margin: 0; color: white;}
    .risk-card p {margin: 0.3rem 0 0 0; font-size: 1rem;}
    .factor {
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        background: rgba(128, 128, 128, 0.12);
        border-left: 5px solid #888;
    }
    .factor-bad {border-left-color: #e53935;}
    .factor-good {border-left-color: #43a047;}
    </style>
    """,
    unsafe_allow_html=True,
)


from pathlib import Path

BASE_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    model = joblib.load(BASE_DIR / "churn_model.pkl")
    columns = joblib.load(BASE_DIR / "model_columns.pkl")
    return model, columns


model, columns = load_model()

st.title("🏦 Bank Customer Churn Predictor")
st.caption("Machine learning model (Random Forest) that estimates the chance a customer will leave the bank.")

st.sidebar.header("Customer Details")
credit_score = st.sidebar.slider("Credit Score", 300, 900, 650)
geography = st.sidebar.selectbox("Country", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
age = st.sidebar.slider("Age", 18, 92, 40)
tenure = st.sidebar.slider("Tenure (years with bank)", 0, 10, 5)
balance = st.sidebar.number_input("Account Balance", 0.0, 300000.0, 50000.0, step=1000.0)
products = st.sidebar.selectbox("Number of Products", [1, 2, 3, 4], index=1)
has_card = st.sidebar.radio("Has Credit Card", ["Yes", "No"], horizontal=True)
active = st.sidebar.radio("Active Member", ["Yes", "No"], horizontal=True)
salary = st.sidebar.number_input("Estimated Salary", 0.0, 300000.0, 60000.0, step=1000.0)
predict = st.sidebar.button("Predict Churn Risk", type="primary", use_container_width=True)


def build_row():
    return {
        "CreditScore": credit_score,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": products,
        "HasCrCard": int(has_card == "Yes"),
        "IsActiveMember": int(active == "Yes"),
        "EstimatedSalary": salary,
        "ZeroBalance": int(balance == 0),
        "BalanceSalaryRatio": balance / (salary + 1),
        "ProductsPerTenure": products / (tenure + 1),
        "Geography_Germany": int(geography == "Germany"),
        "Geography_Spain": int(geography == "Spain"),
        "Gender_Male": int(gender == "Male"),
    }


def risk_factors():
    items = []
    if 41 <= age <= 60:
        items.append(("bad", f"Age {age} is in the 41-60 group, where about 34-56% of customers left."))
    elif age <= 40:
        items.append(("good", f"Age {age} is in a younger group, where only about 8-12% left."))
    if geography == "Germany":
        items.append(("bad", "Customers in Germany churn about 32%, double France and Spain."))
    if active == "No":
        items.append(("bad", "Inactive members churn about 27%, against 14% for active members."))
    else:
        items.append(("good", "Active member: much lower churn than inactive members."))
    if products >= 3:
        items.append(("bad", f"{products} products: customers with 3-4 products almost always left (83-100%)."))
    elif products == 2:
        items.append(("good", "2 products: the safest group, only about 8% left."))
    else:
        items.append(("bad", "1 product: about 28% of these customers left."))
    if gender == "Female":
        items.append(("bad", "Female customers churn about 25%, against 16% for male customers."))
    if balance == 0:
        items.append(("good", "Zero balance customers leave less often (about 14%)."))
    return items


def make_gauge(prob):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            number={"suffix": "%", "font": {"size": 48}},
            title={"text": "Churn Probability", "font": {"size": 20}},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#263238", "thickness": 0.25},
                "steps": [
                    {"range": [0, 30], "color": "#66bb6a"},
                    {"range": [30, 60], "color": "#ffa726"},
                    {"range": [60, 100], "color": "#ef5350"},
                ],
                "threshold": {"line": {"color": "black", "width": 4}, "thickness": 0.8, "value": prob * 100},
            },
        )
    )
    fig.update_layout(height=320, margin=dict(l=20, r=20, t=60, b=10))
    return fig


tab1, tab2, tab3 = st.tabs(["🔮 Prediction", "📊 Data Insights", "ℹ️ About"])

with tab1:
    if not predict:
        st.info("Enter the customer details in the left sidebar and click **Predict Churn Risk**.")
    else:
        data = pd.DataFrame([build_row()])[columns]
        prob = float(model.predict_proba(data)[0][1])

        if prob >= 0.6:
            level, css, advice = "HIGH RISK", "risk-high", "Contact this customer now with a personal retention offer."
        elif prob >= 0.3:
            level, css, advice = "MEDIUM RISK", "risk-medium", "Consider a loyalty offer or a check-in call."
        else:
            level, css, advice = "LOW RISK", "risk-low", "Customer is likely to stay. No action needed."

        left, right = st.columns([1, 1])
        with left:
            st.plotly_chart(make_gauge(prob), use_container_width=True)
        with right:
            st.markdown(
                f'<div class="risk-card {css}"><h2>{level}</h2><p>{advice}</p></div>',
                unsafe_allow_html=True,
            )
            m1, m2 = st.columns(2)
            m1.metric("Churn probability", f"{prob:.0%}")
            m2.metric("Retention probability", f"{1 - prob:.0%}")
            st.markdown("**Key factors from the data analysis**")
            for kind, text in risk_factors():
                st.markdown(f'<div class="factor factor-{kind}">{text}</div>', unsafe_allow_html=True)

        st.caption("Risk levels: Low below 30%, Medium 30-60%, High above 60%.")

with tab2:
    st.subheader("What the dataset shows")
    st.caption("Churn rate (%) in the training data of 10,000 bank customers. Overall churn: 20.4%.")

    age_df = pd.DataFrame(
        {"Age group": ["18-30", "31-40", "41-50", "51-60", "60+"], "Churn %": [7.5, 12.1, 34.0, 56.2, 24.8]}
    )
    geo_df = pd.DataFrame({"Country": ["France", "Germany", "Spain"], "Churn %": [16.2, 32.4, 16.7]})
    prod_df = pd.DataFrame({"Products": ["1", "2", "3", "4"], "Churn %": [27.7, 7.6, 82.7, 100.0]})
    act_df = pd.DataFrame({"Member type": ["Active", "Inactive"], "Churn %": [14.3, 26.9]})

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(age_df, x="Age group", y="Churn %", title="Churn by age group", text="Churn %",
                     color="Churn %", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.bar(prod_df, x="Products", y="Churn %", title="Churn by number of products", text="Churn %",
                     color="Churn %", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig = px.bar(geo_df, x="Country", y="Churn %", title="Churn by country", text="Churn %",
                     color="Churn %", color_continuous_scale="Reds")
        fig.update_layout(coloraxis_showscale=False, height=350)
        st.plotly_chart(fig, use_container_width=True)
    with c4:
        fig = px.pie(act_df, names="Member type", values="Churn %", title="Churn rate: active vs inactive",
                     hole=0.5, color_discrete_sequence=["#43a047", "#e53935"])
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("About this project")
    st.markdown(
        """
        **Goal:** predict which bank customers are likely to leave, so the bank can focus retention offers on them.

        **Data:** public Churn Modelling dataset, 10,000 customers, about 20% churned.

        **Method:**
        - Feature engineering (balance-to-salary ratio, products per tenure year, zero-balance flag)
        - Stratified 80/20 train-test split
        - Models compared: Logistic Regression and Random Forest
        - Random Forest with class weights and a 0.30 decision threshold chosen for higher recall

        **Result:** ROC-AUC of about 0.85, catching about 66% of customers who actually left.

        **Tech stack:** Python, pandas, scikit-learn, SHAP, Plotly, Streamlit.
        """
    )
