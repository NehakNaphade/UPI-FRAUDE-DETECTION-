
import os
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="UPI Sentinel | Fraud Intelligence",
    page_icon="🛡️",
    layout="wide"
)

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(145deg, #0b1020, #111a30);
    color: #f1f5f9;
}
[data-testid="stSidebar"] {
    background: #0b1220;
}
h1, h2, h3 {
    color: #e2e8f0 !important;
}
[data-testid="stMetric"] {
    background: #17243b;
    border: 1px solid #2b3b58;
    padding: 18px;
    border-radius: 14px;
}
div.stButton > button {
    background: #635bff;
    color: white;
    border: 0;
    border-radius: 9px;
    min-height: 44px;
    font-weight: 600;
}
div.stButton > button:hover {
    background: #817aff;
    color: white;
}
</style>
""", unsafe_allow_html=True)


# ---------- LOAD DATA ----------
@st.cache_data
def load_data():
    return pd.read_csv("upi_transactions.csv")


@st.cache_resource
def load_model():
    return joblib.load("models/upi_fraud_model.joblib")


st.title("🛡️ UPI SENTINEL")
st.caption("AI-ASSISTED PAYMENT FRAUD INTELLIGENCE | MACHINE LEARNING")

if not os.path.exists("upi_transactions.csv"):
    st.error("Dataset not found. Run: python generate_data.py")
    st.stop()

if not os.path.exists("models/upi_fraud_model.joblib"):
    st.error("Model not found. Run: python train_model.py")
    st.stop()

df = load_data()
saved = load_model()
model = saved["model"]
features = saved["features"]

# Predictions for dashboard analytics
df["fraud_probability"] = model.predict_proba(df[features])[:, 1]
df["predicted_fraud"] = model.predict(df[features])

# ---------- NAVIGATION ----------
page = st.sidebar.radio(
    "NAVIGATION",
    [
        "📊 Command Center",
        "🔍 Transaction Scanner",
        "🤖 AI Investigation Agent",
        "📁 Transaction Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.success("ML Model Loaded")
st.sidebar.caption("Random Forest Classifier")
st.sidebar.caption("Demo dataset — synthetic data")


# ---------- COMMAND CENTER ----------
if page == "📊 Command Center":
    st.subheader("Fraud Intelligence Overview")
    st.caption("Analytics generated from the demonstration dataset")

    total = len(df)
    fraud_count = int(df["predicted_fraud"].sum())
    fraud_rate = fraud_count / total * 100
    avg_amount = df["amount"].mean()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{total:,}")
    c2.metric("Model Fraud Flags", f"{fraud_count:,}")
    c3.metric("Flagged Rate", f"{fraud_rate:.2f}%")
    c4.metric("Avg. Transaction", f"₹{avg_amount:,.0f}")

    st.markdown("### 📈 Transaction Intelligence")

    left, right = st.columns(2)

    with left:
        labels = df["predicted_fraud"].map({
            0: "Legitimate prediction",
            1: "Fraud prediction"
        })
        pie_df = labels.value_counts().rename_axis(
            "Prediction"
        ).reset_index(name="Count")

        fig = px.pie(
            pie_df,
            names="Prediction",
            values="Count",
            hole=0.65,
            title="Predicted Transaction Distribution",
            color="Prediction",
            color_discrete_map={
                "Legitimate prediction": "#22c55e",
                "Fraud prediction": "#f43f5e"
            }
        )
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        bins = df.copy()
        bins["Amount Range"] = pd.cut(
            bins["amount"],
            bins=[0, 500, 2000, 10000, 50000, 200001],
            labels=[
                "₹0–500", "₹500–2K", "₹2K–10K",
                "₹10K–50K", "₹50K+"
            ]
        )
        amount_chart = bins.groupby(
            ["Amount Range", "predicted_fraud"],
            observed=False
        ).size().reset_index(name="Transactions")

        amount_chart["Prediction"] = amount_chart[
            "predicted_fraud"
        ].map({0: "Legitimate", 1: "Fraud flagged"})

        fig2 = px.bar(
            amount_chart,
            x="Amount Range",
            y="Transactions",
            color="Prediction",
            barmode="group",
            title="Predictions by Transaction Amount",
            color_discrete_map={
                "Legitimate": "#3b82f6",
                "Fraud flagged": "#f43f5e"
            }
        )
        fig2.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🚨 Highest-Risk Transactions")
    risky = df.nlargest(10, "fraud_probability")[
        features + ["fraud_probability", "predicted_fraud"]
    ].copy()
    risky["fraud_probability"] = (
        risky["fraud_probability"] * 100
    ).round(1).astype(str) + "%"

    st.dataframe(risky, use_container_width=True, hide_index=True)


# ---------- TRANSACTION SCANNER ----------
elif page == "🔍 Transaction Scanner":
    st.subheader("🔎 Transaction Risk Scanner")
    st.write("Enter transaction details to evaluate with the ML model.")

    with st.form("transaction_form"):
        amount = st.number_input(
            "Transaction amount (₹)",
            min_value=1.0, max_value=200000.0, value=1500.0
        )
        hour = st.slider("Transaction hour", 0, 23, 14)
        frequency = st.number_input(
            "Transactions in last hour",
            min_value=0, max_value=100, value=1
        )
        new_payee = st.selectbox(
            "New payee?", ["No", "Yes"]
        )
        distance = st.number_input(
            "Unusual location distance (km)",
            min_value=0.0, max_value=1000.0, value=2.0
        )
        submitted = st.form_submit_button(
            "⚡ Analyze Transaction"
        )

    if submitted:
        tx = pd.DataFrame([{
            "amount": amount,
            "hour": hour,
            "transactions_last_1h": frequency,
            "new_payee": int(new_payee == "Yes"),
            "distance_km": distance
        }])[features]

        probability = float(model.predict_proba(tx)[0][1])
        prediction = int(model.predict(tx)[0])

        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            number={"suffix": "%"},
            title={"text": "Estimated Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#a78bfa"},
                "steps": [
                    {"range": [0, 30], "color": "#14532d"},
                    {"range": [30, 60], "color": "#854d0e"},
                    {"range": [60, 100], "color": "#7f1d1d"}
                ]
            }
        ))
        gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            font_color="#e2e8f0"
        )
        st.plotly_chart(gauge, use_container_width=True)

        if prediction == 1:
            st.error("⚠️ MODEL RESULT: POTENTIAL FRAUD")
        else:
            st.success("✅ MODEL RESULT: LEGITIMATE CLASSIFICATION")

        st.info(
            "This is a synthetic-data demonstration. "
            "A prediction is not proof of fraud."
        )


# ---------- LOCAL AI-STYLE INVESTIGATION AGENT ----------
elif page == "🤖 AI Investigation Agent":
    st.subheader("🤖 Fraud Investigation Agent")
    st.write(
        "The local agent combines the trained model prediction "
        "with transparent risk indicators and recommended checks."
    )

    row_id = st.number_input(
        "Select dataset row number",
        min_value=0,
        max_value=len(df) - 1,
        value=0,
        step=1
    )

    if st.button("🧠 Investigate Transaction"):
        tx = df.iloc[[int(row_id)]]
        x = tx[features]
        prob = float(model.predict_proba(x)[0][1])
        pred = int(model.predict(x)[0])

        findings = []
        recommendations = []

        if tx["amount"].iloc[0] >= 10000:
            findings.append("Transaction amount is relatively high.")
        if tx["hour"].iloc[0] <= 5:
            findings.append("Transaction occurred during overnight hours.")
        if tx["transactions_last_1h"].iloc[0] >= 5:
            findings.append("High transaction frequency in one hour.")
        if tx["new_payee"].iloc[0] == 1:
            findings.append("Payee is marked as new.")
        if tx["distance_km"].iloc[0] >= 20:
            findings.append("Large reported location distance.")

        if not findings:
            findings.append(
                "No risk indicators from the configured checks were triggered."
            )

        recommendations.append(
            "Verify transaction context and customer confirmation."
        )
        if pred == 1:
            recommendations.append(
                "Refer for additional review before making a decision."
            )
        if tx["new_payee"].iloc[0] == 1:
            recommendations.append(
                "Verify the new payee using an approved process."
            )
        if tx["transactions_last_1h"].iloc[0] >= 5:
            recommendations.append(
                "Review recent transaction frequency."
            )

        st.metric("Model Risk Probability", f"{prob:.1%}")
        st.write(
            "**Model classification:**",
            "Potential Fraud" if pred == 1 else "Legitimate"
        )

        st.markdown("### 🔍 Investigation Findings")
        for finding in findings:
            st.markdown(f"- {finding}")

        st.markdown("### 🛠️ Recommended Checks")
        for item in dict.fromkeys(recommendations):
            st.markdown(f"- {item}")

        st.caption(
            "Agent type: local rule-based investigation assistant. "
            "It does not call an external generative AI model. "
            "Findings are indicators, not proof of fraud."
        )


# ---------- TRANSACTION EXPLORER ----------
elif page == "📁 Transaction Explorer":
    st.subheader("📁 Transaction Explorer")

    minimum = st.slider(
        "Minimum predicted fraud probability (%)",
        min_value=0, max_value=100, value=0
    )

    view = df[df["fraud_probability"] >= minimum / 100].copy()
    view["fraud_probability"] = (
        view["fraud_probability"] * 100
    ).round(2)

    st.caption(f"{len(view):,} matching records")
    st.dataframe(view, use_container_width=True, hide_index=True)

    csv = view.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download filtered CSV",
        data=csv,
        file_name="upi_fraud_analysis.csv",
        mime="text/csv"
    )
