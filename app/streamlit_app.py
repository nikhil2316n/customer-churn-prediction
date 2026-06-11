# ============================================================
# PHASE 9: STREAMLIT DEPLOYMENT
# A professional web app for churn prediction
# Run with: streamlit run app/streamlit_app.py
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import os

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #2c3e50;
        text-align: center;
        padding: 1rem 0;
    }
    .metric-card {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        border-left: 4px solid #3498db;
    }
    .risk-high   { color: #e74c3c; font-weight: 700; font-size: 1.5rem; }
    .risk-medium { color: #f39c12; font-weight: 700; font-size: 1.5rem; }
    .risk-low    { color: #2ecc71; font-weight: 700; font-size: 1.5rem; }
    .stAlert { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Load Model & Metadata ──────────────────────────────────
@st.cache_resource
def load_model():
    # Build absolute paths using this file's location
    # This works no matter where you run streamlit from
    base_dir      = os.path.dirname(os.path.abspath(__file__))
    root_dir      = os.path.dirname(base_dir)

    model_path    = os.path.join(root_dir, 'models', 'final_model.pkl')
    scaler_path   = os.path.join(root_dir, 'models', 'scaler_fe.pkl')
    metadata_path = os.path.join(root_dir, 'models', 'deploy_metadata.json')

    model    = joblib.load(model_path)
    scaler   = joblib.load(scaler_path)
    with open(metadata_path) as f:
        metadata = json.load(f)

    return model, scaler, metadata

model, scaler, metadata = load_model()

# ── Helper: Preprocess single input ────────────────────────
def preprocess_input(data: dict) -> pd.DataFrame:
    """
    Takes raw user input from sidebar and transforms it
    into the exact format the model expects.
    Must match the preprocessing done in Phase 4 & 5.
    """
    df = pd.DataFrame([data])

    # ── Feature Engineering (match Phase 5) ────────────────
    # tenure_group
    def tenure_group(t):
        if t <= 12:   return 'New'
        elif t <= 24: return 'Developing'
        elif t <= 48: return 'Established'
        else:         return 'Loyal'

    df['tenure_group'] = df['tenure'].apply(tenure_group)

    # num_services
    service_cols = ['PhoneService','MultipleLines','InternetService',
                    'OnlineSecurity','OnlineBackup','DeviceProtection',
                    'TechSupport','StreamingTV','StreamingMovies']
    df['num_services'] = df[service_cols].apply(
        lambda row: sum(1 for v in row if v == 'Yes'), axis=1
    )

    # avg_monthly_spend_rate
    df['avg_monthly_spend_rate'] = df['TotalCharges'] / (df['tenure'] + 1)

    # charge_increase_flag
    df['charge_increase_flag'] = (
        df['MonthlyCharges'] > df['avg_monthly_spend_rate'] * 1.1
    ).astype(int)

    # is_high_risk
    df['is_high_risk'] = (
        (df['Contract'] == 'Month-to-month') &
        (df['InternetService'] == 'Fiber optic') &
        (df['PaymentMethod'] == 'Electronic check')
    ).astype(int)

    # vulnerability_score
    df['vulnerability_score'] = (
        df['SeniorCitizen'] +
        (df['Partner'] == 'No').astype(int) +
        (df['Dependents'] == 'No').astype(int)
    )

    # ── Binary Encoding ────────────────────────────────────
    binary_map = {'Yes': 1, 'No': 0,
                  'Male': 1, 'Female': 0}
    binary_cols = ['gender','Partner','Dependents','PhoneService',
                   'PaperlessBilling','MultipleLines','OnlineSecurity',
                   'OnlineBackup','DeviceProtection','TechSupport',
                   'StreamingTV','StreamingMovies']
    for col in binary_cols:
        df[col] = df[col].map(binary_map)

    # ── One-Hot Encoding ───────────────────────────────────
    ohe_cols = ['InternetService','Contract','PaymentMethod','tenure_group']
    df = pd.get_dummies(df, columns=ohe_cols)

    # ── Align columns with training data ───────────────────
    train_cols = metadata['features']
    for col in train_cols:
        if col not in df.columns:
            df[col] = 0               # add missing dummy columns
    df = df[train_cols]               # reorder to match training

    # ── Scale numeric features ─────────────────────────────
    numeric_cols = ['tenure','MonthlyCharges','TotalCharges',
                    'num_services','avg_monthly_spend_rate',
                    'vulnerability_score']
    numeric_present = [c for c in numeric_cols if c in df.columns]
    df[numeric_present] = scaler.transform(df[numeric_present])

    return df


# ── Risk Tier ──────────────────────────────────────────────
def get_risk_tier(prob):
    if prob >= 0.65:
        return 'HIGH RISK',   '🔴', 'risk-high'
    elif prob >= 0.35:
        return 'MEDIUM RISK', '🟡', 'risk-medium'
    else:
        return 'LOW RISK',    '🟢', 'risk-low'


# ════════════════════════════════════════════════════════════
# MAIN APP LAYOUT
# ════════════════════════════════════════════════════════════

st.markdown('<p class="main-header">📡 Customer Churn Prediction System</p>',
            unsafe_allow_html=True)
st.markdown("---")

# ── Sidebar — Model Info ───────────────────────────────────
with st.sidebar:
    st.header("📊 Model Information")
    st.success(f"**Model:** {metadata['model_name'].replace('_',' ')}")
    st.info(f"**AUC-ROC:** {metadata['auc_roc']}")
    st.info(f"**F1 Score:** {metadata['f1_score']}")
    st.info(f"**Precision:** {metadata['precision']}")
    st.info(f"**Recall:** {metadata['recall']}")
    st.markdown("---")
    st.caption("Built with XGBoost + SHAP")
    st.caption("IBM Telco Dataset | 7,043 customers")


# ── Two Tabs ───────────────────────────────────────────────
tab1, tab2 = st.tabs(["🔍 Single Prediction", "📁 Batch Prediction"])


# ════════════════════════════════════════════════════════════
# TAB 1 — SINGLE CUSTOMER PREDICTION
# ════════════════════════════════════════════════════════════
with tab1:

    st.subheader("Enter Customer Details")

    # Input form — 3 columns layout
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Demographics**")
        gender         = st.selectbox("Gender", ["Male", "Female"])
        senior         = st.selectbox("Senior Citizen", [0, 1])
        partner        = st.selectbox("Has Partner", ["Yes", "No"])
        dependents     = st.selectbox("Has Dependents", ["Yes", "No"])

    with col2:
        st.markdown("**Account Information**")
        tenure         = st.slider("Tenure (months)", 0, 72, 12)
        contract       = st.selectbox("Contract Type",
                            ["Month-to-month", "One year", "Two year"])
        payment        = st.selectbox("Payment Method",
                            ["Electronic check", "Mailed check",
                             "Bank transfer (automatic)",
                             "Credit card (automatic)"])
        paperless      = st.selectbox("Paperless Billing", ["Yes", "No"])
        monthly        = st.number_input("Monthly Charges ($)",
                            min_value=0.0, max_value=200.0, value=65.0)
        total          = st.number_input("Total Charges ($)",
                            min_value=0.0, max_value=10000.0,
                            value=float(tenure * monthly))

    with col3:
        st.markdown("**Services**")
        phone          = st.selectbox("Phone Service", ["Yes", "No"])
        multi_lines    = st.selectbox("Multiple Lines",
                            ["Yes", "No", "No phone service"])
        internet       = st.selectbox("Internet Service",
                            ["Fiber optic", "DSL", "No"])
        online_sec     = st.selectbox("Online Security",
                            ["Yes", "No", "No internet service"])
        online_bkp     = st.selectbox("Online Backup",
                            ["Yes", "No", "No internet service"])
        device_prot    = st.selectbox("Device Protection",
                            ["Yes", "No", "No internet service"])
        tech_sup       = st.selectbox("Tech Support",
                            ["Yes", "No", "No internet service"])
        streaming_tv   = st.selectbox("Streaming TV",
                            ["Yes", "No", "No internet service"])
        streaming_mov  = st.selectbox("Streaming Movies",
                            ["Yes", "No", "No internet service"])

    st.markdown("---")

    # ── Predict Button ─────────────────────────────────────
    if st.button("🔮 Predict Churn", type="primary", use_container_width=True):

        # Build input dict
        input_data = {
            'gender': gender, 'SeniorCitizen': senior,
            'Partner': partner, 'Dependents': dependents,
            'tenure': tenure, 'PhoneService': phone,
            'MultipleLines': multi_lines, 'InternetService': internet,
            'OnlineSecurity': online_sec, 'OnlineBackup': online_bkp,
            'DeviceProtection': device_prot, 'TechSupport': tech_sup,
            'StreamingTV': streaming_tv, 'StreamingMovies': streaming_mov,
            'Contract': contract, 'PaperlessBilling': paperless,
            'PaymentMethod': payment,
            'MonthlyCharges': monthly, 'TotalCharges': total
        }

        # Preprocess
        processed = preprocess_input(input_data)

        # Predict
        threshold = metadata.get('threshold', 0.4)
        prob      = model.predict_proba(processed)[0][1]
        prediction = int(prob >= threshold)

        # Risk tier
        risk_label, risk_icon, risk_class = get_risk_tier(prob)

        # ── Results Display ────────────────────────────────
        st.markdown("---")
        st.subheader("Prediction Results")

        res_col1, res_col2, res_col3 = st.columns(3)

        with res_col1:
            st.metric(
                label="Churn Probability",
                value=f"{prob*100:.1f}%"
            )
        with res_col2:
            st.metric(
                label="Prediction",
                value="Will Churn" if prediction == 1 else "Will Stay"
            )
        with res_col3:
            st.metric(
                label="Risk Tier",
                value=f"{risk_icon} {risk_label}"
            )

        # ── Probability Gauge ──────────────────────────────
        fig, ax = plt.subplots(figsize=(8, 1.5))
        color = '#e74c3c' if prob >= 0.65 else \
                '#f39c12' if prob >= 0.35 else '#2ecc71'
        ax.barh(['Churn Risk'], [prob],
                color=color, height=0.4)
        ax.barh(['Churn Risk'], [1 - prob],
                left=[prob], color='#ecf0f1', height=0.4)
        ax.axvline(x=threshold, color='gray',
                   linestyle='--', linewidth=1.5,
                   label=f'Threshold = {threshold}')
        ax.set_xlim(0, 1)
        ax.set_xlabel('Probability')
        ax.set_title('Churn Probability Gauge', fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # ── Business Recommendation ────────────────────────
        st.markdown("---")
        st.subheader("💼 Business Recommendation")

        if prediction == 1:
            st.error(f"""
            **⚠️ Action Required — {risk_label}**

            This customer has a **{prob*100:.1f}% probability** of churning.

            **Recommended Actions:**
            - 📧 Send a personalized retention offer within 48 hours
            - 💰 Offer a discount or plan upgrade
            - 📞 Schedule a customer success call
            - 🎁 Provide loyalty rewards or free service add-ons

            **Estimated Revenue at Risk:** ${monthly * 12:.0f}/year
            """)
        else:
            st.success(f"""
            **✅ Low Churn Risk**

            This customer has only a **{prob*100:.1f}% probability** of churning.

            **Recommended Actions:**
            - Continue standard engagement
            - Consider upsell opportunities (additional services)
            - Monitor next billing cycle for changes
            """)


# ════════════════════════════════════════════════════════════
# TAB 2 — BATCH PREDICTION
# ════════════════════════════════════════════════════════════
with tab2:

    st.subheader("Upload CSV for Batch Predictions")
    st.info("""
    Upload a CSV file with the same columns as the IBM Telco dataset.
    The app will predict churn probability for every customer
    and return a downloadable results file.
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=['csv']
    )

    if uploaded_file is not None:

        batch_df = pd.read_csv(uploaded_file)
        st.write(f"Loaded {len(batch_df)} customers")
        st.dataframe(batch_df.head(5))

        if st.button("🔮 Predict All", type="primary"):

            with st.spinner("Running predictions..."):

                results = []
                errors  = 0

                for _, row in batch_df.iterrows():
                    try:
                        processed = preprocess_input(row.to_dict())
                        prob      = model.predict_proba(processed)[0][1]
                        tier, _, _= get_risk_tier(prob)
                        results.append({
                            'churn_probability' : round(prob, 4),
                            'prediction'        : int(prob >= 0.4),
                            'risk_tier'         : tier
                        })
                    except Exception:
                        errors += 1
                        results.append({
                            'churn_probability' : None,
                            'prediction'        : None,
                            'risk_tier'         : 'ERROR'
                        })

                results_df = pd.concat(
                    [batch_df.reset_index(drop=True),
                     pd.DataFrame(results)],
                    axis=1
                )

            # ── Summary stats ──────────────────────────────
            st.success(f"Predictions complete! {len(results_df)} customers processed.")

            col1, col2, col3 = st.columns(3)
            col1.metric("High Risk",
                        len(results_df[results_df['risk_tier']=='HIGH RISK']))
            col2.metric("Medium Risk",
                        len(results_df[results_df['risk_tier']=='MEDIUM RISK']))
            col3.metric("Low Risk",
                        len(results_df[results_df['risk_tier']=='LOW RISK']))

            # ── Show results ───────────────────────────────
            st.dataframe(results_df[[
                'churn_probability','prediction','risk_tier'
            ]].head(20))

            # ── Download button ────────────────────────────
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="⬇️ Download Results CSV",
                data=csv,
                file_name='churn_predictions.csv',
                mime='text/csv',
                use_container_width=True
            )