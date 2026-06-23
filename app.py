# =============================================================
# app.py  — Loan Default Predictor Web App
# =============================================================
# Run locally with:   streamlit run app.py
# (you must run train.py FIRST so model.pkl exists)
#
# This file IS the entire web app.  Streamlit turns Python code
# into a real web page — no HTML or JavaScript needed.
# =============================================================

# ---- IMPORTS -----------------------------------------------
import os
import joblib               # loads model.pkl from disk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')       # prevents pop-up windows; required inside Streamlit
import matplotlib.pyplot as plt
import streamlit as st      # the web-app framework
import shap                 # explains the model's decisions

# ============================================================
# PAGE CONFIG  (must be the VERY FIRST Streamlit call)
# ============================================================
st.set_page_config(
    page_title="Loan Default Predictor",
    page_icon="🏦",
    layout="centered",
)

# ============================================================
# LOAD MODEL  (cached — only runs once even if the user clicks many times)
# ============================================================
@st.cache_resource   # Streamlit stores the result so it is not reloaded on every interaction
def load_artifact():
    """
    Load the trained model bundle saved by train.py.
    Returns the full artifact dictionary, or None if the file is missing.
    """
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")


artifact = load_artifact()

# If train.py has not been run yet, show a helpful error and stop.
if artifact is None:
    st.error(
        "**model.pkl not found.**\n\n"
        "Open a terminal in `C:\\loan-default-app` and run:\n\n"
        "```\npython train.py\n```\n\n"
        "Then refresh this page."
    )
    st.stop()   # nothing below this line runs until the file exists

# Unpack the parts we saved in train.py
model            = artifact["model"]            # the trained RandomForestClassifier
encoder          = artifact["encoder"]          # OrdinalEncoder for text columns
feature_cols     = artifact["feature_cols"]     # the 11 column names in training order
categorical_cols = artifact["categorical_cols"] # the 4 text columns
encoder_cats     = artifact["encoder_categories"]  # {col: [list of valid values]}
nice_names       = artifact["nice_names"]       # {col: "Human Readable Label"}
roc_auc          = artifact.get("roc_auc", None)


# Build a SHAP explainer once and cache it — it is slow to create
@st.cache_resource
def build_explainer(_model):
    """
    Create a TreeExplainer for our Random Forest.
    The underscore on _model tells Streamlit: do not try to hash this object.
    """
    return shap.TreeExplainer(_model)

explainer = build_explainer(model)


# ============================================================
# HEADER
# ============================================================
st.title("🏦 Loan Default Predictor")
st.markdown(
    "Enter the applicant's details below and click **Predict** to see "
    "the estimated probability that they will default on their loan, "
    "plus an explanation of WHY."
)

# Show model performance in a collapsible section
with st.expander("Model performance (held-out test set)", expanded=False):
    st.markdown(
        f"**ROC-AUC: {roc_auc:.4f}**  "
        f"(0.50 = random guessing · 1.00 = perfect · above 0.75 is good)"
        if roc_auc else "ROC-AUC: not available"
    )
    st.caption(
        "Random Forest · 200 trees · class_weight='balanced' · "
        "Trained on credit_risk_dataset (or synthetic equivalent if CSV not provided)"
    )

st.divider()


# ============================================================
# INPUT FORM
# ============================================================
# st.columns([...]) splits the page into side-by-side panels.
st.subheader("Applicant & Loan Details")
col_left, col_right = st.columns(2)

# ---- LEFT COLUMN — person-level features ----
with col_left:
    st.markdown("**About the applicant**")

    person_age = st.number_input(
        "Age (years)",
        min_value=18, max_value=80, value=35, step=1,
        help="Applicant's current age"
    )
    person_income = st.number_input(
        "Annual Income ($)",
        min_value=5_000, max_value=500_000, value=55_000, step=1_000,
        help="Gross annual income in US dollars"
    )
    person_emp_length = st.number_input(
        "Employment Length (years)",
        min_value=0.0, max_value=41.0, value=5.0, step=0.5,
        help="How long they have been continuously employed at their current job"
    )
    person_home_ownership = st.selectbox(
        "Home Ownership",
        options=encoder_cats["person_home_ownership"],
        help="Current housing situation"
    )
    cb_person_default_on_file = st.selectbox(
        "Past Default on File?",
        options=encoder_cats["cb_person_default_on_file"],
        format_func=lambda x: "Yes — has defaulted before" if x == "Y" else "No — clean history",
        help="Has the applicant ever defaulted on a previous loan?"
    )
    cb_person_cred_hist_length = st.number_input(
        "Credit History Length (years)",
        min_value=2, max_value=30, value=8, step=1,
        help="How many years of credit history exist for this person"
    )

# ---- RIGHT COLUMN — loan-level features ----
with col_right:
    st.markdown("**About the loan**")

    loan_amnt = st.number_input(
        "Loan Amount ($)",
        min_value=500, max_value=35_000, value=10_000, step=500,
        help="Dollar amount being requested"
    )
    loan_int_rate = st.slider(
        "Interest Rate (%)",
        min_value=5.0, max_value=35.0, value=12.0, step=0.25,
        help="Annual interest rate on the loan"
    )
    loan_grade = st.selectbox(
        "Loan Grade  (A = best credit · G = worst)",
        options=encoder_cats["loan_grade"],
        help="Credit grade assigned to this loan application"
    )
    loan_intent = st.selectbox(
        "Loan Purpose",
        options=encoder_cats["loan_intent"],
        help="What the applicant intends to spend the money on"
    )

    # loan_percent_income is calculated automatically — the user does not type it in
    loan_percent_income = round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    st.metric(
        label="Loan as % of Annual Income  (auto-calculated)",
        value=f"{loan_percent_income:.1%}",
        help="Higher ratios are riskier because the loan is a large chunk of yearly income"
    )

st.divider()


# ============================================================
# PREDICT BUTTON
# ============================================================
# Everything below only runs when the user clicks this button.
predict_clicked = st.button(
    "🔍  Predict Default Risk",
    type="primary",           # renders as the prominent blue button
    use_container_width=True  # stretches the button to full page width
)

if predict_clicked:

    # ---- Step A: Build a single-row DataFrame ----
    # We MUST arrange columns in the EXACT ORDER used during training.
    # feature_cols (loaded from model.pkl) is that order.
    raw_row = {
        "person_age"                : [person_age],
        "person_income"             : [person_income],
        "person_emp_length"         : [person_emp_length],
        "loan_amnt"                 : [loan_amnt],
        "loan_int_rate"             : [loan_int_rate],
        "loan_percent_income"       : [loan_percent_income],
        "cb_person_cred_hist_length": [cb_person_cred_hist_length],
        "person_home_ownership"     : [person_home_ownership],
        "loan_intent"               : [loan_intent],
        "loan_grade"                : [loan_grade],
        "cb_person_default_on_file" : [cb_person_default_on_file],
    }
    X_df = pd.DataFrame(raw_row)[feature_cols]   # reorder to match training

    # ---- Step B: Encode the text columns as numbers ----
    # The model was trained on numbers, not strings like "RENT".
    # We apply the same encoder that was used during training.
    X_enc = X_df.copy()
    X_enc[categorical_cols] = encoder.transform(X_enc[categorical_cols])

    # ---- Step C: Predict ----
    # predict_proba returns [[P(no default), P(default)]] — we take column 1.
    default_prob = float(model.predict_proba(X_enc)[0][1])

    # ============================================================
    # SHOW RESULT
    # ============================================================
    st.subheader("Prediction Result")

    # Pick colour and label based on risk level
    if default_prob >= 0.65:
        emoji, risk_label, bg_color = "🔴", "HIGH RISK",    "#ffe0e0"
    elif default_prob >= 0.40:
        emoji, risk_label, bg_color = "🟡", "MEDIUM RISK",  "#fff8dc"
    else:
        emoji, risk_label, bg_color = "🟢", "LOW RISK",     "#e0ffe0"

    # Coloured banner (unsafe_allow_html lets us use inline HTML for styling)
    st.markdown(
        f"""
        <div style="
            background-color:{bg_color};
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        ">
            <h2>{emoji}&nbsp; {risk_label}</h2>
            <h3>Default Probability: <strong>{default_prob:.1%}</strong></h3>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Progress bar as a visual risk gauge (0.0 to 1.0)
    st.progress(
        default_prob,
        text=f"Risk score: {default_prob:.1%}  "
             f"(0% = certainly repays · 100% = certainly defaults)"
    )

    st.divider()

    # ============================================================
    # SHAP EXPLANATION  (the "why" behind the prediction)
    # ============================================================
    st.subheader("Why this prediction?  (SHAP Explanation)")
    st.markdown(
        "**Red bars** push the risk score **higher** (toward default).  \n"
        "**Blue bars** push the risk score **lower** (away from default).  \n"
        "The length of each bar shows how much that feature mattered *for this specific applicant*."
    )

    with st.spinner("Computing SHAP values..."):
        try:
            # Pass as a numpy array (most compatible with all SHAP versions)
            X_arr   = X_enc.values
            sv      = explainer.shap_values(X_arr)

            # SHAP returns different formats across versions — handle both.
            # For a binary classifier we want SHAP values for class 1 (default).
            if isinstance(sv, list):
                # Older SHAP API: list [class0_array, class1_array]
                contributions = np.asarray(sv[1])[0]       # [0] = first (only) sample
            else:
                sv = np.asarray(sv)
                if sv.ndim == 3:
                    # Newer API shape: (samples, features, classes)
                    contributions = sv[0, :, 1]
                else:
                    # Two-class squeeze: (samples, features)
                    contributions = sv[0]

            # Map column names → human-readable chart labels
            labels = [nice_names.get(c, c) for c in feature_cols]

            # Sort features by absolute impact so the most influential sits at the top
            order          = np.argsort(np.abs(contributions))
            sv_sorted      = contributions[order]
            labels_sorted  = np.array(labels)[order]

            # Draw the horizontal bar chart
            fig, ax = plt.subplots(figsize=(8, 5))
            bar_colors = ["#d73027" if v > 0 else "#4575b4" for v in sv_sorted]
            ax.barh(labels_sorted, sv_sorted, color=bar_colors)
            ax.axvline(x=0, color="black", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP value  (positive = pushes toward default)")
            ax.set_title("Feature Impact on This Applicant's Score")
            # Give the bars a little breathing room at each edge
            x_abs = max(abs(sv_sorted.max()), abs(sv_sorted.min())) * 1.3 or 0.1
            ax.set_xlim(-x_abs, x_abs)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)   # free memory — important in long-running Streamlit apps

        except Exception as err:
            st.warning(f"SHAP explanation unavailable: {err}")

    # ============================================================
    # SUMMARY METRICS  (a quick-reference card at the bottom)
    # ============================================================
    st.divider()
    st.subheader("Applicant Snapshot")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Loan Amount",   f"${loan_amnt:,}")
        st.metric("Loan Grade",    loan_grade)
    with c2:
        st.metric("Interest Rate", f"{loan_int_rate}%")
        st.metric("Past Default",  "Yes ⚠️" if cb_person_default_on_file == "Y" else "No ✓")
    with c3:
        st.metric("Annual Income", f"${person_income:,}")
        st.metric("Loan Burden",   f"{loan_percent_income:.1%} of income")

# ============================================================
# FOOTER
# ============================================================
st.divider()
st.caption(
    "Random Forest · SHAP explainability · "
    "Educational / portfolio project — not real financial advice."
)
