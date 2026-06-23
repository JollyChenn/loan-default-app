# =============================================================
# app.py  —  CreditScope: Loan Underwriting Dashboard
# =============================================================
# A bank-grade Streamlit dashboard that, for any loan applicant:
#   • Predicts default probability with a Random Forest
#   • Calculates real loan financials (monthly payment, DTI, total cost)
#   • Issues an underwriting decision (Approve / Conditional / Decline)
#   • Recommends risk-adjusted pricing
#   • Runs stress tests (rate shock, income shock)
#   • Explains the decision with SHAP
#
# RUN LOCALLY:    streamlit run app.py
# =============================================================

# ---- IMPORTS -----------------------------------------------
import os
import hashlib                # used to generate a unique application ID
from datetime import datetime # timestamp for the underwriting report
import joblib                 # loads model.pkl from disk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")         # non-interactive backend (required by Streamlit)
import matplotlib.pyplot as plt
import streamlit as st        # the web-app framework
import shap                   # SHAP explanations
import plotly.graph_objects as go   # gauge charts


# ============================================================
# PAGE CONFIG (must be the VERY FIRST Streamlit call)
# ============================================================
st.set_page_config(
    page_title            = "CreditScope — Loan Underwriting Dashboard",
    page_icon             = "🏦",
    layout                = "wide",
    initial_sidebar_state = "expanded",
)


# ============================================================
# CUSTOM CSS — fintech polish on top of the dark theme
# ============================================================
st.markdown(
    """
    <style>
        /* Gradient hero header */
        .hero-header {
            background: linear-gradient(135deg, #0f766e 0%, #10b981 50%, #06b6d4 100%);
            padding: 30px 40px;
            border-radius: 14px;
            margin-bottom: 28px;
            box-shadow: 0 8px 28px rgba(16,185,129,0.18);
        }
        .hero-header h1 { color: white; margin: 0; font-size: 34px; letter-spacing: -0.5px; }
        .hero-header p  { color: rgba(255,255,255,0.92); margin: 8px 0 0; font-size: 15px; }

        /* Metric cards — soft border, rounded */
        div[data-testid="stMetric"] {
            background: #161b2e;
            padding: 16px 20px;
            border-radius: 12px;
            border: 1px solid #1f2742;
        }
        div[data-testid="stMetricValue"] { font-size: 26px !important; font-weight: 600; }

        /* Tighten section padding */
        .block-container { padding-top: 2rem; padding-bottom: 3rem; }

        /* Big decision banner */
        .decision-banner {
            padding: 24px 28px;
            border-radius: 14px;
            margin: 8px 0 18px;
            text-align: center;
        }
        .decision-banner h1 { margin: 0; font-size: 36px; letter-spacing: 0.5px; }
        .decision-banner p  { margin: 6px 0 0; font-size: 15px; opacity: 0.9; }

        .ref-strip {
            background: #161b2e;
            border-left: 4px solid #10b981;
            padding: 12px 18px;
            border-radius: 6px;
            margin-bottom: 20px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            color: #8b96a8;
        }

        /* Sidebar branding */
        .sidebar-brand {
            text-align: center;
            padding: 8px 0 18px;
            border-bottom: 1px solid #1f2742;
            margin-bottom: 16px;
        }
        .sidebar-brand h2 {
            background: linear-gradient(90deg, #10b981, #06b6d4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0; font-size: 26px; letter-spacing: -0.5px;
        }
        .sidebar-brand p { color: #8b96a8; margin: 4px 0 0; font-size: 12px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL (cached — only loads once per session)
# ============================================================
@st.cache_resource
def load_artifact():
    """Load the trained model bundle saved by train.py."""
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")


artifact = load_artifact()

if artifact is None:
    st.error(
        "**model.pkl not found.**\n\n"
        "Open a terminal in this folder and run `python train.py`, then refresh."
    )
    st.stop()

# Unpack what we saved during training
model            = artifact["model"]
encoder          = artifact["encoder"]
feature_cols     = artifact["feature_cols"]
categorical_cols = artifact["categorical_cols"]
encoder_cats     = artifact["encoder_categories"]
nice_names       = artifact["nice_names"]
roc_auc          = artifact.get("roc_auc", None)


@st.cache_resource
def build_explainer(_model):
    """SHAP TreeExplainer (cached because it's slow to build)."""
    return shap.TreeExplainer(_model)


explainer = build_explainer(model)


# ============================================================
# SIDEBAR — branding + guide (model stats moved to About tab)
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>🏦 CreditScope</h2>
            <p>Loan Underwriting Dashboard</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📖 Quick Guide")
    st.markdown(
        """
        1. Fill in applicant & loan details
        2. Click **Run Risk Analysis**
        3. Review underwriting decision
        4. Check financials & stress tests
        5. Download report (CSV)
        """
    )
    st.divider()

    st.markdown("### 🎯 Risk Tiers")
    st.markdown(
        """
        🟢 **Low**     · &lt; 40 %
        🟡 **Medium**  · 40 – 65 %
        🔴 **High**    · &gt; 65 %
        """
    )
    st.divider()

    st.markdown("### 💡 Decision Logic")
    st.caption(
        "Decisions combine the ML risk score with classic underwriting rules: "
        "debt-to-income, past defaults, and loan grade."
    )
    st.divider()
    st.caption("v2.0 · Educational use only")


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero-header">
        <h1>🏦 CreditScope — Loan Underwriting Dashboard</h1>
        <p>AI-powered default prediction · Financial analysis · Stress testing · SHAP explainability</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABS
# ============================================================
tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["🎯  Underwrite",  "📁  Batch Scoring",  "📊  Model Insights",  "ℹ️  About"]
)


# =============================================================
# HELPER FUNCTIONS — shared across tabs
# =============================================================
def encode_row(df_raw):
    """Apply the saved encoder so text columns become integers the model can use."""
    df_raw = df_raw[feature_cols].copy()
    df_raw[categorical_cols] = encoder.transform(df_raw[categorical_cols])
    return df_raw


def risk_label(prob):
    """Return (emoji, label, color) for a default probability."""
    if prob >= 0.65:
        return "🔴", "HIGH RISK",    "#ef4444"
    if prob >= 0.40:
        return "🟡", "MEDIUM RISK",  "#f59e0b"
    return     "🟢", "LOW RISK",     "#10b981"


def underwriting_decision(prob, default_on_file, percent_income, loan_grade):
    """
    Combine the ML risk score with classic underwriting rules to issue
    a final decision: APPROVE, CONDITIONAL APPROVAL, or DECLINE.
    Returns (verdict_text, decision_color, reasoning_bullets).
    """
    notes = []
    if default_on_file == "Y":
        notes.append("Prior default on file — significant red flag")
    if percent_income > 0.30:
        notes.append(f"Loan is {percent_income:.0%} of annual income — high debt burden")
    if loan_grade in ("E", "F", "G"):
        notes.append(f"Loan grade {loan_grade} indicates weak credit profile")
    if prob >= 0.65:
        notes.append(f"Model default probability {prob:.0%} exceeds high-risk threshold")

    if prob < 0.40 and default_on_file == "N":
        return "✅  APPROVE", "#10b981", notes or [
            "Risk profile within acceptable underwriting parameters",
            "No history of prior default",
            f"Debt-to-income within healthy range ({percent_income:.0%})",
        ]
    if prob < 0.65:
        return "⚠️  CONDITIONAL", "#f59e0b", notes or [
            "Acceptable with modifications to loan terms",
        ]
    return "❌  DECLINE", "#ef4444", notes


def monthly_payment(principal, annual_rate_pct, months=36):
    """
    Standard loan amortization formula:
        M = P * r(1+r)^n / ((1+r)^n - 1)
    Returns the fixed monthly payment for a fully amortizing loan.
    """
    r = (annual_rate_pct / 100) / 12
    if r == 0:
        return principal / months
    return principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)


def dti_ratio(monthly_pmt, annual_income):
    """Debt-to-income = monthly loan payment / monthly income."""
    monthly_income = annual_income / 12
    return monthly_pmt / monthly_income if monthly_income > 0 else 0.0


def dti_tier(dti):
    """Bank-standard DTI tiers."""
    if dti < 0.28:
        return "Healthy",      "#10b981"
    if dti < 0.43:
        return "Cautionary",   "#f59e0b"
    return     "Unaffordable", "#ef4444"


def make_risk_gauge(prob, title="Default Risk"):
    """Plotly half-gauge for risk probability."""
    _, label, color = risk_label(prob)
    fig = go.Figure(
        go.Indicator(
            mode    = "gauge+number",
            value   = prob * 100,
            number  = {"suffix": " %", "font": {"size": 44, "color": "#e6edf3"}},
            title   = {"text": f"<b>{title}</b><br><span style='font-size:14px;color:{color}'>{label}</span>",
                       "font": {"size": 17, "color": "#e6edf3"}},
            gauge   = {
                "axis":        {"range": [0, 100], "tickcolor": "#8b96a8", "tickfont": {"color": "#8b96a8"}},
                "bar":         {"color": color, "thickness": 0.30},
                "bgcolor":     "#161b2e",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 40],   "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [40, 65],  "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3"},
        height=290,
        margin={"l": 20, "r": 20, "t": 60, "b": 10},
    )
    return fig


def make_dti_gauge(dti, title="Debt-to-Income"):
    """Plotly gauge for affordability (DTI)."""
    label, color = dti_tier(dti)
    fig = go.Figure(
        go.Indicator(
            mode    = "gauge+number",
            value   = dti * 100,
            number  = {"suffix": " %", "font": {"size": 44, "color": "#e6edf3"}},
            title   = {"text": f"<b>{title}</b><br><span style='font-size:14px;color:{color}'>{label}</span>",
                       "font": {"size": 17, "color": "#e6edf3"}},
            gauge   = {
                "axis":        {"range": [0, 80], "tickcolor": "#8b96a8", "tickfont": {"color": "#8b96a8"}},
                "bar":         {"color": color, "thickness": 0.30},
                "bgcolor":     "#161b2e",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 28],  "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [28, 43], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [43, 80], "color": "rgba(239, 68, 68, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3"},
        height=290,
        margin={"l": 20, "r": 20, "t": 60, "b": 10},
    )
    return fig


def get_shap_contributions(X_enc):
    """SHAP contributions for the default-class on the first row."""
    sv = explainer.shap_values(X_enc.values)
    if isinstance(sv, list):
        return np.asarray(sv[1])[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        return sv[0, :, 1]
    return sv[0]


def risk_adjusted_pricing(prob, current_rate, loan_amount):
    """
    Recommend a rate / max loan amount for a given risk score.
    Real banks use 'risk-based pricing' — higher risk = higher rate.
    """
    suggested_rate   = current_rate
    suggested_amount = loan_amount
    conditions       = []

    if prob >= 0.65:
        suggested_rate   = current_rate + 4.0   # add 4 pts for high-risk
        suggested_amount = round(loan_amount * 0.50, -2)   # halve the loan
        conditions.append("Require a co-signer with strong credit")
        conditions.append("Limit term to ≤ 24 months")
        conditions.append("Require proof of stable employment ≥ 2 years")
    elif prob >= 0.40:
        suggested_rate   = current_rate + 2.0
        suggested_amount = round(loan_amount * 0.75, -2)
        conditions.append("Consider lower loan amount or shorter term")
        conditions.append("Verify recent bank statements (3 months)")
    else:
        conditions.append("Standard terms acceptable")
        conditions.append("Eligible for promotional rate if applicable")

    return min(suggested_rate, 35.0), suggested_amount, conditions


def stress_test(X_enc, base_prob, base_payment, base_dti, annual_income):
    """
    Re-run the model under adverse scenarios:
      • Interest rate rises 2 percentage points
      • Annual income drops 20 %
    Returns a DataFrame with each scenario's impact.
    """
    rows = []

    # Scenario 1: Rate shock +2 pts
    X_rate = X_enc.copy()
    X_rate["loan_int_rate"] = X_rate["loan_int_rate"] + 2.0
    p1 = float(model.predict_proba(X_rate)[0][1])
    new_payment = monthly_payment(
        principal       = float(X_enc["loan_amnt"].iloc[0]),
        annual_rate_pct = float(X_rate["loan_int_rate"].iloc[0]),
    )
    new_dti = dti_ratio(new_payment, annual_income)
    rows.append({
        "Scenario":         "Rate +2 pts",
        "Default Prob":     f"{p1:.1%}",
        "Δ Prob":           f"{(p1 - base_prob)*100:+.1f} pts",
        "Monthly Payment":  f"${new_payment:,.0f}",
        "DTI":              f"{new_dti:.1%}",
    })

    # Scenario 2: Income shock −20%
    X_inc = X_enc.copy()
    new_income = annual_income * 0.80
    X_inc["person_income"]       = new_income
    X_inc["loan_percent_income"] = round(float(X_inc["loan_amnt"].iloc[0]) / new_income, 4)
    p2 = float(model.predict_proba(X_inc)[0][1])
    new_dti2 = dti_ratio(base_payment, new_income)
    rows.append({
        "Scenario":         "Income −20%",
        "Default Prob":     f"{p2:.1%}",
        "Δ Prob":           f"{(p2 - base_prob)*100:+.1f} pts",
        "Monthly Payment":  f"${base_payment:,.0f}",
        "DTI":              f"{new_dti2:.1%}",
    })

    # Scenario 3: Combined rate +2 AND income −20%
    X_both = X_inc.copy()
    X_both["loan_int_rate"] = X_both["loan_int_rate"] + 2.0
    p3 = float(model.predict_proba(X_both)[0][1])
    new_payment3 = monthly_payment(
        principal=float(X_both["loan_amnt"].iloc[0]),
        annual_rate_pct=float(X_both["loan_int_rate"].iloc[0]),
    )
    new_dti3 = dti_ratio(new_payment3, new_income)
    rows.append({
        "Scenario":         "Combined shock",
        "Default Prob":     f"{p3:.1%}",
        "Δ Prob":           f"{(p3 - base_prob)*100:+.1f} pts",
        "Monthly Payment":  f"${new_payment3:,.0f}",
        "DTI":              f"{new_dti3:.1%}",
    })

    return pd.DataFrame(rows)


def make_application_id(raw_dict):
    """Generate a deterministic 8-char application ID from the raw inputs."""
    h = hashlib.md5(str(raw_dict).encode()).hexdigest()[:8].upper()
    return f"LDP-{h}"


# =============================================================
# TAB 1 — UNDERWRITE (single applicant)
# =============================================================
with tab_predict:

    # ---- Input form inside a bordered card ----
    with st.container(border=True):
        st.markdown("#### 📋  New Loan Application")

        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**👤 Applicant**")
            person_age = st.number_input(
                "Age (years)", min_value=18, max_value=80, value=35, step=1
            )
            person_income = st.number_input(
                "Annual Income ($)", min_value=5_000, max_value=500_000,
                value=55_000, step=1_000
            )
            person_emp_length = st.number_input(
                "Employment (years)", min_value=0.0, max_value=41.0,
                value=5.0, step=0.5
            )
            person_home_ownership = st.selectbox(
                "Home Ownership", options=encoder_cats["person_home_ownership"]
            )

        with c2:
            st.markdown("**📜 Credit History**")
            cb_person_default_on_file = st.selectbox(
                "Past Default on File?",
                options=encoder_cats["cb_person_default_on_file"],
                format_func=lambda x: "Yes" if x == "Y" else "No",
            )
            cb_person_cred_hist_length = st.number_input(
                "Credit History (years)", min_value=2, max_value=30,
                value=8, step=1
            )
            loan_grade = st.selectbox(
                "Loan Grade  (A=best · G=worst)",
                options=encoder_cats["loan_grade"]
            )
            loan_intent = st.selectbox(
                "Loan Purpose", options=encoder_cats["loan_intent"]
            )

        with c3:
            st.markdown("**💰 Loan Terms**")
            loan_amnt = st.number_input(
                "Loan Amount ($)", min_value=500, max_value=35_000,
                value=10_000, step=500
            )
            loan_int_rate = st.slider(
                "Interest Rate (%)", min_value=5.0, max_value=35.0,
                value=12.0, step=0.25
            )
            loan_term_months = st.selectbox(
                "Loan Term",
                options=[12, 24, 36, 48, 60],
                index=2,           # default = 36 months
                format_func=lambda x: f"{x} months ({x // 12} years)",
            )

    # Calculated behind the scenes (no clutter on the form)
    loan_percent_income = (
        round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    )

    # ---- Action button ----
    predict_clicked = st.button(
        "🔍  Run Risk Analysis",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        # Build the single-row DataFrame from the form inputs
        raw = pd.DataFrame([{
            "person_age": person_age,
            "person_income": person_income,
            "person_emp_length": person_emp_length,
            "loan_amnt": loan_amnt,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "person_home_ownership": person_home_ownership,
            "loan_intent": loan_intent,
            "loan_grade": loan_grade,
            "cb_person_default_on_file": cb_person_default_on_file,
        }])
        X_enc        = encode_row(raw)
        default_prob = float(model.predict_proba(X_enc)[0][1])

        # ============================================================
        # ❶  APPLICATION REFERENCE STRIP
        # ============================================================
        app_id    = make_application_id(raw.to_dict())
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        st.markdown(
            f"""
            <div class="ref-strip">
                <b>Application ID:</b> {app_id}  &nbsp;·&nbsp;
                <b>Generated:</b> {timestamp}  &nbsp;·&nbsp;
                <b>Model:</b> RandomForest v1.0  &nbsp;·&nbsp;
                <b>Status:</b> <span style="color:#10b981">PROCESSED</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ============================================================
        # ❷  UNDERWRITING DECISION BANNER
        # ============================================================
        verdict, dec_color, reasons = underwriting_decision(
            default_prob, cb_person_default_on_file, loan_percent_income, loan_grade,
        )
        st.markdown(
            f"""
            <div class="decision-banner" style="
                background: linear-gradient(135deg, {dec_color}22, {dec_color}11);
                border: 2px solid {dec_color}66;
            ">
                <h1 style="color:{dec_color}">{verdict}</h1>
                <p style="color:#e6edf3">Default probability: <b>{default_prob:.1%}</b> &nbsp;·&nbsp; Application {app_id}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ============================================================
        # ❸  TWIN GAUGES  +  REASONING
        # ============================================================
        # Compute loan financials we'll need throughout
        mp     = monthly_payment(loan_amnt, loan_int_rate, months=loan_term_months)
        total_paid     = mp * loan_term_months
        total_interest = total_paid - loan_amnt
        dti            = dti_ratio(mp, person_income)

        with st.container(border=True):
            st.markdown("#### 📊  Risk & Affordability Assessment")
            g1, g2, g3 = st.columns([1, 1, 1.2])

            with g1:
                st.plotly_chart(make_risk_gauge(default_prob), use_container_width=True)
            with g2:
                st.plotly_chart(make_dti_gauge(dti), use_container_width=True)
            with g3:
                st.markdown("**📌 Underwriting Notes**")
                for r in reasons[:5]:
                    st.markdown(f"- {r}")

        # ============================================================
        # ❹  LOAN FINANCIALS  (real bank-style breakdown)
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 💰  Loan Financial Summary")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Monthly Payment", f"${mp:,.2f}")
            f2.metric("Total Interest",  f"${total_interest:,.0f}")
            f3.metric("Total Cost",      f"${total_paid:,.0f}")
            f4.metric("DTI Ratio",       f"{dti:.1%}",
                      delta=f"{dti_tier(dti)[0]}",
                      delta_color="off")

            f5, f6, f7, f8 = st.columns(4)
            f5.metric("Loan Amount",     f"${loan_amnt:,}")
            f6.metric("Interest Rate",   f"{loan_int_rate:.2f}%")
            f7.metric("Term",            f"{loan_term_months} mo")
            f8.metric("Annual Income",   f"${person_income:,}")

        # ============================================================
        # ❺  RISK-ADJUSTED PRICING RECOMMENDATION
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 🎯  Risk-Adjusted Pricing Recommendation")
            new_rate, new_amount, conditions = risk_adjusted_pricing(
                default_prob, loan_int_rate, loan_amnt
            )

            p1, p2, p3 = st.columns(3)
            p1.metric(
                label = "Suggested Rate",
                value = f"{new_rate:.2f}%",
                delta = f"{(new_rate - loan_int_rate):+.2f} pts",
                delta_color="inverse",
            )
            p2.metric(
                label = "Suggested Max Amount",
                value = f"${new_amount:,.0f}",
                delta = f"${(new_amount - loan_amnt):+,.0f}",
                delta_color="off",
            )
            new_mp = monthly_payment(new_amount, new_rate, months=loan_term_months)
            p3.metric(
                label = "Adjusted Monthly Payment",
                value = f"${new_mp:,.0f}",
            )

            st.markdown("**Conditions for approval:**")
            for cond in conditions:
                st.markdown(f"- {cond}")

        # ============================================================
        # ❻  STRESS TEST
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 🧪  Stress Test — Adverse Scenarios")
            st.caption(
                "How resilient is this loan if conditions worsen? "
                "We re-score the applicant under three classic shock scenarios."
            )
            stress_df = stress_test(X_enc, default_prob, mp, dti, person_income)
            st.dataframe(stress_df, use_container_width=True, hide_index=True)

        # ============================================================
        # ❼  SHAP EXPLANATION
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 🔍  Why this decision? — SHAP feature impact")
            st.caption(
                "**Red bars** push toward default · **Blue bars** push away. "
                "Longer bars matter more for this applicant."
            )

            with st.spinner("Computing SHAP values..."):
                contributions = get_shap_contributions(X_enc)
                labels        = [nice_names.get(c, c) for c in feature_cols]
                order         = np.argsort(np.abs(contributions))
                sv_sorted     = contributions[order]
                labels_sorted = np.array(labels)[order]

                fig, ax = plt.subplots(figsize=(10, 4.5))
                fig.patch.set_facecolor("#0b0f1a")
                ax.set_facecolor("#0b0f1a")
                bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in sv_sorted]
                ax.barh(labels_sorted, sv_sorted, color=bar_colors)
                ax.axvline(0, color="#8b96a8", linewidth=0.8, linestyle="--")
                ax.set_xlabel("SHAP value  (positive = pushes toward default)", color="#e6edf3")
                for spine in ax.spines.values():
                    spine.set_color("#1f2742")
                ax.tick_params(colors="#e6edf3")
                x_abs = max(abs(sv_sorted.max()), abs(sv_sorted.min())) * 1.3 or 0.1
                ax.set_xlim(-x_abs, x_abs)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

        # ============================================================
        # ❽  DOWNLOAD REPORT
        # ============================================================
        report = raw.T.reset_index()
        report.columns = ["Field", "Value"]
        contribs_df = pd.DataFrame({
            "Field": [f"SHAP_{c}" for c in feature_cols],
            "Value": contributions,
        })
        summary_df = pd.DataFrame({
            "Field": [
                "application_id", "timestamp", "decision",
                "default_probability", "monthly_payment",
                "total_interest", "total_cost", "dti_ratio",
                "suggested_rate", "suggested_amount",
            ],
            "Value": [
                app_id, timestamp, verdict.strip(),
                round(default_prob, 4), round(mp, 2),
                round(total_interest, 2), round(total_paid, 2), round(dti, 4),
                round(new_rate, 2), round(new_amount, 2),
            ],
        })
        full_report = pd.concat([summary_df, report, contribs_df], ignore_index=True)
        csv_bytes   = full_report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label=f"📥  Download underwriting report — {app_id}.csv",
            data=csv_bytes,
            file_name=f"{app_id}_underwriting_report.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =============================================================
# TAB 2 — BATCH SCORING (CSV upload)
# =============================================================
with tab_batch:
    st.subheader("Batch Loan Scoring")
    st.markdown(
        "Upload a CSV file with **the same columns used during training** to "
        "score many applicants at once.  Each row gets a default probability "
        "and a risk tier, which you can then download."
    )

    with st.expander("📋  Required columns (click to expand)"):
        st.code("\n".join(feature_cols), language="text")
        template_row = {c: "" for c in feature_cols}
        defaults = {
            "person_age": 35, "person_income": 55_000, "person_emp_length": 5.0,
            "loan_amnt": 10_000, "loan_int_rate": 12.0, "loan_percent_income": 0.18,
            "cb_person_cred_hist_length": 8,
            "person_home_ownership": encoder_cats["person_home_ownership"][0],
            "loan_intent":           encoder_cats["loan_intent"][0],
            "loan_grade":            encoder_cats["loan_grade"][0],
            "cb_person_default_on_file": encoder_cats["cb_person_default_on_file"][0],
        }
        template_row.update(defaults)
        template_df    = pd.DataFrame([template_row])[feature_cols]
        template_bytes = template_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️  Download CSV template",
            data=template_bytes,
            file_name="loan_applicants_template.csv",
            mime="text/csv",
        )

    uploaded = st.file_uploader("Upload applicants CSV", type=["csv"])

    if uploaded is not None:
        try:
            df_batch = pd.read_csv(uploaded)
            st.success(f"✓ Loaded **{len(df_batch):,}** rows.")

            missing = [c for c in feature_cols if c not in df_batch.columns]
            if missing:
                st.error(f"Missing required columns: {missing}.")
            else:
                X_enc_batch = encode_row(df_batch)
                probs       = model.predict_proba(X_enc_batch)[:, 1]

                df_out = df_batch.copy()
                df_out["default_probability"] = probs.round(4)
                df_out["risk_tier"] = pd.cut(
                    probs,
                    bins   = [-0.01, 0.40, 0.65, 1.01],
                    labels = ["LOW", "MEDIUM", "HIGH"],
                )

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Applicants", f"{len(df_out):,}")
                m2.metric("🟢 Low Risk",   int((df_out['risk_tier'] == 'LOW').sum()))
                m3.metric("🟡 Medium Risk", int((df_out['risk_tier'] == 'MEDIUM').sum()))
                m4.metric("🔴 High Risk",  int((df_out['risk_tier'] == 'HIGH').sum()))

                st.dataframe(df_out, use_container_width=True, hide_index=True)

                csv_out = df_out.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="⬇️  Download scored CSV",
                    data=csv_out,
                    file_name="loan_applicants_scored.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Could not process the file: {e}")


# =============================================================
# TAB 3 — MODEL INSIGHTS
# =============================================================
with tab_insights:
    st.subheader("How the Model Decides")
    st.markdown(
        "These charts come from the **training run** and show the model's "
        "*overall* behaviour — across thousands of applicants, not just one."
    )

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        st.markdown("#### Global Feature Importance (SHAP)")
        if os.path.exists("shap_importance.png"):
            st.image("shap_importance.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate shap_importance.png")

    with chart_col2:
        st.markdown("#### Confusion Matrix (test set)")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate confusion_matrix.png")

    st.divider()
    st.markdown("#### Encoded Categories")
    st.caption("The model sees integers, not strings — these are the mappings learned from training data.")
    for col, cats in encoder_cats.items():
        st.markdown(f"**{nice_names.get(col, col)}** (`{col}`)")
        st.code({c: i for i, c in enumerate(cats)}, language="python")


# =============================================================
# TAB 4 — ABOUT (model snapshot lives here now)
# =============================================================
with tab_about:
    st.subheader("📊 Model Snapshot")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Test ROC-AUC",     f"{roc_auc:.4f}" if roc_auc else "n/a")
    m2.metric("Algorithm",        "Random Forest")
    m3.metric("Trees",            "200")
    m4.metric("Features",         len(feature_cols))

    st.divider()

    st.subheader("About this project")
    st.markdown(
        """
        **CreditScope** is a portfolio project demonstrating a full end-to-end
        machine-learning workflow with bank-grade UX:

        1. **Data**          — Kaggle credit-risk dataset (or generated equivalent)
        2. **Modelling**     — Random Forest (200 trees · `class_weight='balanced'`)
        3. **Evaluation**    — Stratified train/test split · ROC-AUC · precision · recall
        4. **Explainability**— SHAP per-applicant *and* global feature importance
        5. **Financials**    — Loan amortization · DTI · risk-adjusted pricing
        6. **Stress testing**— Rate shock · income shock · combined scenarios
        7. **Deployment**    — Streamlit Community Cloud (free hosting)

        ---

        **Tech stack**
        - `scikit-learn` — model training
        - `SHAP` — explainability
        - `Streamlit` + `Plotly` — interactive web UI
        - `pandas` / `numpy` — data wrangling
        - `joblib` — model serialization

        ---

        **What makes it bank-grade?**
        - Real underwriting workflow: ML score + DTI + classic credit rules
        - Risk-adjusted pricing recommendations (higher risk → higher rate)
        - Stress testing under adverse scenarios
        - Auditable per-applicant SHAP explanations
        - Persistent application reference IDs for downstream tracking

        ---

        **Disclaimer**
        This app is for educational and portfolio purposes only.  Predictions
        are statistical estimates and must **not** be used to make real
        lending decisions.
        """
    )
    st.caption("Built with Python · Streamlit · Plotly · SHAP")
