# =============================================================
# app.py  —  CreditScope: Bank-Grade Loan Underwriting Dashboard
# =============================================================
# An end-to-end underwriting tool modelled after professional bank
# workflows (Maybank, BCA, FICO-style scoring) that, for any applicant:
#   • Predicts default probability with a Random Forest
#   • Converts that into a FICO-style credit score (300-850)
#   • Calculates DSR (Debt Service Ratio) — Maybank/BCA standard
#   • Computes loan financials (monthly payment, total cost)
#   • Returns max eligible loan amount based on income & DSR cap
#   • Routes to approval authority level (L1/L2/L3)
#   • Runs compliance checklist (KYC / AML / Bureau / Income / Age)
#   • Generates amortization schedule preview
#   • Stress tests under rate / income shocks
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
# DEVELOPER PROFILE (shown in sidebar + About tab)
# ============================================================
DEVELOPER_NAME     = "Jolly Chen Wang"
DEVELOPER_LINKEDIN = "https://www.linkedin.com/in/jollychenwang/"
DEVELOPER_GITHUB   = "https://github.com/JollyChenn/loan-default-app"


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

        /* Compliance pills */
        .pill {
            display: inline-block;
            padding: 6px 12px;
            margin: 4px 6px 4px 0;
            border-radius: 20px;
            font-size: 13px;
            font-weight: 500;
        }
        .pill-pass { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid #10b98155; }
        .pill-fail { background: rgba(239,68,68,0.15);  color: #ef4444; border: 1px solid #ef444455; }
        .pill-warn { background: rgba(245,158,11,0.15); color: #f59e0b; border: 1px solid #f59e0b55; }

        /* Credit score card */
        .score-card {
            background: linear-gradient(135deg, #161b2e 0%, #1f2742 100%);
            border: 1px solid #2d3859;
            border-radius: 14px;
            padding: 22px;
            text-align: center;
        }
        .score-card .score-num { font-size: 56px; font-weight: 800; letter-spacing: -2px; line-height: 1; }
        .score-card .score-label { font-size: 13px; color: #8b96a8; margin-top: 6px; letter-spacing: 1px; }

        /* Developer profile in sidebar */
        .dev-profile {
            background: #1a1f2e;
            border-radius: 10px;
            padding: 14px;
            margin-top: 14px;
            text-align: center;
            border: 1px solid #1f2742;
        }
        .dev-profile p { margin: 0 0 6px; color: #8b96a8; font-size: 12px; }
        .dev-profile a {
            display: inline-block;
            margin: 4px 4px 0;
            padding: 5px 12px;
            background: #0a66c2;
            color: white !important;
            text-decoration: none !important;
            border-radius: 6px;
            font-size: 12px;
            font-weight: 500;
        }
        .dev-profile a.gh { background: #24292f; }
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

# Unpack the parts we saved during training
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
# SIDEBAR — branding + guide + developer profile
# ============================================================
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>🏦 CreditScope</h2>
            <p>Bank-Grade Underwriting</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 📖 Quick Guide")
    st.markdown(
        """
        1. Enter applicant & loan details
        2. Click **Run Risk Analysis**
        3. Review decision + credit score
        4. Check compliance & financials
        5. Download underwriting report
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

    st.markdown("### 💳 Credit Score Bands")
    st.markdown(
        """
        🌟 **Exceptional** · 800+
        ✨ **Very Good**   · 740–799
        ✅ **Good**        · 670–739
        ⚠️ **Fair**        · 580–669
        🔻 **Poor**        · &lt; 580
        """
    )
    st.divider()

    # Developer profile card with LinkedIn + GitHub
    st.markdown(
        f"""
        <div class="dev-profile">
            <p>Built by</p>
            <b style="color:#e6edf3">{DEVELOPER_NAME}</b><br>
            <a href="{DEVELOPER_LINKEDIN}" target="_blank">💼 LinkedIn</a>
            <a class="gh" href="{DEVELOPER_GITHUB}" target="_blank">⭐ GitHub</a>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("v2.1 · Educational use only")


# ============================================================
# HERO HEADER
# ============================================================
st.markdown(
    """
    <div class="hero-header">
        <h1>🏦 CreditScope — Loan Underwriting Dashboard</h1>
        <p>AI-powered default prediction · FICO-style scoring · DSR · Compliance · Stress testing · SHAP explainability</p>
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


def credit_score_from_prob(prob):
    """
    Map default probability (0-1) onto a FICO-style 300-850 scale.
    Lower default probability → higher credit score.
    Real bureaus (Experian, Equifax, TransUnion) use proprietary models;
    this is a transparent linear approximation for demo purposes.
    """
    return int(round(850 - (prob * 550)))


def credit_score_tier(score):
    """FICO-standard tier classification (used by Maybank, BCA, US banks)."""
    if score >= 800: return "Exceptional", "#10b981"
    if score >= 740: return "Very Good",   "#10b981"
    if score >= 670: return "Good",        "#06b6d4"
    if score >= 580: return "Fair",        "#f59e0b"
    return                "Poor",          "#ef4444"


def underwriting_decision(prob, default_on_file, percent_income, loan_grade):
    """
    Combine the ML risk score with classic underwriting rules to issue
    a final decision: APPROVE, CONDITIONAL APPROVAL, or DECLINE.
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
    Standard loan amortization formula:  M = P · r(1+r)^n / ((1+r)^n - 1)
    """
    r = (annual_rate_pct / 100) / 12
    if r == 0:
        return principal / months
    return principal * (r * (1 + r) ** months) / ((1 + r) ** months - 1)


def dsr_ratio(monthly_pmt, annual_income):
    """
    DSR (Debt Service Ratio) — the Maybank / BCA / Bank Negara standard.
    Same formula as DTI = monthly payment / monthly income.
    Bank Negara Malaysia limits DSR to 60–70% for personal loans.
    """
    monthly_income = annual_income / 12
    return monthly_pmt / monthly_income if monthly_income > 0 else 0.0


def dsr_tier(dsr):
    """Bank-standard DSR tiers (Bank Negara guidance + bank policy)."""
    if dsr < 0.30:
        return "Healthy",      "#10b981"
    if dsr < 0.45:
        return "Cautionary",   "#f59e0b"
    return     "Over-leveraged", "#ef4444"


def max_eligible_loan(annual_income, annual_rate_pct, months, dsr_cap=0.40):
    """
    Reverse-amortization: maximum loan an applicant qualifies for at
    a given DSR cap (the percentage of monthly income that may go to debt).
    Most retail banks cap DSR at 40-50% for unsecured personal loans.
    """
    monthly_capacity = (annual_income / 12) * dsr_cap
    r = (annual_rate_pct / 100) / 12
    if r == 0:
        return monthly_capacity * months
    return monthly_capacity * ((1 + r) ** months - 1) / (r * (1 + r) ** months)


def approval_authority(loan_amnt, prob):
    """
    Route the loan to the right approval authority. Real banks have
    tiered sign-off limits — small loans clear at the branch, large
    ones go to credit committee.
    """
    if loan_amnt < 10_000 and prob < 0.40:
        return "L1 — Branch Officer",     "#10b981"
    if loan_amnt < 25_000 and prob < 0.65:
        return "L2 — Branch Manager",     "#f59e0b"
    return     "L3 — Credit Committee",   "#ef4444"


def compliance_checks(loan_amnt, cred_hist_length, employment_length, person_age, default_on_file):
    """
    Simulated underwriting compliance checklist — every retail bank
    runs these (KYC, AML, Bureau, income, age eligibility) before any
    loan is disbursed.  Returns a list of (label, status, detail).
    """
    return [
        ("KYC Verification",   "PASS",
         "Identity documents on file"),
        ("AML Screening",      "PASS" if loan_amnt <= 30_000 else "REVIEW",
         "Within standard threshold" if loan_amnt <= 30_000 else "Enhanced due diligence needed"),
        ("Credit Bureau Check","PASS" if cred_hist_length >= 2 else "FAIL",
         f"{cred_hist_length} years of credit history"),
        ("Income Verification","PASS" if employment_length >= 1 else "REVIEW",
         f"{employment_length} years employed"),
        ("Age Eligibility",    "PASS" if 21 <= person_age <= 65 else "FAIL",
         f"Applicant is {person_age} years old"),
        ("Adverse History",    "PASS" if default_on_file == "N" else "FLAG",
         "Clean record" if default_on_file == "N" else "Prior default on file"),
    ]


def amortization_schedule(principal, annual_rate_pct, months, n=6):
    """
    First N months of the loan's amortization schedule — shows the
    classic principal-vs-interest split a banker walks the customer through.
    """
    r = (annual_rate_pct / 100) / 12
    M = monthly_payment(principal, annual_rate_pct, months)
    balance = principal
    rows = []
    for i in range(1, min(n, months) + 1):
        interest  = balance * r
        prin_paid = M - interest
        balance  -= prin_paid
        rows.append({
            "Month":     i,
            "Payment":   f"${M:,.2f}",
            "Principal": f"${prin_paid:,.2f}",
            "Interest":  f"${interest:,.2f}",
            "Balance":   f"${max(balance, 0):,.2f}",
        })
    return pd.DataFrame(rows)


def make_risk_gauge(prob, title="Default Risk"):
    """Plotly half-gauge for risk probability."""
    _, label, color = risk_label(prob)
    fig = go.Figure(
        go.Indicator(
            mode    = "gauge+number",
            value   = prob * 100,
            number  = {"suffix": " %", "font": {"size": 40, "color": "#e6edf3"}},
            title   = {"text": f"<b>{title}</b><br><span style='font-size:13px;color:{color}'>{label}</span>",
                       "font": {"size": 16, "color": "#e6edf3"}},
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
        height=260,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
    )
    return fig


def make_dsr_gauge(dsr, title="Debt Service Ratio"):
    """Plotly gauge for affordability (DSR)."""
    label, color = dsr_tier(dsr)
    fig = go.Figure(
        go.Indicator(
            mode    = "gauge+number",
            value   = dsr * 100,
            number  = {"suffix": " %", "font": {"size": 40, "color": "#e6edf3"}},
            title   = {"text": f"<b>{title}</b><br><span style='font-size:13px;color:{color}'>{label}</span>",
                       "font": {"size": 16, "color": "#e6edf3"}},
            gauge   = {
                "axis":        {"range": [0, 80], "tickcolor": "#8b96a8", "tickfont": {"color": "#8b96a8"}},
                "bar":         {"color": color, "thickness": 0.30},
                "bgcolor":     "#161b2e",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 30],  "color": "rgba(16, 185, 129, 0.25)"},
                    {"range": [30, 45], "color": "rgba(245, 158, 11, 0.25)"},
                    {"range": [45, 80], "color": "rgba(239, 68, 68, 0.25)"},
                ],
            },
        )
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#e6edf3"},
        height=260,
        margin={"l": 20, "r": 20, "t": 50, "b": 10},
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
    """Risk-based pricing — higher risk → higher rate + smaller loan."""
    suggested_rate   = current_rate
    suggested_amount = loan_amount
    conditions       = []

    if prob >= 0.65:
        suggested_rate   = current_rate + 4.0
        suggested_amount = round(loan_amount * 0.50, -2)
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


def stress_test(X_enc, base_prob, base_payment, annual_income):
    """Re-run the model under three adverse scenarios."""
    rows = []

    # Scenario 1: Rate shock +2 pts
    X_rate = X_enc.copy()
    X_rate["loan_int_rate"] = X_rate["loan_int_rate"] + 2.0
    p1 = float(model.predict_proba(X_rate)[0][1])
    new_payment = monthly_payment(
        principal       = float(X_enc["loan_amnt"].iloc[0]),
        annual_rate_pct = float(X_rate["loan_int_rate"].iloc[0]),
    )
    new_dsr = dsr_ratio(new_payment, annual_income)
    rows.append({"Scenario": "Rate +2 pts",
                 "Default Prob": f"{p1:.1%}",
                 "Δ Prob":       f"{(p1 - base_prob)*100:+.1f} pts",
                 "Monthly Pmt":  f"${new_payment:,.0f}",
                 "DSR":          f"{new_dsr:.1%}"})

    # Scenario 2: Income shock −20%
    X_inc = X_enc.copy()
    new_income = annual_income * 0.80
    X_inc["person_income"]       = new_income
    X_inc["loan_percent_income"] = round(float(X_inc["loan_amnt"].iloc[0]) / new_income, 4)
    p2 = float(model.predict_proba(X_inc)[0][1])
    new_dsr2 = dsr_ratio(base_payment, new_income)
    rows.append({"Scenario": "Income −20%",
                 "Default Prob": f"{p2:.1%}",
                 "Δ Prob":       f"{(p2 - base_prob)*100:+.1f} pts",
                 "Monthly Pmt":  f"${base_payment:,.0f}",
                 "DSR":          f"{new_dsr2:.1%}"})

    # Scenario 3: Combined shock
    X_both = X_inc.copy()
    X_both["loan_int_rate"] = X_both["loan_int_rate"] + 2.0
    p3 = float(model.predict_proba(X_both)[0][1])
    new_payment3 = monthly_payment(
        principal=float(X_both["loan_amnt"].iloc[0]),
        annual_rate_pct=float(X_both["loan_int_rate"].iloc[0]),
    )
    new_dsr3 = dsr_ratio(new_payment3, new_income)
    rows.append({"Scenario": "Combined shock",
                 "Default Prob": f"{p3:.1%}",
                 "Δ Prob":       f"{(p3 - base_prob)*100:+.1f} pts",
                 "Monthly Pmt":  f"${new_payment3:,.0f}",
                 "DSR":          f"{new_dsr3:.1%}"})

    return pd.DataFrame(rows)


def make_application_id(raw_dict):
    """Generate a deterministic 8-char application ID from the raw inputs."""
    h = hashlib.md5(str(raw_dict).encode()).hexdigest()[:8].upper()
    return f"LDP-{h}"


def render_compliance_pills(checks):
    """Build a single HTML row of compliance pill badges."""
    pill_html = ""
    for label, status, detail in checks:
        klass = {"PASS": "pill-pass", "FAIL": "pill-fail",
                 "REVIEW": "pill-warn", "FLAG": "pill-warn"}[status]
        icon  = {"PASS": "✓", "FAIL": "✗", "REVIEW": "⚠", "FLAG": "⚑"}[status]
        pill_html += f'<span class="pill {klass}" title="{detail}">{icon} {label}</span>'
    return pill_html


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

    # Computed behind the scenes
    loan_percent_income = (
        round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
    )

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

        # Derived bank metrics
        cs_score          = credit_score_from_prob(default_prob)
        cs_label, cs_col  = credit_score_tier(cs_score)
        mp                = monthly_payment(loan_amnt, loan_int_rate, loan_term_months)
        total_paid        = mp * loan_term_months
        total_interest    = total_paid - loan_amnt
        dsr               = dsr_ratio(mp, person_income)
        dsr_lbl, dsr_col  = dsr_tier(dsr)
        max_loan          = max_eligible_loan(person_income, loan_int_rate, loan_term_months)
        auth, auth_col    = approval_authority(loan_amnt, default_prob)
        checks            = compliance_checks(
            loan_amnt, cb_person_cred_hist_length, person_emp_length,
            person_age, cb_person_default_on_file,
        )

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
        # ❷  DECISION BANNER
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
        # ❸  CREDIT-BUREAU SCORE CARD  +  KEY KPIs
        # ============================================================
        k1, k2, k3, k4 = st.columns([1.1, 1, 1, 1])
        with k1:
            st.markdown(
                f"""
                <div class="score-card">
                    <div class="score-label">CREDITSCOPE SCORE</div>
                    <div class="score-num" style="color:{cs_col}">{cs_score}</div>
                    <div style="color:{cs_col}; font-weight:600; font-size:15px; margin-top:4px">{cs_label}</div>
                    <div style="color:#8b96a8; font-size:11px; margin-top:6px">FICO-equivalent · 300–850</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with k2:
            st.metric("Debt Service Ratio (DSR)", f"{dsr:.1%}",
                      delta=dsr_lbl, delta_color="off")
        with k3:
            st.metric("Max Eligible Amount", f"${max_loan:,.0f}",
                      delta=f"vs requested ${loan_amnt:,.0f}", delta_color="off")
        with k4:
            st.metric("Approval Authority", auth.split('—')[0].strip(),
                      delta=auth.split('—')[1].strip(), delta_color="off")

        # ============================================================
        # ❹  TWIN GAUGES + UNDERWRITING NOTES
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 📊  Risk & Affordability Assessment")
            g1, g2, g3 = st.columns([1, 1, 1.3])
            with g1:
                st.plotly_chart(make_risk_gauge(default_prob), use_container_width=True)
            with g2:
                st.plotly_chart(make_dsr_gauge(dsr), use_container_width=True)
            with g3:
                st.markdown("**📌 Underwriting Notes**")
                for r in reasons[:5]:
                    st.markdown(f"- {r}")

        # ============================================================
        # ❺  COMPLIANCE CHECKLIST  (Maybank / BCA standard)
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 🛡️  Compliance Checklist")
            st.caption(
                "Standard KYC/AML/Bureau checks every retail bank runs "
                "before disbursement. Hover a pill to see detail."
            )
            st.markdown(render_compliance_pills(checks), unsafe_allow_html=True)

        # ============================================================
        # ❻  LOAN FINANCIALS
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 💰  Loan Financial Summary")
            f1, f2, f3, f4 = st.columns(4)
            f1.metric("Monthly Payment", f"${mp:,.2f}")
            f2.metric("Total Interest",  f"${total_interest:,.0f}")
            f3.metric("Total Cost",      f"${total_paid:,.0f}")
            f4.metric("Effective APR",   f"{loan_int_rate:.2f}%")

            f5, f6, f7, f8 = st.columns(4)
            f5.metric("Principal",       f"${loan_amnt:,}")
            f6.metric("Term",            f"{loan_term_months} mo")
            f7.metric("Annual Income",   f"${person_income:,}")
            f8.metric("Loan / Income",   f"{loan_percent_income:.1%}")

        # ============================================================
        # ❼  REPAYMENT SCHEDULE PREVIEW
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 📅  Repayment Schedule — First 6 Months")
            st.caption(
                "Classic amortization split. Early payments are interest-heavy; "
                "the balance shrinks faster over time."
            )
            sched = amortization_schedule(loan_amnt, loan_int_rate, loan_term_months, n=6)
            st.dataframe(sched, use_container_width=True, hide_index=True)

        # ============================================================
        # ❽  RISK-ADJUSTED PRICING RECOMMENDATION
        # ============================================================
        with st.container(border=True):
            st.markdown("#### 🎯  Risk-Adjusted Pricing Recommendation")
            new_rate, new_amount, conditions = risk_adjusted_pricing(
                default_prob, loan_int_rate, loan_amnt
            )
            p1, p2, p3 = st.columns(3)
            p1.metric("Suggested Rate", f"{new_rate:.2f}%",
                      delta=f"{(new_rate - loan_int_rate):+.2f} pts", delta_color="inverse")
            p2.metric("Suggested Max Amount", f"${new_amount:,.0f}",
                      delta=f"${(new_amount - loan_amnt):+,.0f}", delta_color="off")
            new_mp = monthly_payment(new_amount, new_rate, months=loan_term_months)
            p3.metric("Adjusted Monthly Payment", f"${new_mp:,.0f}")
            st.markdown("**Conditions for approval:**")
            for cond in conditions:
                st.markdown(f"- {cond}")

        # ============================================================
        # ❾  STRESS TEST  (collapsible — keeps the page lean)
        # ============================================================
        with st.expander("🧪  Stress Test — Adverse Scenarios", expanded=False):
            st.caption(
                "How resilient is this loan if conditions worsen? "
                "We re-score the applicant under three classic shock scenarios."
            )
            stress_df = stress_test(X_enc, default_prob, mp, person_income)
            st.dataframe(stress_df, use_container_width=True, hide_index=True)

        # ============================================================
        # ❿  SHAP EXPLANATION  (collapsible)
        # ============================================================
        with st.expander("🔍  Why this decision? — SHAP feature impact", expanded=False):
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
        # Make `contributions` available if SHAP expander wasn't opened
        if "contributions" not in dir():
            contributions = get_shap_contributions(X_enc)

        # ============================================================
        # ⓫  DOWNLOAD REPORT
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
                "default_probability", "creditscope_score", "credit_tier",
                "monthly_payment", "total_interest", "total_cost",
                "dsr_ratio", "max_eligible_loan",
                "approval_authority",
                "suggested_rate", "suggested_amount",
            ],
            "Value": [
                app_id, timestamp, verdict.strip(),
                round(default_prob, 4), cs_score, cs_label,
                round(mp, 2), round(total_interest, 2), round(total_paid, 2),
                round(dsr, 4), round(max_loan, 2),
                auth,
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
        "score many applicants at once.  Each row gets a default probability, "
        "a CreditScope score, and a risk tier — downloadable as CSV."
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
                df_out["creditscope_score"]   = [credit_score_from_prob(p) for p in probs]
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
# TAB 4 — ABOUT
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
        machine-learning workflow built around how real banks (Maybank, BCA,
        and other retail banks) actually underwrite consumer loans:

        ### What it does
        1. **Predicts default risk**   — Random Forest classifier
        2. **Issues a CreditScope**    — FICO-style 300–850 bureau score
        3. **Calculates DSR**          — Debt Service Ratio (Bank Negara standard)
        4. **Runs compliance**         — KYC · AML · Bureau · Income · Age · Adverse history
        5. **Computes financials**     — Monthly payment · total cost · amortization schedule
        6. **Sizes max eligibility**   — Reverse-amortization to DSR cap
        7. **Routes for approval**     — L1 Branch Officer · L2 Manager · L3 Credit Committee
        8. **Recommends pricing**      — Risk-adjusted rate + conditions
        9. **Stress tests**            — Rate shock · income shock · combined scenarios
        10. **Explains decisions**     — SHAP per applicant + global

        ### Tech stack
        - `scikit-learn` — model training
        - `SHAP` — explainability
        - `Streamlit` + `Plotly` — interactive web UI
        - `pandas` / `numpy` — data wrangling
        - `joblib` — model serialization

        ### Disclaimer
        This app is for educational and portfolio purposes only.  Predictions
        are statistical estimates and must **not** be used to make real
        lending decisions.
        """
    )

    st.divider()

    # ---- Developer profile / credits ----
    st.subheader("👤 Built By")
    dev_l, dev_r = st.columns([1, 2])
    with dev_l:
        st.markdown(
            f"""
            <div class="score-card">
                <div class="score-label">DEVELOPER</div>
                <div style="font-size:22px; font-weight:700; color:#e6edf3; margin-top:8px">{DEVELOPER_NAME}</div>
                <div style="margin-top:14px">
                    <a href="{DEVELOPER_LINKEDIN}" target="_blank"
                       style="display:inline-block; padding:8px 16px; background:#0a66c2;
                              color:white; text-decoration:none; border-radius:6px;
                              font-weight:500; margin:4px">
                       💼 LinkedIn
                    </a>
                    <a href="{DEVELOPER_GITHUB}" target="_blank"
                       style="display:inline-block; padding:8px 16px; background:#24292f;
                              color:white; text-decoration:none; border-radius:6px;
                              font-weight:500; margin:4px">
                       ⭐ GitHub
                    </a>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with dev_r:
        st.markdown(
            f"""
            **Hi, I'm {DEVELOPER_NAME}.**
            I build practical machine-learning tools — this project is part of my
            portfolio learning Python, scikit-learn, and Streamlit.

            If you'd like to chat about the project, suggest improvements,
            or discuss opportunities, feel free to connect on
            [LinkedIn]({DEVELOPER_LINKEDIN}) or check out the
            [source code]({DEVELOPER_GITHUB}).

            Thanks for visiting CreditScope!
            """
        )

    st.caption("Built with Python · Streamlit · Plotly · SHAP")
