# =============================================================
# app.py  —  Loan Default Predictor  (Professional Edition)
# =============================================================
# A polished Streamlit dashboard with:
#   • Dark fintech theme
#   • Tabbed layout: Predict / Batch / Model Insights / About
#   • Plotly gauge chart for the risk score
#   • Sidebar with quick-info panel
#   • CSV batch upload (score many applicants at once)
#   • Downloadable single-applicant report
#
# RUN LOCALLY:    streamlit run app.py
# (you must run `python train.py` first so model.pkl exists)
# =============================================================

# ---- IMPORTS -----------------------------------------------
import os
import io                     # for building downloadable CSV bytes in memory
import joblib                 # loads model.pkl from disk
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")         # non-interactive backend (required by Streamlit)
import matplotlib.pyplot as plt
import streamlit as st        # the web-app framework
import shap                   # explains the model's decisions
import plotly.graph_objects as go   # used for the fancy gauge chart


# ============================================================
# PAGE CONFIG  (must be the VERY FIRST Streamlit call)
# ============================================================
st.set_page_config(
    page_title = "Loan Default Predictor",
    page_icon  = "🏦",
    layout     = "wide",          # full-width — looks more like a real dashboard
    initial_sidebar_state = "expanded",
)


# ============================================================
# CUSTOM CSS  —  small polish on top of the dark theme
# ============================================================
# Streamlit lets us inject CSS via st.markdown(unsafe_allow_html=True).
# This adds:
#   • subtle gradient header
#   • rounded "cards" around the metric blocks
#   • tighter spacing
st.markdown(
    """
    <style>
        /* Header gradient strip at the top */
        .hero-header {
            background: linear-gradient(90deg, #10b981 0%, #0ea5e9 100%);
            padding: 28px 36px;
            border-radius: 12px;
            margin-bottom: 24px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }
        .hero-header h1 { color: white; margin: 0; font-size: 32px; }
        .hero-header p  { color: rgba(255,255,255,0.9); margin: 6px 0 0; font-size: 16px; }

        /* Metric "cards" — soft border + rounded corners */
        div[data-testid="stMetric"] {
            background: #161b2e;
            padding: 14px 18px;
            border-radius: 10px;
            border: 1px solid #1f2742;
        }

        /* Tighten section padding */
        .block-container { padding-top: 2rem; }

        /* Sidebar branding */
        .sidebar-brand {
            text-align: center;
            padding: 10px 0 20px;
            border-bottom: 1px solid #1f2742;
            margin-bottom: 18px;
        }
        .sidebar-brand h2 { color: #10b981; margin: 0; font-size: 24px; }
        .sidebar-brand p  { color: #8b96a8; margin: 4px 0 0; font-size: 13px; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD MODEL  (cached so it only loads once per session)
# ============================================================
@st.cache_resource   # Streamlit caches the return value across reruns
def load_artifact():
    """Load the trained model bundle saved by train.py."""
    if not os.path.exists("model.pkl"):
        return None
    return joblib.load("model.pkl")


artifact = load_artifact()

# If train.py has not been run yet, show a helpful error and stop the app.
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


# Build the SHAP explainer once (caching saves a lot of time)
@st.cache_resource
def build_explainer(_model):
    """Create a TreeExplainer (underscore prefix tells Streamlit not to hash it)."""
    return shap.TreeExplainer(_model)


explainer = build_explainer(model)


# ============================================================
# SIDEBAR  —  branding + quick info + global controls
# ============================================================
with st.sidebar:
    # Branded heading using the CSS we injected above
    st.markdown(
        """
        <div class="sidebar-brand">
            <h2>🏦 CreditScope</h2>
            <p>AI-powered loan risk analysis</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Quick-glance model stats
    st.markdown("### Model Snapshot")
    st.metric(label="Test ROC-AUC", value=f"{roc_auc:.3f}" if roc_auc else "n/a")
    st.metric(label="Algorithm",    value="Random Forest")
    st.metric(label="Features",     value=len(feature_cols))
    st.divider()

    # Lightweight "how to read the risk score" key
    st.markdown("### Risk Tiers")
    st.markdown(
        """
        🟢 **Low**     — < 40 %
        🟡 **Medium**  — 40 – 65 %
        🔴 **High**    — > 65 %
        """
    )
    st.divider()
    st.caption("v1.1 · Educational use only")


# ============================================================
# HERO HEADER  (main area, top)
# ============================================================
st.markdown(
    """
    <div class="hero-header">
        <h1>🏦 Loan Default Predictor</h1>
        <p>Random Forest · SHAP Explainability · Real-time scoring</p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# TABBED LAYOUT  —  the four main sections of the dashboard
# ============================================================
tab_predict, tab_batch, tab_insights, tab_about = st.tabs(
    ["🎯  Predict",  "📁  Batch Upload",  "📊  Model Insights",  "ℹ️  About"]
)


# =============================================================
# HELPER FUNCTIONS  (used across multiple tabs)
# =============================================================
def encode_row(df_raw):
    """
    Take a DataFrame of raw applicant rows and return the encoded version
    that the model can consume.  Used by both the single-predict tab and
    the batch-upload tab so the logic stays in ONE place.
    """
    df_raw = df_raw[feature_cols].copy()                    # enforce column order
    df_raw[categorical_cols] = encoder.transform(df_raw[categorical_cols])
    return df_raw


def risk_label(prob):
    """Return (emoji, label, color) for a given default probability."""
    if prob >= 0.65:
        return "🔴", "HIGH RISK",    "#ef4444"
    if prob >= 0.40:
        return "🟡", "MEDIUM RISK",  "#f59e0b"
    return     "🟢", "LOW RISK",     "#10b981"


def make_gauge(prob):
    """
    Build a Plotly gauge chart for a default probability (0–1).
    Gauge charts look much more professional than a flat progress bar.
    """
    _, label, color = risk_label(prob)
    pct = prob * 100

    fig = go.Figure(
        go.Indicator(
            mode    = "gauge+number+delta",
            value   = pct,
            number  = {"suffix": " %", "font": {"size": 48, "color": "#e6edf3"}},
            delta   = {                                  # difference from a 50 % baseline
                "reference": 50,
                "increasing": {"color": "#ef4444"},
                "decreasing": {"color": "#10b981"},
                "suffix":     " pts vs avg",
            },
            title   = {"text": f"<b>{label}</b>", "font": {"size": 20, "color": color}},
            gauge   = {
                "axis":         {"range": [0, 100], "tickcolor": "#8b96a8", "tickfont": {"color": "#8b96a8"}},
                "bar":          {"color": color, "thickness": 0.30},
                "bgcolor":      "#161b2e",
                "borderwidth":  0,
                "steps": [                               # coloured tier bands
                    {"range": [0, 40],   "color": "#10b98133"},   # green
                    {"range": [40, 65],  "color": "#f59e0b33"},   # amber
                    {"range": [65, 100], "color": "#ef444433"},   # red
                ],
                "threshold": {                           # marks the 50 % "neutral" line
                    "line":      {"color": "#e6edf3", "width": 3},
                    "thickness": 0.85,
                    "value":     50,
                },
            },
        )
    )
    # Match the app's dark theme — transparent background, light text
    fig.update_layout(
        paper_bgcolor = "rgba(0,0,0,0)",
        font          = {"color": "#e6edf3"},
        height        = 320,
        margin        = {"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return fig


def get_shap_contributions(X_enc):
    """Return SHAP values for the default-class for the first row of X_enc."""
    sv = explainer.shap_values(X_enc.values)
    # SHAP returns different shapes across versions — handle both.
    if isinstance(sv, list):
        return np.asarray(sv[1])[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                  # (samples, features, classes)
        return sv[0, :, 1]
    return sv[0]                       # (samples, features)


def recommendation_text(prob, default_on_file, percent_income, loan_grade):
    """
    Generate a short human-readable risk recommendation.
    Real underwriters look at multiple factors — we mimic that here in plain English.
    """
    _, label, _ = risk_label(prob)

    notes = []
    if default_on_file == "Y":
        notes.append("Applicant has a **prior default on file** — major red flag.")
    if percent_income > 0.30:
        notes.append(f"Loan is **{percent_income:.0%} of annual income** — high debt burden.")
    if loan_grade in ("E", "F", "G"):
        notes.append(f"Loan grade **{loan_grade}** indicates weak credit profile.")

    if label.startswith("LOW"):
        verdict = "✅ **Recommendation:** Approve — risk profile is acceptable."
    elif label.startswith("MEDIUM"):
        verdict = "⚠️ **Recommendation:** Conditional approval — consider lower amount, higher rate, or co-signer."
    else:
        verdict = "❌ **Recommendation:** Decline — risk significantly exceeds typical underwriting limits."

    return verdict, notes


# =============================================================
# TAB 1 — SINGLE PREDICTION
# =============================================================
with tab_predict:

    st.subheader("Applicant & Loan Details")
    col_left, col_right = st.columns(2)

    # ---- LEFT COLUMN — person-level inputs ----
    with col_left:
        st.markdown("**👤 About the Applicant**")
        person_age = st.number_input(
            "Age (years)", min_value=18, max_value=80, value=35, step=1
        )
        person_income = st.number_input(
            "Annual Income ($)", min_value=5_000, max_value=500_000,
            value=55_000, step=1_000
        )
        person_emp_length = st.number_input(
            "Employment Length (years)", min_value=0.0, max_value=41.0,
            value=5.0, step=0.5
        )
        person_home_ownership = st.selectbox(
            "Home Ownership", options=encoder_cats["person_home_ownership"]
        )
        cb_person_default_on_file = st.selectbox(
            "Past Default on File?",
            options=encoder_cats["cb_person_default_on_file"],
            format_func=lambda x: "Yes — has defaulted before" if x == "Y" else "No — clean history",
        )
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (years)", min_value=2, max_value=30, value=8, step=1
        )

    # ---- RIGHT COLUMN — loan-level inputs ----
    with col_right:
        st.markdown("**💰 About the Loan**")
        loan_amnt = st.number_input(
            "Loan Amount ($)", min_value=500, max_value=35_000,
            value=10_000, step=500
        )
        loan_int_rate = st.slider(
            "Interest Rate (%)", min_value=5.0, max_value=35.0,
            value=12.0, step=0.25
        )
        loan_grade = st.selectbox(
            "Loan Grade  (A = best, G = worst)",
            options=encoder_cats["loan_grade"]
        )
        loan_intent = st.selectbox(
            "Loan Purpose", options=encoder_cats["loan_intent"]
        )
        loan_percent_income = (
            round(loan_amnt / person_income, 4) if person_income > 0 else 0.0
        )
        st.metric(
            label="Loan as % of Annual Income (auto)",
            value=f"{loan_percent_income:.1%}"
        )

    st.divider()

    # ---- Predict button ----
    predict_clicked = st.button(
        "🔍  Run Risk Analysis",
        type="primary",
        use_container_width=True,
    )

    if predict_clicked:

        # Build a single-row DataFrame from the form inputs
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

        # ---- Result section ----
        st.markdown("## 📊 Risk Analysis Report")

        result_left, result_right = st.columns([1.1, 1])

        # LEFT: the gauge chart
        with result_left:
            st.plotly_chart(make_gauge(default_prob), use_container_width=True)

        # RIGHT: numeric verdict + recommendation
        with result_right:
            verdict, notes = recommendation_text(
                default_prob,
                cb_person_default_on_file,
                loan_percent_income,
                loan_grade,
            )
            st.markdown(f"### Probability: **{default_prob:.1%}**")
            st.markdown(verdict)
            if notes:
                st.markdown("**Key risk factors:**")
                for n in notes:
                    st.markdown(f"- {n}")

        st.divider()

        # ---- SHAP explanation chart ----
        st.subheader("Why this prediction? — SHAP feature impact")
        st.caption(
            "**Red bars** push toward default · **Blue bars** push away. "
            "Longer bars matter more for THIS specific applicant."
        )

        with st.spinner("Computing SHAP values..."):
            contributions = get_shap_contributions(X_enc)
            labels        = [nice_names.get(c, c) for c in feature_cols]
            order         = np.argsort(np.abs(contributions))
            sv_sorted     = contributions[order]
            labels_sorted = np.array(labels)[order]

            # Build a matplotlib bar chart on a dark background to match the theme
            fig, ax = plt.subplots(figsize=(9, 5))
            fig.patch.set_facecolor("#0b0f1a")
            ax.set_facecolor("#0b0f1a")
            bar_colors = ["#ef4444" if v > 0 else "#3b82f6" for v in sv_sorted]
            ax.barh(labels_sorted, sv_sorted, color=bar_colors)
            ax.axvline(0, color="#8b96a8", linewidth=0.8, linestyle="--")
            ax.set_xlabel("SHAP value  (positive = pushes toward default)", color="#e6edf3")
            ax.set_title("Per-Feature Contribution to This Applicant's Score", color="#e6edf3")
            # Style the axis labels for dark mode
            for spine in ax.spines.values():
                spine.set_color("#1f2742")
            ax.tick_params(colors="#e6edf3")
            x_abs = max(abs(sv_sorted.max()), abs(sv_sorted.min())) * 1.3 or 0.1
            ax.set_xlim(-x_abs, x_abs)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

        # ---- Downloadable report ----
        st.divider()
        st.subheader("📥 Download Report")

        # Build a tidy CSV of inputs + prediction + SHAP values
        report = raw.T.reset_index()
        report.columns = ["Field", "Value"]
        contribs_df = pd.DataFrame({
            "Field": [f"SHAP_{c}" for c in feature_cols],
            "Value": contributions,
        })
        verdict_df = pd.DataFrame({
            "Field": ["default_probability", "risk_tier"],
            "Value": [round(default_prob, 4), risk_label(default_prob)[1]],
        })
        full_report = pd.concat([verdict_df, report, contribs_df], ignore_index=True)

        # to_csv() returns a string; encode it to bytes for the download button
        csv_bytes = full_report.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="⬇️  Download applicant report (CSV)",
            data=csv_bytes,
            file_name="loan_risk_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

        # ---- Snapshot card at the bottom ----
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
            st.metric("Loan Burden",   f"{loan_percent_income:.1%}")


# =============================================================
# TAB 2 — BATCH UPLOAD  (score many applicants from a CSV)
# =============================================================
with tab_batch:
    st.subheader("Batch Loan Scoring")
    st.markdown(
        "Upload a CSV file with **the same columns used during training** "
        "to score many applicants at once.  Each row gets a default-probability "
        "and a risk tier, which you can then download."
    )

    # Show the user exactly which columns we need
    with st.expander("📋  Required columns (click to expand)"):
        st.code("\n".join(feature_cols), language="text")
        # Build a small template CSV they can download
        template_row = {c: "" for c in feature_cols}
        # Fill realistic placeholder defaults so the template is more useful
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

            # Check that all required columns are present
            missing = [c for c in feature_cols if c not in df_batch.columns]
            if missing:
                st.error(
                    f"Missing required columns: {missing}.  "
                    "Download the template above to see the expected format."
                )
            else:
                # Run the model on every row
                X_enc_batch = encode_row(df_batch)
                probs       = model.predict_proba(X_enc_batch)[:, 1]

                df_out = df_batch.copy()
                df_out["default_probability"] = probs.round(4)
                df_out["risk_tier"] = pd.cut(
                    probs,
                    bins   = [-0.01, 0.40, 0.65, 1.01],
                    labels = ["LOW", "MEDIUM", "HIGH"],
                )

                # Summary metrics at the top of the results
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total Applicants", f"{len(df_out):,}")
                m2.metric("🟢 Low Risk",   int((df_out['risk_tier'] == 'LOW').sum()))
                m3.metric("🟡 Medium Risk", int((df_out['risk_tier'] == 'MEDIUM').sum()))
                m4.metric("🔴 High Risk",  int((df_out['risk_tier'] == 'HIGH').sum()))

                st.dataframe(df_out, use_container_width=True, hide_index=True)

                # Let the user download the scored CSV
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
# TAB 3 — MODEL INSIGHTS  (global feature importance + metrics)
# =============================================================
with tab_insights:
    st.subheader("How the Model Decides")
    st.markdown(
        "These charts come from the **training run** and show the model's "
        "*overall* behaviour — across thousands of applicants, not just one."
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Test ROC-AUC", f"{roc_auc:.4f}" if roc_auc else "n/a")
    c2.metric("Algorithm", "Random Forest")
    c3.metric("Trees / Max depth", "200 / 8")

    st.divider()

    chart_col1, chart_col2 = st.columns(2)

    # Global SHAP feature importance (saved by train.py)
    with chart_col1:
        st.markdown("#### Global Feature Importance (SHAP)")
        if os.path.exists("shap_importance.png"):
            st.image("shap_importance.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate shap_importance.png")

    # Confusion matrix (saved by train.py)
    with chart_col2:
        st.markdown("#### Confusion Matrix (test set)")
        if os.path.exists("confusion_matrix.png"):
            st.image("confusion_matrix.png", use_container_width=True)
        else:
            st.info("Run `python train.py` to generate confusion_matrix.png")

    st.divider()
    st.markdown("#### Encoded Categories")
    st.caption(
        "The model sees integers, not strings.  These are the mappings learned from training data."
    )
    for col, cats in encoder_cats.items():
        st.markdown(f"**{nice_names.get(col, col)}** (`{col}`)")
        st.code({c: i for i, c in enumerate(cats)}, language="python")


# =============================================================
# TAB 4 — ABOUT
# =============================================================
with tab_about:
    st.subheader("About this project")
    st.markdown(
        """
        **Loan Default Predictor** is a portfolio project demonstrating a full
        end-to-end ML workflow:

        1. **Data**  — Credit-risk loan dataset (Kaggle) or generated equivalent
        2. **Modelling**  — Random Forest (200 trees, balanced class weights)
        3. **Evaluation**  — Stratified train/test split, ROC-AUC, precision, recall
        4. **Explainability**  — SHAP values per prediction *and* globally
        5. **Deployment**  — Streamlit Community Cloud (free hosting)

        ---

        **Tech stack**
        - `scikit-learn` — model training
        - `SHAP` — explainability
        - `Streamlit` + `Plotly` — interactive web UI
        - `pandas` / `numpy` — data wrangling
        - `joblib` — model serialization

        ---

        **Disclaimer**
        This app is for educational and portfolio purposes only.  Predictions
        are statistical estimates and must **not** be used to make real
        lending decisions.
        """
    )
    st.caption("Built with ❤️  using Python + Streamlit.")
