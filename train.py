# =============================================================
# train.py  — Loan Default Predictor
# =============================================================
# PURPOSE: Train a Random Forest model that predicts whether
#          a loan applicant will default (fail to repay).
#
# RUN ONCE, locally, with:   python train.py
#
# OUTPUT:
#   model.pkl           — saved model (the web app loads this)
#   confusion_matrix.png
#   shap_importance.png
# =============================================================

# ---- 1. IMPORTS ------------------------------------------------
import os
import numpy as np          # math / array operations
import pandas as pd         # tables of data (DataFrames)
import joblib               # saves/loads Python objects to disk

import matplotlib
matplotlib.use('Agg')       # use non-interactive backend (no pop-up window)
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split      # split data into train/test
from sklearn.ensemble         import RandomForestClassifier # our model
from sklearn.preprocessing    import OrdinalEncoder         # turns text -> numbers
from sklearn.metrics import (
    roc_auc_score,            # main evaluation metric  (0.5 = random, 1.0 = perfect)
    classification_report,    # precision, recall, F1 for each class
    confusion_matrix,         # counts of correct / wrong predictions
    ConfusionMatrixDisplay,   # fancy plot of the confusion matrix
)
import shap                   # explains WHY the model made each prediction

print("=" * 60)
print("LOAN DEFAULT PREDICTOR — Training Script")
print("=" * 60)

# ============================================================
# STEP 1: LOAD OR GENERATE DATA
# ============================================================
# We look for the real Kaggle CSV first.  If it is not there,
# we generate 5,000 synthetic (fake-but-realistic) loan rows.
# The column names match the Kaggle dataset exactly, so you
# can drop in the real file anytime and it will just work.
#
# Real dataset (optional):
#   https://www.kaggle.com/datasets/laotse/credit-risk-dataset
#   Download -> save as credit_risk_dataset.csv in this folder

CSV_PATH = "credit_risk_dataset.csv"

if os.path.exists(CSV_PATH):
    # ---- Load the real Kaggle CSV ----
    print(f"\n[DATA] Real dataset found -> loading {CSV_PATH}")
    df = pd.read_csv(CSV_PATH)

else:
    # ---- Generate synthetic data ----
    print("\n[DATA] No CSV found — generating 5,000 synthetic loan records.")
    print("       Tip: place 'credit_risk_dataset.csv' here to train on real data.\n")

    np.random.seed(42)   # set a fixed random seed so results are reproducible
    N = 5000             # number of fake loan applications to create

    # Helper: sigmoid converts a raw "risk score" into a probability 0–1
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    # ---- Person-level features ----
    person_age = np.random.normal(32, 10, N).clip(18, 80).astype(int)

    # log-normal income: most people earn $30k–$80k, a few earn much more
    person_income = np.random.lognormal(10.5, 0.7, N).clip(10_000, 500_000).astype(int)

    person_home_ownership = np.random.choice(
        ['RENT', 'MORTGAGE', 'OWN', 'OTHER'], N, p=[0.50, 0.35, 0.12, 0.03]
    )

    # employment length in years; exponential so 0-5 yrs is most common
    person_emp_length = np.round(np.random.exponential(5, N).clip(0, 41), 1)

    # ---- Loan-level features ----
    loan_intent = np.random.choice(
        ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
        N, p=[0.20, 0.15, 0.17, 0.12, 0.16, 0.20]
    )

    # Grade A = best creditworthiness -> low interest; Grade G = worst -> high interest
    loan_grade = np.random.choice(
        ['A', 'B', 'C', 'D', 'E', 'F', 'G'],
        N, p=[0.22, 0.28, 0.20, 0.15, 0.10, 0.03, 0.02]
    )

    loan_amnt = np.random.lognormal(8.5, 0.8, N).clip(500, 35_000).astype(int)

    # Interest rate is correlated with grade (worse grade -> higher rate)
    grade_to_rate = {'A': 5.5, 'B': 9.0, 'C': 12.5, 'D': 15.5, 'E': 19.0, 'F': 23.0, 'G': 27.0}
    base_rate = np.array([grade_to_rate[g] for g in loan_grade])
    loan_int_rate = np.round((base_rate + np.random.normal(0, 1.5, N)).clip(5.0, 35.0), 2)

    # loan_percent_income = what fraction of annual income is the loan?
    loan_percent_income = np.round(loan_amnt / person_income, 2).clip(0.0, 0.8)

    # Credit bureau history
    cb_person_default_on_file  = np.random.choice(['Y', 'N'], N, p=[0.17, 0.83])
    cb_person_cred_hist_length = np.random.randint(2, 30, N)

    # ---- Build a realistic default probability ----
    # Each factor nudges risk up or down; we convert to 0/1 with sigmoid.
    grade_idx = np.array([['A','B','C','D','E','F','G'].index(g) for g in loan_grade])

    risk_score = (
          grade_idx * 0.35                                        # worse grade  -> more risk
        + (cb_person_default_on_file == 'Y').astype(int) * 1.2  # past default  -> big risk
        + loan_percent_income * 2.5                              # high burden   -> more risk
        + (loan_int_rate / 35) * 0.6                            # high rate      -> more risk
        - np.log(person_income / 50_000) * 0.3                  # high income   -> less risk
        - (person_emp_length / 10) * 0.2                        # long employed -> less risk
        + np.random.normal(0, 0.9, N)                           # random noise
    )

    # Convert risk score -> default probability; subtract 2.5 to get ~20% default rate
    default_prob   = sigmoid(risk_score - 2.5)
    loan_status    = (np.random.random(N) < default_prob).astype(int)  # 1 = default

    # ---- Assemble DataFrame ----
    df = pd.DataFrame({
        'person_age':                 person_age,
        'person_income':              person_income,
        'person_home_ownership':      person_home_ownership,
        'person_emp_length':          person_emp_length,
        'loan_intent':                loan_intent,
        'loan_grade':                 loan_grade,
        'loan_amnt':                  loan_amnt,
        'loan_int_rate':              loan_int_rate,
        'loan_status':                loan_status,          # <-- TARGET (1=default, 0=no default)
        'loan_percent_income':        loan_percent_income,
        'cb_person_default_on_file':  cb_person_default_on_file,
        'cb_person_cred_hist_length': cb_person_cred_hist_length,
    })

print(f"Rows in dataset  : {len(df):,}")
print(f"Default rate     : {df['loan_status'].mean():.1%}")
print(f"Columns          : {list(df.columns)}")
print()

# ============================================================
# STEP 2: SEPARATE FEATURES (X) AND TARGET (y)
# ============================================================
# X = the 11 input columns the model uses
# y = the column we are trying to predict (1=default, 0=no default)

TARGET = 'loan_status'

# These are text/category columns — we will encode them as numbers
CATEGORICAL_COLS = [
    'person_home_ownership',
    'loan_intent',
    'loan_grade',
    'cb_person_default_on_file',
]

# These are already numbers — no encoding needed
NUMERICAL_COLS = [
    'person_age',
    'person_income',
    'person_emp_length',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
]

# The final feature list (numerical first, then categorical)
# IMPORTANT: the app must use features in this EXACT order
FEATURE_COLS = NUMERICAL_COLS + CATEGORICAL_COLS

X = df[FEATURE_COLS].copy()
y = df[TARGET].copy()

print(f"[STEP 2] Features: {FEATURE_COLS}")
print(f"[STEP 2] Target  : '{TARGET}' (0=no default, 1=default)")

# ============================================================
# STEP 3: ENCODE CATEGORICAL COLUMNS AS NUMBERS
# ============================================================
# Random Forest cannot handle text like "RENT" or "A".
# OrdinalEncoder replaces each unique text value with an integer.
#   e.g.  "MORTGAGE" -> 0,  "OTHER" -> 1,  "OWN" -> 2,  "RENT" -> 3
# We save the encoder so the web app can apply the SAME mapping
# when the user types in their choices.

print("\n[STEP 3] Encoding categorical columns...")

encoder = OrdinalEncoder(
    handle_unknown='use_encoded_value',
    unknown_value=-1    # if a user somehow enters a value we've never seen, use -1
)
X[CATEGORICAL_COLS] = encoder.fit_transform(X[CATEGORICAL_COLS])

# Print the mappings so you can see what each number means
for col, cats in zip(CATEGORICAL_COLS, encoder.categories_):
    mapping = {cat: i for i, cat in enumerate(cats)}
    print(f"  {col}: {mapping}")

# ============================================================
# STEP 4: TRAIN / TEST SPLIT
# ============================================================
# We hold out 20% of the data to evaluate the model fairly.
# stratify=y means both splits have the same default rate —
# otherwise a lucky split could skew our numbers.

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size   = 0.20,
    random_state= 42,
    stratify    = y
)
print(f"\n[STEP 4] Train rows: {len(X_train):,}  |  Test rows: {len(X_test):,}")

# ============================================================
# STEP 5: TRAIN THE RANDOM FOREST
# ============================================================
# A Random Forest is a collection of Decision Trees.
# Each tree votes; the majority vote is the final prediction.

print("\n[STEP 5] Training RandomForestClassifier (this may take 10–30 seconds)...")

model = RandomForestClassifier(
    n_estimators  = 200,      # 200 individual trees in the forest
    max_depth     = 8,        # each tree grows at most 8 levels deep (prevents overfitting)
    min_samples_leaf = 20,    # each leaf must contain ≥ 20 training samples
    class_weight  = 'balanced', # compensates for the imbalance (fewer defaults than non-defaults)
    random_state  = 42,
    n_jobs        = -1        # use all available CPU cores for speed
)

model.fit(X_train, y_train)
print("[STEP 5] Training complete!")

# ============================================================
# STEP 6: EVALUATE THE MODEL
# ============================================================
# predict_proba gives a PROBABILITY (0.0–1.0) of default.
# predict gives the hard label (0 or 1) using 0.5 threshold.

y_pred_proba = model.predict_proba(X_test)[:, 1]  # column 1 = P(default)
y_pred       = model.predict(X_test)

roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n[STEP 6] ROC-AUC Score : {roc_auc:.4f}")
print("         (0.5 = random guessing, 1.0 = perfect, > 0.70 is good)")
print()
print("[STEP 6] Classification Report:")
print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

# ---- Plot the confusion matrix ----
# Rows = actual class, Columns = predicted class
# Top-left (TN) = correctly predicted no-default
# Bottom-right (TP) = correctly predicted default

cm   = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Default", "Default"])
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=100)
plt.close()
print("[STEP 6] Confusion matrix plot saved -> confusion_matrix.png")

# ============================================================
# STEP 7: SHAP FEATURE IMPORTANCE
# ============================================================
# SHAP (SHapley Additive exPlanations) tells us which features
# the model relies on most, on average across many predictions.

print("\n[STEP 7] Computing SHAP values (30–60 seconds)...")

# Use a 500-row sample from the training set (full dataset would be slow)
sample_size = min(500, len(X_train))
idx_sample  = np.random.choice(len(X_train), size=sample_size, replace=False)
X_sample    = X_train.iloc[idx_sample]

explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample.values)  # pass as numpy array

# For a binary classifier, shap_values is a list: [class0_vals, class1_vals]
# We care about class 1 (default)
if isinstance(shap_values, list):
    sv_class1 = shap_values[1]   # shape: (n_samples, n_features)
else:
    sv_class1 = shap_values[:, :, 1]

# Mean absolute SHAP value per feature = average importance
mean_shap  = np.abs(sv_class1).mean(axis=0)
sorted_idx = np.argsort(mean_shap)   # sort lowest to highest so chart reads top-down

# Human-readable labels (same order as FEATURE_COLS)
NICE_NAMES = {
    'person_age':                 'Age',
    'person_income':              'Annual Income',
    'person_emp_length':          'Employment Length',
    'loan_amnt':                  'Loan Amount',
    'loan_int_rate':              'Interest Rate',
    'loan_percent_income':        'Loan / Income Ratio',
    'cb_person_cred_hist_length': 'Credit History Length',
    'person_home_ownership':      'Home Ownership',
    'loan_intent':                'Loan Purpose',
    'loan_grade':                 'Loan Grade',
    'cb_person_default_on_file':  'Past Default on File',
}
feature_labels = [NICE_NAMES.get(c, c) for c in FEATURE_COLS]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(
    np.array(feature_labels)[sorted_idx],
    mean_shap[sorted_idx],
    color='steelblue'
)
ax.set_xlabel("Mean |SHAP Value|  (larger = feature matters more)")
ax.set_title("Feature Importance (SHAP) — averaged over 500 training samples")
plt.tight_layout()
plt.savefig("shap_importance.png", dpi=100)
plt.close()
print("[STEP 7] SHAP feature importance saved -> shap_importance.png")

# ============================================================
# STEP 8: SAVE THE MODEL ARTIFACT
# ============================================================
# We bundle everything the web app needs into one .pkl file.
# joblib.dump() serialises it (turns it into bytes on disk).

artifact = {
    'model'             : model,          # the trained RandomForestClassifier
    'encoder'           : encoder,        # OrdinalEncoder fitted on training data
    'feature_cols'      : FEATURE_COLS,   # exact feature order (MUST match app.py)
    'categorical_cols'  : CATEGORICAL_COLS,
    'numerical_cols'    : NUMERICAL_COLS,
    'encoder_categories': {               # human-readable: which values each col can take
        col: list(cats)
        for col, cats in zip(CATEGORICAL_COLS, encoder.categories_)
    },
    'nice_names'        : NICE_NAMES,
    'roc_auc'           : roc_auc,
}

joblib.dump(artifact, 'model.pkl')

print(f"\n[STEP 8] model.pkl saved  (ROC-AUC = {roc_auc:.4f})")
print()
print("=" * 60)
print("All done!  Next step: run the web app:")
print("   streamlit run app.py")
print("=" * 60)
