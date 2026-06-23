# 🏦 Loan Default Predictor — Deployed Web App

A machine-learning web app that predicts whether a loan applicant will default
and explains **why**, feature by feature, using SHAP.

**Live app:** https://loan-default-ml.streamlit.app

---

## What this project demonstrates

| Skill | How it shows up |
|---|---|
| Supervised classification | Random Forest (200 trees, `class_weight='balanced'`) |
| Categorical encoding | OrdinalEncoder converts text (grades, home ownership) to numbers |
| Evaluation | ROC-AUC, precision, recall, confusion matrix |
| Explainability | Per-applicant SHAP bar chart |
| **Deployment** | Streamlit web app — live public URL on Streamlit Community Cloud |

---

## Dataset

**Option A (recommended for portfolio quality):** Download the real dataset from Kaggle:

[https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)

Save the file as `credit_risk_dataset.csv` in this folder, then run `python train.py`.

**Option B (works out of the box):** If no CSV is found, `train.py` automatically
generates 5,000 synthetic-but-realistic loan records with the same column structure.

---

## Run it locally

> Only type the part **after** the `>`. Use Python 3.12 (some ML packages
> don't have 3.14 wheels yet).

```
cd C:\loan-default-app
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

Your browser opens automatically at `http://localhost:8501`.

---

## Deploy for free (public URL)

1. **Push to GitHub**
   ```
   git init
   git add app.py train.py requirements.txt README.md .gitignore model.pkl
   git commit -m "Initial loan default predictor"
   git remote add origin https://github.com/YOUR-USERNAME/loan-default-app.git
   git push -u origin main
   ```
   (Use a GitHub Personal Access Token as your password when prompted.)

2. **Connect to Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
   - Click **New app** → select this repo → set main file: `app.py`.
   - In Advanced Settings, set Python version to **3.12**.
   - Click **Deploy!**

3. **Copy the URL** into the "Live app" line at the top of this README, commit, and share it.

---

## Project structure

```
loan-default-app/
├── train.py              # trains + evaluates the model, saves model.pkl
├── app.py                # Streamlit web app (form → score → SHAP chart)
├── model.pkl             # saved model bundle — commit this to GitHub!
├── requirements.txt      # pinned package versions
├── README.md
└── .gitignore
```

---

## How the two scripts connect

`train.py` generates (or loads) loan data, encodes categorical columns with
`OrdinalEncoder`, trains a `RandomForestClassifier`, evaluates it on a
held-out 20% test set, and saves **one bundle** (`model.pkl`) containing
the model, the encoder, the feature order, and the ROC-AUC score.

`app.py` loads that single bundle. When the user submits the form, it
encodes their text inputs with the **same encoder**, feeds the result to
the model, and passes the prediction to SHAP to produce the per-feature
explanation chart.

---

_Educational / portfolio project — predictions are model estimates, not real lending decisions._
