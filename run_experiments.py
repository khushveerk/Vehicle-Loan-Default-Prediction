#!/usr/bin/env python3
"""
Vehicle-Loan Default Prediction for First-Time Borrowers
DU B.Sc. (H) CS - Semester VI Machine Learning Assignment (Problem 42)
Author: Khushveer Kaur (Roll No. 24020570052)
"""

import os
import re
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

# 1. Load Dataset
CSV_FILE = "train.csv"
if not os.path.exists(CSV_FILE):
    print(f"Error: '{CSV_FILE}' not found in the current folder.")
    exit(1)

print("[1/5] Loading and cleaning raw dataset...")
df = pd.read_csv(CSV_FILE)
print(f"  Loaded {len(df):,} records across {len(df.columns)} columns.")

# Automatically find description column regardless of dot or underscore
desc_col = next((c for c in df.columns if "DESCRIPTION" in c.upper()), None)
score_col = next((c for c in df.columns if "CNS.SCORE" in c.upper() or "CNS_SCORE" in c.upper()), "PERFORM_CNS.SCORE")
inquiry_col = next((c for c in df.columns if "INQUIRIES" in c.upper()), "NO.OF_INQUIRIES")
delinq_col = next((c for c in df.columns if "DELINQUENT" in c.upper()), "DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS")
new_accts_col = next((c for c in df.columns if "NEW.ACCTS" in c.upper() or "NEW_ACCTS" in c.upper()), "NEW.ACCTS.IN.LAST.SIX.MONTHS")
avg_age_col = next((c for c in df.columns if "AVERAGE.ACCT.AGE" in c.upper() or "AVERAGE_ACCT_AGE" in c.upper()), "AVERAGE.ACCT.AGE")
credit_len_col = next((c for c in df.columns if "CREDIT.HISTORY.LENGTH" in c.upper() or "CREDIT_HISTORY_LENGTH" in c.upper()), "CREDIT.HISTORY.LENGTH")

# Helper to parse string durations (e.g., '2yrs 5mon') to months
def parse_duration_to_months(text):
    if pd.isna(text):
        return 0
    yrs = re.findall(r"(\d+)yrs", str(text))
    mon = re.findall(r"(\d+)mon", str(text))
    y = int(yrs[0]) if yrs else 0
    m = int(mon[0]) if mon else 0
    return y * 12 + m

# Date calculations & Age
dob_col = next((c for c in df.columns if "BIRTH" in c.upper()), "Date.of.Birth")
df[dob_col] = pd.to_datetime(df[dob_col], format="%d-%m-%y", errors="coerce")
birth_year = df[dob_col].dt.year
birth_year = np.where(birth_year > 2018, birth_year - 100, birth_year)
df["Age"] = 2018 - birth_year
df["Age"] = df["Age"].fillna(df["Age"].median())

# Duration features
df["Avg_Loan_Tenure_Months"] = df[avg_age_col].apply(parse_duration_to_months)
df["Credit_History_Months"] = df[credit_len_col].apply(parse_duration_to_months)

# Identify Thin-File / First-Time Borrowers
if desc_col:
    df["Is_Thin_File"] = df[desc_col].astype(str).str.contains(
        "No Bureau History|Not Scored", case=False, na=False
    ).astype(int)
else:
    df["Is_Thin_File"] = (df[score_col] <= 0).astype(int)

# Employment encoding
emp_col = next((c for c in df.columns if "EMPLOYMENT" in c.upper()), "Employment.Type")
df[emp_col] = df[emp_col].fillna("Unemployed_or_Self")
df["Employment_Salaried"] = (df[emp_col] == "Salaried").astype(int)
df["Employment_SelfEmployed"] = (df[emp_col] == "Self employed").astype(int)

# Feature Subset Selection
numeric_features = [
    "disbursed_amount", "asset_cost", "ltv", "Age",
    score_col, "PRI.NO.OF.ACCTS", "PRI.ACTIVE.ACCTS", "PRI.OVERDUE.ACCTS",
    "PRI.CURRENT.BALANCE", "PRI.SANCTIONED.AMOUNT", "PRI.DISBURSED.AMOUNT",
    "PRIMARY.INSTAL.AMT", new_accts_col, delinq_col,
    inquiry_col, "Avg_Loan_Tenure_Months", "Credit_History_Months",
    "Employment_Salaried", "Employment_SelfEmployed", "Is_Thin_File"
]

# Ensure features exist in dataframe
selected_features = [c for c in numeric_features if c in df.columns]
target_col = next((c for c in df.columns if "DEFAULT" in c.upper()), "loan_default")

X = df[selected_features].copy().fillna(df[selected_features].median())
y = df[target_col].copy()

# 2. Split (80:20)
print("[2/5] Stratified train-test partitioning (80:20)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Model Training
print("[3/5] Training 7 syllabus-prescribed classifiers...")
models = {
    "Baseline (Prior Default)": DummyClassifier(strategy="stratified", random_state=42),
    "Logistic Regression (balanced)": LogisticRegression(C=0.1, penalty="l2", class_weight="balanced", max_iter=500, random_state=42),
    "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-9),
    "Decision Tree (depth=6, balanced)": DecisionTreeClassifier(max_depth=6, min_samples_leaf=50, class_weight="balanced", random_state=42),
    "k-NN (k=51, sub=30k)": KNeighborsClassifier(n_neighbors=51, metric="euclidean", n_jobs=-1),
    "Perceptron (balanced)": Perceptron(alpha=1e-3, class_weight="balanced", random_state=42),
    "MLP Classifier (64-32 units)": MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-4, max_iter=25, early_stopping=True, random_state=42),
}

knn_sub = np.random.RandomState(42).choice(len(X_train_scaled), size=30000, replace=False)
results = []
roc_data = {}

for name, clf in models.items():
    print(f"  Fitting {name}...")
    if "k-NN" in name:
        clf.fit(X_train_scaled[knn_sub], y_train.iloc[knn_sub])
    else:
        clf.fit(X_train_scaled, y_train)

    y_pred = clf.predict(X_test_scaled)
    if hasattr(clf, "predict_proba"):
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
    elif hasattr(clf, "decision_function"):
        df_vals = clf.decision_function(X_test_scaled)
        y_prob = (df_vals - df_vals.min()) / (df_vals.max() - df_vals.min() + 1e-8)
    else:
        y_prob = y_pred

    results.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 3),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 3),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 3),
        "F1-Score": round(f1_score(y_test, y_pred, zero_division=0), 3),
        "ROC-AUC": round(roc_auc_score(y_test, y_prob), 3),
    })
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_data[name] = (fpr, tpr, roc_auc_score(y_test, y_prob))

print("\n" + "="*80)
print("TEST BENCHMARK RESULTS (n = 46,631):")
print("="*80)
print(pd.DataFrame(results).to_string(index=False))

# 4. Fairness Audit
print("\n[4/5] Executing Algorithmic Fairness Audit...")
best_model = models["Logistic Regression (balanced)"]
test_preds = best_model.predict(X_test_scaled)

thin_mask = (X_test["Is_Thin_File"] == 1).values
y_thin, p_thin = y_test.iloc[thin_mask], test_preds[thin_mask]
y_scored, p_scored = y_test.iloc[~thin_mask], test_preds[~thin_mask]

fpr_thin = np.sum((y_thin == 0) & (p_thin == 1)) / np.sum(y_thin == 0)
fpr_scored = np.sum((y_scored == 0) & (p_scored == 1)) / np.sum(y_scored == 0)

fairness_df = pd.DataFrame({
    "Cohort": ["Thin-File (First-Time)", "Established (Scored)"],
    "Sample Size (n)": [len(y_thin), len(y_scored)],
    "Actual Default Rate": [f"{y_thin.mean():.1%}", f"{y_scored.mean():.1%}"],
    "Model Flagged Default / Denial Rate": [f"{p_thin.mean():.1%}", f"{p_scored.mean():.1%}"],
    "False Positive Rate (FPR)": [f"{fpr_thin:.1%}", f"{fpr_scored:.1%}"]
})
print(fairness_df.to_string(index=False))

# 5. Save Plot
print("\n[5/5] Generating plots...")
plt.figure(figsize=(7, 5), dpi=300)
for name, (fpr, tpr, auc) in roc_data.items():
    ls = "--" if "Baseline" in name else "-"
    plt.plot(fpr, tpr, linestyle=ls, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", alpha=0.3)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves on Holdout Test Set")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig("fig3_roc_curves.png")
plt.close()
print("  Saved 'fig3_roc_curves.png'")

print("\nExecution complete! Everything ran successfully.")