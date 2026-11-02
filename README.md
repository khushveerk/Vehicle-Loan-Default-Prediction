   # Vehicle-Loan Default Prediction for First-Time Borrowers: A Credit-Risk Benchmark and Financial Inclusion Audit


**Course:** Machine Learning (B.Sc. (H) Computer Science, Semester VI) — University of Delhi  
**Author:** Khushveer Kaur   
**SDG Focus:** SDG 1 (No Poverty) & SDG 8 (Decent Work and Economic Growth)  
**Article Type:** Applied / Academic Research Paper (Journal Format)

---


## 📌 Executive Summary

Vehicle financing (two-wheelers, auto-rickshaws, and light commercial vehicles) serves as a primary gateway into formal credit for millions of low-income Indian households. However, first-time borrowers frequently lack formal bureau credit histories ("thin-file" borrowers), creating acute information asymmetry that prompts lenders to enforce blanket credit rationing.

This project implements an end-to-end, leakage-free machine learning benchmark across **233,154 loan records** from the **L&T Financial Services Vehicle Loan Default Challenge**. It evaluates 7 classical syllabus classifiers, analyzes risk drivers, and performs a quantitative **algorithmic fairness audit** comparing thin-file applicants against bureau-scored borrowers under the Four-Fifths Fair Lending Doctrine.

---

## 📊 Dataset & Preprocessing

- **Dataset Source:** [Kaggle - L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction)
- **Cohort Size:** 233,154 records across 41 raw operational features.
- **Target Variable:** `loan_default` (Binary: 1 = defaulted on first EMI, 0 = non-default).
- **Class Imbalance:** 21.7% default base rate (~1:3.6 default-to-non-default ratio).

### Key Preprocessing Steps
1. **Missing Data Imputation:** Missingness profile analyzed (Fig. 1); only `Employment.Type` exhibited missing values (3.28%), imputed as `Unemployed_or_Self` and dummy-encoded.
2. **Century Rollover Correction:** 2-digit birth dates (`dd-mm-yy`) corrected to compute exact borrower `Age`.
3. **Duration Parsing:** `AVERAGE.ACCT.AGE` and `CREDIT.HISTORY.LENGTH` strings parsed via regular expressions into integer month equivalents.
4. **Thin-File Segmentation:** Extracted binary indicator `Is_Thin_File` identifying applicants flagged with zero bureau credit history or "Not Scored".
5. **Leakage-Free Protocol:** Stratified 80:20 split (Train: $n = 186,523$, Test: $n = 46,631$). `StandardScaler` fitted strictly on the training partition only.

---

## 🔬 Experimental Methodology

All modeling adheres strictly to classical supervised machine learning paradigms from the undergraduate syllabus:

1. **Stratified Prior Baseline** (Empirical probability benchmark)
2. **Logistic Regression** (L2 regularized, cost-sensitive balanced class weighting)
3. **Gaussian Naive Bayes** (Numerical variance floor smoothing `var_smoothing` $= 10^{-9}$)
4. **Decision Tree Classifier** (Pre-pruning: `max_depth=6`, `min_samples_leaf=50`, balanced weights)
5. **$k$-Nearest Neighbours** ($k=51$, Euclidean metric, stratified subsample $n=30\text{k}$)
6. **Single-Layer Perceptron** (Mistake-driven threshold updates with L2 penalty)
7. **Multilayer Perceptron (MLP)** (Feedforward architecture with 64–32 ReLU hidden units, early stopping)

---

## 📈 Key Results & Performance Benchmark

Evaluation conducted on the independent holdout test partition ($n = 46,631$):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Prior Baseline** | 0.660 | 0.215 | 0.213 | 0.214 | 0.498 |
| **Logistic Regression (balanced)** | 0.556 | 0.274 | **0.634** | **0.383** | **0.618** |
| **Gaussian Naive Bayes** | 0.772 | 0.295 | 0.035 | 0.062 | 0.592 |
| **Decision Tree (depth=6, balanced)** | 0.560 | 0.275 | **0.631** | **0.383** | **0.611** |
| **$k$-NN Classifier ($k=51$)** | 0.783 | 0.407 | 0.003 | 0.007 | 0.598 |
| **Perceptron (balanced)** | 0.570 | 0.243 | 0.464 | 0.319 | 0.550 |
| **MLP Classifier (64–32 units)** | **0.783** | **0.577** | 0.004 | 0.008 | **0.622** |

> **Key Finding:** Unweighted models (MLP, $k$-NN) suffer from near-total minority default omission (Recall $< 0.01$). Balanced Logistic Regression and shallow Decision Trees achieve the best clinical sensitivity, detecting over **63% of early-instalment defaults**.

---

## ⚖️ Algorithmic Fairness & Inclusion Audit

Subgroup analysis comparing Thin-File (unbanked, first-time, $n = 26,010$) versus Established (bureau-scored, $n = 20,621$) applicants under the tuned Logistic Regression classifier:

| Metric | Thin-File (First-Time) | Established (Bureau-Scored) |
| :--- | :---: | :---: |
| **Sample Size ($n$)** | 26,010 | 20,621 |
| **Observed True Default Rate** | **22.9%** | **20.2%** |
| **Model Flagged Rejection Rate** | **62.1%** | **35.2%** |
| **False Positive Rate (FPR)** | **59.2%** | **31.1%** |
| **Selection (Approval) Rate** | **37.9%** | **64.8%** |
| **Disparate Impact Ratio (DIR)** | **0.585** *(Severe violation of 80% rule)* | 1.000 *(Reference)* |

### The Inclusion Paradox & Fair Lending Violation
- **The Paradox:** Thin-file borrowers exhibited a delinquency rate of 22.9%, yet credit scoring models penalize the absence of credit scores, generating an unwarranted **62.1% rejection rate** and a **59.2% False Positive Rate**.
- **Four-Fifths Rule Violation:** The Disparate Impact Ratio ($\text{DIR} = 37.9\% / 64.8\% = \mathbf{0.585}$) breaches the regulatory **0.80 benchmark**, demonstrating that uncalibrated algorithmic scoring institutionalizes credit rationing against unbanked applicants.

---

##  Visualizations

The repository generates publication-grade figures:

-  Missing value profile across raw features.
  <img width="1950" height="960" alt="Image" src="https://github.com/user-attachments/assets/94567bbe-967b-45da-9e9e-48a818585a8b" />
  
-  End-to-End 6-stage leakage-free methodology flowchart.
 <img width="2160" height="1260" alt="Image" src="https://github.com/user-attachments/assets/1a2b440d-c4b7-4f99-adcb-819a6f25e6db" />
 
- **`fig3_roc_curves.png`**: ROC curves comparing all 7 classifiers against the baseline.
- **`fig4_fairness.png`**: Subgroup error & approval rate disparity with DIR callout.

---

## 🚀 How to Run the Project

### Prerequisites
Install Python 3.10+ and the required scientific packages:
```bash
pip install numpy pandas matplotlib scikit-learn python-docx reportlab
```

### Execution
1. Place `train.csv` from the Kaggle dataset in your root directory.
2. Run the complete experiment pipeline:
   ```bash
   python run_experiments.py
   ```
3. To generate all high-resolution figures standalone:
   ```bash
   python generate_all_plots.py
   ```

---

## 📂 Repository Structure

```text
├── run_experiments.py             # Complete ML training, evaluation & metrics pipeline
├── generate_all_plots.py          # Standalone 400 DPI figure generation script
├── train.csv                      # L&T challenge training dataset (Kaggle)
├── fig1_missingness.png           # Data missingness profile chart
├── fig2_workflow.png              # Methodological architecture flowchart
├── fig3_roc_curves.png            # Model ROC comparative curve
├── fig4_fairness.png              # Algorithmic fairness & disparity audit chart
├── Khushveer_Kaur_Problem_42...pdf # Final 6-page University journal submission paper
└── README.md                      # Project documentation and reproduction guide
```

---

## 📜 References & Compliance

1. F. Pedregosa et al., *"Scikit-learn: Machine learning in Python,"* Journal of Machine Learning Research, vol. 12, pp. 2825–2830, 2011.
2. T. Hastie, R. Tibshirani, and J. Friedman, *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*, 2nd ed. Springer, 2009.
3. Reserve Bank of India, *"Report of the Working Group on Digital Lending,"* RBI Bulletin, Nov. 2021.
4. M. Hardt, E. Price, and N. Srebro, *"Equality of opportunity in supervised learning,"* NeurIPS, pp. 3315–3323, 2016.
5. L&T Financial Services, *"Vehicle Loan Default Prediction Challenge Data,"* Kaggle Competition Dataset, 2019.
