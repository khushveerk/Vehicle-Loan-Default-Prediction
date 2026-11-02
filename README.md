# Vehicle Loan Default Prediction for First-Time Borrowers
### A Credit-Risk Benchmark and Financial Inclusion Audit Using Classical Machine Learning

An applied machine learning project that studies early vehicle-loan default prediction in India, benchmarks classical classification algorithms, and examines differences in model outcomes between first-time borrowers and borrowers with established credit histories.

**Course:** Machine Learning — B.Sc. (Hons.) Computer Science, Semester VI  
**Institution:** University of Delhi  
**Author:** Khushveer Kaur  
**Problem:** No. 42  
**Dataset:** L&T Financial Services Vehicle Loan Default Prediction  
**Research themes:** Credit-risk prediction, class imbalance, financial inclusion and algorithmic fairness  
**SDG alignment:** SDG 1 — No Poverty; SDG 8 — Decent Work and Economic Growth

---

## 1. Project Overview

Vehicle financing can help households obtain productive assets such as two-wheelers, auto-rickshaws and light commercial vehicles. For people entering the formal credit system for the first time, however, the absence of an established credit-bureau history can make credit assessment difficult.

Such applicants are often described as **thin-file borrowers**. A conventional credit-scoring process may have less historical information with which to assess them. This project investigates whether classical machine learning classifiers can predict early vehicle-loan default and whether the evaluated model produces different outcomes for thin-file and established borrowers.

The experiment uses 233,154 records from the L&T Financial Services Vehicle Loan Default Prediction dataset. It compares six classical machine learning classifiers with a prior-probability baseline and evaluates the resulting predictions on a stratified holdout test set.

A separate subgroup audit compares observed default rates, model-flagged rejection rates, false positive rates and approval rates for thin-file and established borrowers.

### Research objectives

- Build a supervised binary-classification pipeline for vehicle-loan default prediction.
- Compare classical classifiers using multiple evaluation metrics.
- Examine how class imbalance affects the detection of default cases.
- Identify the model configurations that provide stronger default-class recall.
- Compare prediction outcomes for thin-file and established borrowers.
- Calculate the Disparate Impact Ratio (DIR) using subgroup approval rates.
- Discuss the implications of these results for financial inclusion.

### Project at a glance

| Component | Details |
|---|---|
| Dataset records | 233,154 |
| Raw attributes | 41 |
| Target | `loan_default` |
| Task | Binary classification |
| Default prevalence | Approximately 21.7% |
| Train-test split | Stratified 80:20 |
| Training records | 186,523 |
| Test records | 46,631 |
| Classifiers | 6, plus a prior baseline |
| Model selection | 5-fold stratified cross-validation |
| Fairness groups | Thin-file and established borrowers |
| Main fairness result | DIR = 0.585 |

---

## 2. Dataset

**Source:** [L&T Vehicle Loan Default Prediction — Kaggle](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction)

The dataset contains vehicle-loan records and operational attributes relating to borrowers, financed vehicles and credit histories.

### Target variable

The target is `loan_default`:

- `1` — the borrower defaulted on the first scheduled EMI.
- `0` — the borrower did not default on the first scheduled EMI.

The dataset is imbalanced: approximately 21.7% of records belong to the default class, while approximately 78.3% belong to the non-default class.

### Selected features

| Feature | Description |
|---|---|
| `disbursed_amount` | Loan amount disbursed to the dealer |
| `asset_cost` | Cost of the financed vehicle |
| `ltv` | Loan-to-Value ratio |
| `Age` | Borrower age derived from date of birth |
| `PERFORM_CNS.SCORE` | Credit-bureau score |
| `PRI.ACTIVE.ACCTS` | Number of active primary credit accounts |
| `DELINQUENT.ACCTS` | Delinquent credit-account information |
| `NO.OF_INQUIRIES` | Number of credit-bureau inquiries |
| `Is_Thin_File` | Derived indicator for borrowers without an established bureau history |
| `loan_default` | Binary prediction target |

---

## 3. Data Preprocessing and Feature Engineering

The raw dataset requires several transformations before model training.

### 3.1 Missing-value analysis

Missingness is examined across the raw variables. In the supplied dataset, `Employment.Type` is the feature identified with missing values, accounting for approximately 3.28%.

The missing category is assigned `Unemployed_or_Self` and subsequently dummy-encoded, as described in the project methodology.

![Missing-value profile](https://github.com/user-attachments/assets/94567bbe-967b-45da-9e9e-48a818585a8b)

*Figure 1. Percentage of missing values per variable in the raw training dataset.*

### 3.2 Date parsing and age calculation

The `Date.of.Birth` field uses a two-digit year representation. The century rollover is corrected before deriving borrower age so that dates are not incorrectly interpreted as future birth dates.

### 3.3 Account-duration conversion

The fields `AVERAGE.ACCT.AGE` and `CREDIT.HISTORY.LENGTH` contain duration strings such as `2yrs 5mon`. These are parsed into integer month values.

For example:
(2 × 12) + 5 = 29 months

### 3.4 Thin-file identification

The `Is_Thin_File` indicator is derived from `PERFORM_CNS.SCORE.DESCRIPTION`. Applicants described as having no bureau history available or as not scored are identified as thin-file borrowers.

This derived feature is used in the subgroup fairness analysis.

### 3.5 Train-test split and scaling

The records are divided using an 80:20 stratified split:

| Partition | Records |
|---|---:|
| Training | 186,523 |
| Testing | 46,631 |
| Total | 233,154 |

Stratification preserves the approximate class distribution in both partitions. Scaling statistics are fitted on the training data and then applied to the test data, rather than fitting the scaler on the complete dataset.

---

## 4. Machine Learning Workflow

The experiment follows a sequence of data preparation, model selection, evaluation and subgroup analysis.

1. **Load the dataset:** Read the L&T vehicle-loan records.
2. **Explore the data:** Examine feature types, missingness and target distribution.
3. **Preprocess:** Correct date and duration fields and handle missing values.
4. **Engineer features:** Derive borrower age and the thin-file indicator.
5. **Split the data:** Create stratified training and holdout test partitions.
6. **Train and select models:** Fit the classifiers and select configurations using the training process.
7. **Evaluate:** Compare predictions on the holdout test set.
8. **Audit subgroup outcomes:** Compare thin-file and established borrowers.

> **Figure update required:** The currently supplied workflow image contains an older DIR value (0.781). Replace that annotation with the latest result (0.585) before publishing the image as the current workflow figure.

---

## 5. Classification Models

Six classical supervised classifiers are evaluated. A prior baseline is included for comparison.

### 5.1 Prior baseline

The baseline uses the empirical class prior as a reference point. It provides a simple benchmark against which the trained classifiers can be compared.

### 5.2 Logistic Regression

Logistic Regression estimates the probability of default using a linear combination of input features and the sigmoid function:

\[
P(Y=1\mid X)=\frac{1}{1+e^{-z}}
\]

where \(z\) is the weighted sum of the input features and the intercept.

The evaluated configuration uses L2 regularization and balanced class weights.

### 5.3 Gaussian Naive Bayes

Gaussian Naive Bayes applies Bayes' theorem and models numerical features using Gaussian conditional distributions. The reported configuration uses `var_smoothing = 1e-9`.

### 5.4 Decision Tree

A Decision Tree recursively splits the feature space to form decision rules. The evaluated tree uses Gini impurity and the following constraints:

- `max_depth = 6`
- `min_samples_leaf = 50`
- Balanced class weights

### 5.5 K-Nearest Neighbours

KNN predicts a class using the labels of nearby training observations. The reported configuration uses:

- `k = 51`
- Euclidean distance
- Uniform neighbour weighting
- A stratified training subsample of 30,000 records

### 5.6 Perceptron

The Perceptron is a linear threshold classifier. Its weights are updated during training based on classification mistakes. The reported configuration uses balanced class weighting.

### 5.7 Multilayer Perceptron

The MLP is a feedforward neural network with two hidden layers containing 64 and 32 units. It uses ReLU hidden-layer activation, the Adam optimizer and early stopping, as specified in the experiment.

---

🧮 Understanding the Mathematical Concepts

This section explains the main mathematical ideas behind the project.

1. Logistic Regression and the Sigmoid Function

Logistic Regression calculates a linear combination of input features:

z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ

The sigmoid function converts this value into a probability:

P(Y = 1 | X) = 1 / (1 + e⁻ᶻ)

The resulting probability represents the model's estimated likelihood of default.

2. Gini Impurity

Gini impurity measures the degree of class mixing within a decision-tree node.

Gini = 1 - Σ(pᵢ²)

Where pᵢ represents the proportion of observations belonging to class i.

Gini = 0: The node contains only one class.

Higher Gini: The node contains a greater mixture of classes.

Decision Trees select splits that reduce impurity.

3. Bayesian Classification

Naive Bayes uses Bayes' theorem:

P(C | X) = [P(X | C) × P(C)] / P(X)

It estimates the probability that an observation belongs to a particular class given its feature values.

Gaussian Naive Bayes models the conditional distribution of numerical features using Gaussian distributions.

4. K-Nearest Neighbours

Euclidean distance between two observations is calculated as:

d(x,y) = √Σ(xᵢ - yᵢ)²

The algorithm identifies the nearest training observations and predicts the majority class among the selected neighbours.

5. Perceptron

The Perceptron calculates:

z = wᵀx + b

A threshold function determines the predicted class.

During training, weights are adjusted when the model makes an incorrect prediction.

6. Multilayer Perceptron

A neural-network neuron calculates:

z = Σ(wᵢxᵢ) + b

The ReLU activation function is:

ReLU(z) = max(0,z)

The network learns suitable weights through backpropagation and optimization.

7. Standardization

Standardization transforms numerical features into a comparable scale.

z = (x - μ) / σ

Where:

x = Original feature value

μ = Training-set mean

σ = Training-set standard deviation

The scaling parameters are learned from the training partition.

## 6. Experimental Setup

The dataset is divided into training and holdout test partitions using stratified sampling. Five-fold stratified cross-validation is used for model selection on the training data. The final benchmark metrics below are the values printed by the latest successful execution of `run_experiments.py`.

| Model | Reported configuration |
|---|---|
| Logistic Regression | Balanced class weights |
| Gaussian Naive Bayes | `var_smoothing = 1e-9` |
| Decision Tree | `max_depth = 6`, balanced |
| KNN | `k = 51`, subsample = 30,000 |
| Perceptron | Balanced class weights |
| MLP | Hidden layers = `(64, 32)` |

---

## 7. Model Evaluation and Results

The latest execution reports the following results on the holdout test set (`n = 46,631`).

### 7.1 Classification benchmark

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Prior Baseline | 0.660 | 0.215 | 0.213 | 0.214 | 0.498 |
| Logistic Regression (balanced) | 0.556 | 0.274 | **0.634** | **0.383** | 0.618 |
| Gaussian Naive Bayes | 0.772 | 0.295 | 0.035 | 0.062 | 0.592 |
| Decision Tree (depth=6, balanced) | 0.560 | 0.275 | **0.631** | **0.383** | 0.611 |
| KNN (k=51) | **0.783** | 0.407 | 0.003 | 0.007 | 0.598 |
| Perceptron (balanced) | 0.570 | 0.243 | 0.464 | 0.319 | 0.550 |
| MLP (64–32 units) | **0.783** | **0.577** | 0.004 | 0.008 | **0.622** |

### 7.2 Interpretation

The results show why accuracy alone is not sufficient for evaluating a classifier on an imbalanced dataset.

- **Logistic Regression** has the highest default-class recall (0.634) and shares the highest F1-score (0.383) with the Decision Tree.
- **Decision Tree** achieves a similar recall of 0.631 and an F1-score of 0.383.
- **MLP** has the highest ROC-AUC among the learned classifiers in the terminal output (0.622), but its default-class recall is only 0.004.
- **KNN** also has accuracy of 0.783, but its default-class recall is 0.003.
- **Gaussian Naive Bayes** has accuracy of 0.772 but recall of 0.035.

The balanced Logistic Regression and Decision Tree configurations identify a much larger share of actual default cases than the unweighted models in this experiment. Their lower accuracy reflects a different balance between the two classes; it should not be interpreted without considering recall, precision and F1-score.

### 7.3 ROC curve

ROC curves compare the True Positive Rate against the False Positive Rate at different classification thresholds. ROC-AUC summarizes a model's ability to rank positive cases above negative cases.

> **Figure update required:** The currently supplied ROC image displays AUC values that differ from the latest terminal output. Regenerate `fig3_roc_curves.png` using the same latest experiment results before embedding it here. Do not use the old plotted AUC labels as the current benchmark.

---

## 8. Algorithmic Fairness and Financial Inclusion Audit

The fairness audit compares the outcomes of the tuned Logistic Regression classifier across two borrower subgroups:

- **Thin-file borrowers:** First-time applicants without an established bureau history.
- **Established borrowers:** Applicants with an available bureau-scored credit history.

### 8.1 Latest subgroup results

| Metric | Thin-file borrowers | Established borrowers |
|---|---:|---:|
| Sample size | 26,010 | 20,621 |
| Actual default rate | 22.9% | 20.2% |
| Model-flagged default / denial rate | 62.1% | 35.2% |
| False Positive Rate | 59.2% | 31.1% |
| Approval rate | 37.9% | 64.8% |

The subgroup results show that the model flags a larger proportion of thin-file borrowers as potential defaulters. The false positive rate is also higher in the thin-file group.

### 8.2 False Positive Rate

The False Positive Rate (FPR) measures the proportion of actual non-defaulters that the model incorrectly classifies as defaulters:

\[
FPR=\frac{FP}{FP+TN}
\]

The reported FPR is 59.2% for thin-file borrowers and 31.1% for established borrowers.

### 8.3 Approval or selection rate

The approval rate is the proportion of a subgroup that is not flagged for rejection:

\[
Selection\ Rate=\frac{Approved\ Applicants}{Total\ Applicants}
\]

The reported selection rates are:

- Thin-file borrowers: 37.9%
- Established borrowers: 64.8%

### 8.4 Disparate Impact Ratio

The Disparate Impact Ratio compares the selection rate of the group being examined with that of the reference group:

\[
DIR=\frac{Selection\ Rate_{Thin-File}}{Selection\ Rate_{Established}}
\]

Using the reported approval rates:

\[
DIR=\frac{37.9}{64.8}\approx 0.585
\]

The resulting ratio is below the commonly used 0.80 Four-Fifths screening benchmark.

This indicates a substantial selection-rate disparity in the evaluated model and dataset. The ratio is a screening measure; it is not, by itself, proof of unlawful discrimination or a complete assessment of fairness.

### 8.5 Interpretation: the inclusion concern

The observed default rates are relatively close (22.9% and 20.2%), while the model's flagged rejection rates differ substantially (62.1% and 35.2%). The approval-rate comparison produces a DIR of approximately 0.585.

These results raise an important financial-inclusion concern: applicants without an established bureau history may experience less favourable model outcomes even when the observed subgroup default rates are similar.

The audit establishes an empirical disparity in this experiment. It does not establish that thin-file status alone caused the disparity.

> **Figure update required:** The currently supplied fairness chart contains values that do not match the latest terminal output. Regenerate it using the subgroup values in the table above before adding it to the repository as the current result.

---

## 9. SDG Alignment and Possible Policy Implications

### SDG 1 — No Poverty

Access to suitable financial services can help households acquire productive assets and participate in economic activity. The observed subgroup disparity motivates closer examination of how first-time borrowers are assessed.

### SDG 8 — Decent Work and Economic Growth

Vehicle financing can support self-employment and small-business activity. Credit assessment methods that account for more than established bureau history may be worth investigating as part of efforts to broaden access to financial services.

### Possible future directions

The current experiment is a benchmark and fairness audit, not a deployed lending system. Further work could investigate:

- Alternative cash-flow information, subject to consent, privacy and data-protection requirements.
- Tiered assessment approaches for applicants with limited bureau histories.
- Fairness-aware model training and threshold-selection methods.
- Longer repayment horizons rather than first-EMI default alone.
- Validation on additional datasets and lending populations.

These are proposed directions and are not claimed as implemented features of this project.

---

## 10. Limitations

- **Class imbalance:** The non-default class is substantially larger than the default class.
- **Prediction horizon:** The target represents first-EMI default, not long-term repayment behaviour.
- **Dataset scope:** Results are based on one publicly available vehicle-finance dataset.
- **Metric trade-offs:** High accuracy does not necessarily mean that a model detects default cases effectively.
- **Fairness interpretation:** DIR is a selection-rate screening measure and does not capture every dimension of fairness.
- **Causal interpretation:** Differences between subgroup metrics do not prove what caused those differences.
- **Deployment:** The models have not been established as suitable for real-world lending decisions.

This project is an academic machine learning benchmark and should not be used as a production credit-approval system.

---

## 11. Tools and Technologies

| Area | Tools |
|---|---|
| Programming language | Python |
| Data handling | Pandas, NumPy |
| Machine learning | Scikit-learn |
| Visualization | Matplotlib |
| Classification | Logistic Regression, Gaussian Naive Bayes, Decision Tree, KNN, Perceptron, MLP |
| Evaluation | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Documentation | Markdown, PDF |

---

## 12. How to Run the Project

### Requirements

- Python 3.10 or later
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- python-docx
- ReportLab

Install the dependencies:

```bash
python -m pip install numpy pandas matplotlib scikit-learn python-docx reportlab
```

### Dataset setup

1. Download the dataset from the [Kaggle dataset page](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction).
2. Place `train.csv` in the project root directory.

### Run the experiment

```bash
python run_experiments.py
```

### Generate plots separately

```bash
python generate_all_plots.py
```

The experiment script runs the training and evaluation pipeline. The plotting script generates the project visualizations. For the figures to match the results documented here, regenerate them from the same code version and data used for the latest terminal output.

---

## 13. Repository Structure

```text
Vehicle-Loan-Default-Prediction/
│
├── run_experiments.py
│   └── Model training, evaluation and fairness audit
│
├── generate_all_plots.py
│   └── Standalone visualization generation
│
├── train.csv
│   └── L&T vehicle-loan training dataset
│
├── fig1_missingness.png
│   └── Missing-value profile
│
├── fig2_workflow.png
│   └── End-to-end methodology workflow
│
├── fig3_roc_curves.png
│   └── ROC comparison of the classifiers
│
├── fig4_fairness.png
│   └── Subgroup fairness comparison
│
├── KhushveerKaurProblem42VehicleLoanPaper.pdf
│   └── Academic research paper
│
└── README.md
    └── Project documentation
```

---

## 14. References

1. Pedregosa, F. et al. (2011). “Scikit-learn: Machine learning in Python.” *Journal of Machine Learning Research*, 12, 2825–2830.
2. Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer.
3. Reserve Bank of India (2021). *Report of the Working Group on Digital Lending including Lending through Online Platforms and Mobile Apps*.
4. Hardt, M., Price, E., and Srebro, N. (2016). “Equality of opportunity in supervised learning.” *Advances in Neural Information Processing Systems*, 3315–3323.
5. Altman, E. I. (1968). “Financial ratios, discriminant analysis and the prediction of corporate bankruptcy.” *The Journal of Finance*, 23(4), 589–609.
6. L&T Financial Services. “Vehicle Loan Default Prediction Challenge Dataset.” Kaggle.

---

## Research Paper

The accompanying six-page research paper presents the study's motivation, methodology, model benchmark, subgroup audit and discussion.

**Title:** *Vehicle-Loan Default Prediction for First-Time Borrowers: A Credit-Risk Benchmark and Financial Inclusion Audit Using Classical Machine Learning.*

---

*This project explores the intersection of classical machine learning, credit-risk assessment and financial inclusion.*

