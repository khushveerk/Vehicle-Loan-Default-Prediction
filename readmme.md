# Vehicle Loan Default Prediction for First-Time Borrowers
### A Credit-Risk Benchmark and Financial Inclusion Audit Using Classical Machine Learning

An applied machine learning study exploring vehicle-loan default prediction in India, comparing classical supervised learning algorithms and investigating whether automated credit-risk assessment disproportionately penalizes first-time borrowers with limited credit histories.

**Course:** Machine Learning — B.Sc. (Hons.) Computer Science, Semester VI  
**Institution:** University of Delhi  
**Author:** Khushveer Kaur  
**Research Problem:** Problem No. 42  
**Dataset:** L&T Financial Services Vehicle Loan Default Prediction  
**Research Focus:** Credit Risk Prediction, Financial Inclusion and Algorithmic Fairness  
**SDG Alignment:** SDG 1 — No Poverty | SDG 8 — Decent Work and Economic Growth

---

## 📌 Project Overview

Access to vehicle financing plays an important role in the financial inclusion of low-income households in India. Two-wheelers, auto-rickshaws and light commercial vehicles often serve as productive assets that support employment and income generation.

However, individuals applying for their first formal loan frequently lack an established credit history. These applicants are commonly referred to as **thin-file borrowers**.

Traditional credit-scoring systems rely heavily on historical bureau information. Consequently, applicants without sufficient credit records may be classified as high-risk even when their actual repayment behaviour does not justify such classification.

This project investigates two interconnected questions:

1. Can classical machine learning algorithms predict early vehicle-loan defaults using applicant, financial and credit-related information?
2. Do these predictive models produce disproportionately high rejection rates for first-time borrowers with limited credit histories?

Using 233,154 loan records from the L&T Financial Services Vehicle Loan Default Prediction dataset, the project develops a leakage-conscious machine learning pipeline, benchmarks multiple classical classifiers and conducts a subgroup fairness audit.

The study combines predictive modelling with an examination of financial inclusion, demonstrating why model accuracy alone is insufficient when machine learning is applied to credit-risk assessment.

### 🎯 Research Objectives

- Develop a supervised machine learning framework for predicting vehicle-loan defaults.
- Compare classical classification algorithms taught in the undergraduate machine learning curriculum.
- Investigate the impact of class imbalance on minority-class default detection.
- Identify important factors associated with early loan default.
- Segment borrowers into thin-file and established-credit groups.
- Evaluate disparities in model rejection rates and false positive rates.
- Examine the implications of automated credit scoring for financial inclusion.

### ✨ Key Project Highlights

| Component | Description |
|---|---|
| Dataset | 233,154 loan records |
| Raw Features | 41 operational attributes |
| Classification | Binary loan-default prediction |
| Models | 6 classical ML classifiers + prior baseline |
| Validation | Stratified 80:20 train-test split |
| Model Selection | 5-fold stratified cross-validation |
| Evaluation | Accuracy, Precision, Recall, F1 and ROC-AUC |
| Fairness Analysis | Thin-file vs established borrowers |
| Fairness Metric | Disparate Impact Ratio (DIR) |
| Main Finding | DIR of 0.585, below the 0.80 benchmark |

---

## 📊 Dataset and Data Understanding

### Dataset Source

The project uses the publicly available L&T Financial Services Vehicle Loan Default Prediction dataset.

**Source:** [Kaggle – L&T Vehicle Loan Default Prediction](https://www.kaggle.com/datasets/mamtadhaker/lt-vehicle-loan-default-prediction)

### Dataset Characteristics

| Property | Description |
|---|---|
| Total Records | 233,154 |
| Raw Attributes | 41 |
| Target Variable | `loan_default` |
| Default Class | 1 |
| Non-default Class | 0 |
| Default Records | 50,594 |
| Non-default Records | 182,560 |
| Default Rate | Approximately 21.7% |
| Non-default Rate | Approximately 78.3% |

The dataset contains information about borrower demographics, vehicle financing, credit bureau history, existing credit accounts and loan-related characteristics.

### Understanding the Target Variable

The prediction target is `loan_default`.

- `1` – Borrower defaulted on the first scheduled EMI.
- `0` – Borrower did not default on the first scheduled EMI.

The task is therefore a binary classification problem.

### Important Features

Some of the important attributes considered in the analysis include:

| Feature | Meaning |
|---|---|
| `disbursed_amount` | Loan amount disbursed to the dealer |
| `asset_cost` | Cost of the financed vehicle |
| `ltv` | Loan-to-Value ratio |
| `Age` | Borrower's age derived from date of birth |
| `PERFORM_CNS.SCORE` | Credit bureau score |
| `PRI.ACTIVE.ACCTS` | Number of active primary credit accounts |
| `DELINQUENT.ACCTS` | Number of delinquent credit accounts |
| `NO.OF_INQUIRIES` | Number of recent credit inquiries |
| `Is_Thin_File` | Indicator identifying borrowers without an established bureau history |
| `loan_default` | Target variable |

---

## 🧹 Data Preprocessing and Feature Engineering

Before training the classifiers, the raw dataset undergoes several transformations to make it suitable for machine learning.

### 1. Missing Value Analysis

The missing-value profile is examined across all raw features.

The analysis identifies missing values in `Employment.Type`, representing approximately 3.28% of the records.

These missing values are assigned the category `Unemployed_or_Self`, followed by dummy encoding.

![Missing Value Profile](https://github.com/user-attachments/assets/94567bbe-967b-45da-9e9e-48a818585a8b)

*Figure 1. Percentage of missing values across the raw dataset.*

### 2. Date Parsing and Age Calculation

The `Date.of.Birth` attribute contains dates represented using a two-digit year format.

This can produce incorrect future birth years when dates are interpreted using an inappropriate century.

The century rollover is corrected before calculating borrower age.

### 3. Account Duration Conversion

The dataset contains account-duration attributes represented as text.

Examples include:

- `AVERAGE.ACCT.AGE`
- `CREDIT.HISTORY.LENGTH`

Values such as `2yrs 5mon` are converted into integer month equivalents using regular-expression-based parsing.

For example:

`2 years 5 months = (2 × 12) + 5 = 29 months`

This makes the duration attributes usable as numerical features.

### 4. Thin-File Borrower Identification

A new binary feature, `Is_Thin_File`, is created using `PERFORM_CNS.SCORE.DESCRIPTION`.

Applicants identified as having no available bureau history or being unscored are marked as thin-file borrowers.

This feature is particularly important for the subsequent financial inclusion and fairness analysis.

### 5. Feature Scaling and Data Leakage Prevention

The dataset is divided into training and testing partitions using stratified sampling.

- Training set: 80% — 186,523 records.
- Testing set: 20% — 46,631 records.

Stratification preserves the approximate default/non-default distribution in both partitions.

Preprocessing statistics are learned from the training data and subsequently applied to the test data.

This prevents information from the holdout test set from influencing the model-training process.

---

## 🧭 Machine Learning Workflow

The complete experiment follows a structured workflow, beginning with raw data and ending with model evaluation and subgroup fairness analysis.

![Machine Learning Workflow](https://github.com/user-attachments/assets/1a2b440d-c4b7-4f99-adcb-819a6f25e6db)

*Figure 2. End-to-end leakage-free machine learning methodology and fairness audit pipeline.*

### Workflow Stages

| Stage | Operation | Purpose |
|---|---|---|
| 1 | Data Loading | Load the L&T vehicle-loan dataset |
| 2 | Data Exploration | Understand feature distributions and missingness |
| 3 | Preprocessing | Handle missing values, dates and categorical attributes |
| 4 | Feature Engineering | Derive age, account duration and thin-file indicators |
| 5 | Train-Test Split | Create stratified training and testing partitions |
| 6 | Model Training | Train and tune classical ML classifiers |
| 7 | Model Evaluation | Compare predictive performance using classification metrics |
| 8 | Fairness Audit | Compare model decisions across borrower subgroups |

---

## 🤖 Machine Learning Models

The project evaluates classical supervised learning algorithms, each using a different approach to learning classification boundaries.

An empirical prior baseline is also included to provide a reference for model performance.

### 1. Stratified Prior Baseline

The baseline predicts using the empirical class distribution.

It provides a reference against which the performance of trained classifiers can be compared.

A useful model should demonstrate meaningful predictive discrimination beyond this baseline.

### 2. Logistic Regression

Logistic Regression estimates the probability of a borrower defaulting on a loan.

It uses the sigmoid function to transform a linear combination of input features into a probability between 0 and 1.

The model uses L2 regularization and balanced class weights to address class imbalance.

It is particularly useful because its coefficients provide an interpretable indication of how individual features influence predicted default risk.

### 3. Gaussian Naive Bayes

Gaussian Naive Bayes applies Bayes' theorem while assuming that numerical features follow Gaussian distributions conditional on the class.

It estimates the probability of each class using the observed feature values.

The model uses `var_smoothing = 1e-9` for numerical stability.

### 4. Decision Tree Classifier

A Decision Tree divides the feature space into smaller regions through a sequence of feature-based decisions.

The algorithm uses Gini impurity to identify useful splitting points.

The model is constrained using:

- `max_depth = 6`
- `min_samples_leaf = 50`
- Balanced class weights

These restrictions help control tree complexity and reduce overfitting.

### 5. K-Nearest Neighbours (KNN)

KNN classifies a borrower by examining the labels of nearby training observations.

The implementation uses:

- Number of neighbours: `k = 51`
- Distance metric: Euclidean
- Uniform voting
- Stratified training subsample of 30,000 observations

Feature scaling is important because KNN relies on distance calculations.

### 6. Single-Layer Perceptron

The Perceptron is a linear classification algorithm inspired by a simplified biological neuron.

It calculates a weighted sum of input features and applies a threshold to produce a class prediction.

Its weights are updated using mistake-driven learning.

The implementation uses L2 regularization and balanced class weighting.

### 7. Multilayer Perceptron (MLP)

The Multilayer Perceptron is a feedforward artificial neural network consisting of an input layer, two hidden layers and an output layer.

The hidden-layer architecture is:

`(64, 32)`

ReLU activation is used in the hidden layers, and the network learns through backpropagation.

The implementation uses the Adam optimizer and early stopping.

---

## 🧮 Understanding the Mathematical Concepts

This section explains the main mathematical ideas behind the project.

### 1. Logistic Regression and the Sigmoid Function

Logistic Regression calculates a linear combination of input features:

`z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ`

The sigmoid function converts this value into a probability:

`P(Y = 1 | X) = 1 / (1 + e⁻ᶻ)`

The resulting probability represents the model's estimated likelihood of default.

### 2. Gini Impurity

Gini impurity measures the degree of class mixing within a decision-tree node.

`Gini = 1 - Σ(pᵢ²)`

Where `pᵢ` represents the proportion of observations belonging to class i.

- Gini = 0: The node contains only one class.
- Higher Gini: The node contains a greater mixture of classes.

Decision Trees select splits that reduce impurity.

### 3. Bayesian Classification

Naive Bayes uses Bayes' theorem:

`P(C | X) = [P(X | C) × P(C)] / P(X)`

It estimates the probability that an observation belongs to a particular class given its feature values.

Gaussian Naive Bayes models the conditional distribution of numerical features using Gaussian distributions.

### 4. K-Nearest Neighbours

Euclidean distance between two observations is calculated as:

`d(x,y) = √Σ(xᵢ - yᵢ)²`

The algorithm identifies the nearest training observations and predicts the majority class among the selected neighbours.

### 5. Perceptron

The Perceptron calculates:

`z = wᵀx + b`

A threshold function determines the predicted class.

During training, weights are adjusted when the model makes an incorrect prediction.

### 6. Multilayer Perceptron

A neural-network neuron calculates:

`z = Σ(wᵢxᵢ) + b`

The ReLU activation function is:

`ReLU(z) = max(0,z)`

The network learns suitable weights through backpropagation and optimization.

### 7. Standardization

Standardization transforms numerical features into a comparable scale.

`z = (x - μ) / σ`

Where:

- `x` = Original feature value
- `μ` = Training-set mean
- `σ` = Training-set standard deviation

The scaling parameters are learned from the training partition.

---

## ⚙️ Experimental Setup and Model Selection

The experiment uses an 80:20 stratified train-test split.

The training partition is used for model fitting and hyperparameter selection, while the independent holdout partition is reserved for final evaluation.

Five-fold stratified cross-validation is used for hyperparameter selection.

| Model | Selected Configuration |
|---|---|
| Logistic Regression | C = 0.1, L2, balanced |
| Gaussian Naive Bayes | var_smoothing = 1e-9 |
| Decision Tree | Depth = 6, minimum leaf = 50, balanced |
| KNN | k = 51, Euclidean distance |
| Perceptron | alpha = 1e-3, L2, balanced |
| MLP | Hidden layers = (64, 32), alpha = 1e-4, early stopping |

The final model comparison is performed on the holdout test set rather than the cross-validation folds.

---

## 📈 Experimental Results

The following results are obtained from the independent test partition containing 46,631 records.

### Classifier Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Prior Baseline | 0.660 | 0.215 | 0.213 | 0.214 | 0.498 |
| Logistic Regression | 0.556 | 0.274 | 0.634 | 0.383 | 0.618 |
| Gaussian Naive Bayes | 0.772 | 0.295 | 0.035 | 0.062 | 0.592 |
| Decision Tree | 0.560 | 0.275 | 0.631 | 0.383 | 0.611 |
| KNN | 0.783 | 0.407 | 0.003 | 0.007 | 0.598 |
| Perceptron | 0.570 | 0.243 | 0.464 | 0.319 | 0.550 |
| MLP | 0.783 | 0.577 | 0.004 | 0.008 | 0.622 |

### Interpreting the Results

The results reveal an important distinction between overall accuracy and minority-class detection.

**Logistic Regression** achieves the highest default-class recall of 0.634 and an F1-score of 0.383.

**Decision Tree** performs similarly, achieving a recall of 0.631 and an F1-score of 0.383.

In comparison, KNN and MLP achieve higher overall accuracy of 0.783 but identify almost none of the actual default cases.

This occurs because the dataset is highly imbalanced. A classifier can obtain high accuracy by favouring the majority non-default class while failing to identify borrowers who actually default.

Therefore, recall and F1-score are particularly important when evaluating default detection.

### ROC Curve Analysis

The Receiver Operating Characteristic (ROC) curve evaluates the relationship between the True Positive Rate and False Positive Rate at different classification thresholds.

ROC-AUC summarizes a classifier's ability to distinguish between the two classes.

![ROC Curves](https://github.com/user-attachments/assets/fc0317b0-f963-449d-b91c-66194c0d2f0a)

*Figure 3. ROC curves comparing the seven experimental classifiers against the prior baseline.*

The results show that Logistic Regression achieves the strongest default-class sensitivity among the evaluated models, while the MLP obtains the highest ROC-AUC among the learned classifiers in the reported benchmark.

---

## ⚖️ Algorithmic Fairness and Financial Inclusion Audit

Predictive performance alone does not establish whether a credit-scoring model treats different borrower groups fairly.

A model may achieve reasonable overall performance while producing substantially different outcomes for applicants with different credit-history profiles.

To investigate this issue, the test dataset is divided into two groups:

1. **Thin-File Borrowers:** First-time applicants without an established credit bureau history.
2. **Established Borrowers:** Applicants with an available bureau-scored credit history.

The fairness analysis is performed using the tuned Logistic Regression classifier.

### Subgroup Comparison

| Metric | Thin-File Borrowers | Established Borrowers |
|---|---:|---:|
| Sample Size | 26,010 | 20,621 |
| Observed Default Rate | 22.9% | 20.2% |
| Model Flagged Rejection Rate | 62.1% | 35.2% |
| False Positive Rate | 59.2% | 31.1% |
| Approval Rate | 37.9% | 64.8% |
| Disparate Impact Ratio | 0.585 | 1.000 (Reference) |

### 1. False Positive Rate (FPR)

The False Positive Rate measures the proportion of actual non-defaulters incorrectly classified as defaulters.

`FPR = FP / (FP + TN)`

A higher FPR indicates that more borrowers who did not default are incorrectly flagged as high-risk.

The thin-file subgroup has an FPR of 59.2%, compared with 31.1% for established borrowers.

This represents a substantial difference in classification errors between the two groups.

### 2. Selection Rate

The selection rate represents the proportion of applicants classified as eligible for approval.

`Selection Rate = Approved Applicants / Total Applicants`

The model's approval rates are:

- Thin-file borrowers: 37.9%
- Established borrowers: 64.8%

### 3. Disparate Impact Ratio (DIR)

The Disparate Impact Ratio compares the selection rate of a group against the selection rate of a reference group.

`DIR = Selection Rate of Thin-File Group / Selection Rate of Established Group`

Substituting the observed values:

`DIR = 37.9 / 64.8`

`DIR ≈ 0.585`

### 4. Four-Fifths Rule

The Four-Fifths Rule uses 0.80 as a commonly applied screening benchmark for identifying potential adverse impact in selection rates.

- DIR ≥ 0.80: Meets the numerical screening benchmark.
- DIR < 0.80: Potential adverse impact requiring further investigation.

The observed DIR of 0.585 falls below this benchmark.

This indicates a substantial selection-rate disparity against thin-file applicants.

### 📊 Fairness Audit Visualization

![Fairness Audit](https://github.com/user-attachments/assets/cd0cec5c-501a-41de-bebb-f019626354db)

*Figure 4. Comparison of approval rates and classification error disparities between thin-file and established borrowers.*

### The Inclusion Paradox

The central finding of this investigation is that borrowers without established bureau histories experience substantially higher rejection rates despite having an observed default rate that is relatively close to that of established borrowers.

The model rejects 62.1% of thin-file applicants, compared with 35.2% of established applicants.

The results highlight the limitations of relying heavily on conventional credit-history indicators when assessing first-time borrowers.

**Important interpretation:** The fairness audit establishes an empirical disparity in this dataset and model. It does not, by itself, prove that the absence of bureau history caused the disparity or establish a legal violation.

---

## 🌱 Research Alignment and Policy Implications

This project connects machine learning with the broader objective of financial inclusion.

### SDG 1 — No Poverty

Access to formal credit can help low-income households acquire productive assets and improve their economic opportunities.

A credit-risk model that disproportionately excludes first-time borrowers may create additional barriers to accessing financial services.

### SDG 8 — Decent Work and Economic Growth

Vehicle financing can support self-employment, transportation services and small-business activities.

More inclusive credit assessment can contribute to wider access to financial services and productive economic opportunities.

### Possible Improvements to Credit Assessment

The findings motivate further investigation into:

1. **Tiered Loan Sanctions:** Using moderated loan-to-value limits for applicants with limited credit histories instead of relying exclusively on binary rejection decisions.
2. **Alternative Cash-Flow Underwriting:** Exploring additional repayment indicators such as regular utility payments and transaction patterns, subject to appropriate consent and privacy safeguards.
3. **Fairness-Constrained Learning:** Incorporating fairness objectives into model optimization.
4. **Extended Repayment Analysis:** Studying repayment behaviour over longer periods rather than restricting the target to first-EMI default.

These are proposed directions for improving the system, not implemented components of the current experiment.

---

## 📦 Generated Visualizations

The project includes four principal visual outputs:

| Figure | Description |
|---|---|
| Figure 1 | Missing-value profile of the raw dataset |
| Figure 2 | End-to-end machine learning workflow |
| Figure 3 | ROC curves comparing the classifiers |
| Figure 4 | Subgroup fairness and approval-rate disparity |

The visualizations provide complementary views of data quality, methodology, predictive performance and financial inclusion outcomes.

---

## 🚀 Quick Start

### Requirements

- Python 3.10 or later
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- python-docx
- ReportLab

### Install Dependencies

```bash
python -m pip install numpy pandas matplotlib scikit-learn python-docx reportlab
```

### Dataset Setup

1. Download the L&T Financial Services Vehicle Loan Default Prediction dataset from Kaggle.
2. Obtain the `train.csv` file.
3. Place the dataset in the project's root directory.

### Run the Experiment

Execute the complete machine learning pipeline:

```bash
python run_experiments.py
```

### Generate Visualizations Separately

To generate the high-resolution figures:

```bash
python generate_all_plots.py
```

The experiment script performs model training, evaluation and metric generation. The standalone plotting script generates the visual outputs.

---

## 🗂️ Repository Structure

```text
Vehicle-Loan-Default-Prediction/
│
├── run_experiments.py
│   └── Complete machine learning training and evaluation pipeline
│
├── generate_all_plots.py
│   └── Standalone visualization generation script
│
├── train.csv
│   └── L&T Financial Services training dataset
│
├── fig1_missingness.png
│   └── Missing-value analysis
│
├── fig2_workflow.png
│   └── Machine learning workflow diagram
│
├── fig3_roc_curves.png
│   └── Comparative ROC curves
│
├── fig4_fairness.png
│   └── Algorithmic fairness audit visualization
│
├── KhushveerKaurProblem42VehicleLoanPaper.pdf
│   └── Six-page academic research paper
│
└── README.md
    └── Project documentation and reproduction guide
```

---

## ⚠️ Limitations and Considerations

Although the project provides useful insights into vehicle-loan default prediction and subgroup disparities, several limitations should be considered.

- **Class Imbalance:** The dataset contains considerably more non-default observations than default observations, affecting classifier behaviour.
- **Limited Prediction Horizon:** The target focuses on the first scheduled EMI default and does not represent long-term repayment behaviour.
- **Dataset Scope:** The experiment uses one publicly available vehicle-finance dataset and may not generalize to every lending institution or borrower population.
- **Model Performance:** Several classifiers achieve high accuracy but extremely low default recall, demonstrating why accuracy should not be considered independently.
- **Fairness Measurement:** The Four-Fifths Rule is a screening benchmark, not a complete measure of algorithmic fairness or a standalone legal determination.
- **Subgroup Definition:** Thin-file status is constructed using available bureau-description information and may not capture every form of limited credit history.
- **Causality:** Observed differences in approval and error rates do not establish the underlying causal mechanism.
- **Future Validation:** Further testing across institutions, geographic regions and longer repayment periods would be required before considering real-world lending applications.

The project is an academic research and benchmarking exercise, not a production credit-scoring or loan-approval system.

---

## 🛠️ Tools and Technologies

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Data Manipulation | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Data Visualization | Matplotlib |
| Model Evaluation | Precision, Recall, F1-score, ROC-AUC |
| Classification | Logistic Regression, Naive Bayes, Decision Trees, KNN, Perceptron, MLP |
| Documentation | Markdown, PDF |

---

## 📚 References

1. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
2. Hastie, T., Tibshirani, R., and Friedman, J. (2009). *The Elements of Statistical Learning: Data Mining, Inference, and Prediction*. 2nd ed. Springer.
3. Reserve Bank of India (2021). *Report of the Working Group on Digital Lending including Lending through Online Platforms and Mobile Apps*.
4. Hardt, M., Price, E., and Srebro, N. (2016). Equality of opportunity in supervised learning. *Advances in Neural Information Processing Systems*, 3315–3323.
5. Altman, E. I. (1968). Financial ratios, discriminant analysis and the prediction of corporate bankruptcy. *The Journal of Finance*, 23(4), 589–609.
6. L&T Financial Services. (2019). *Vehicle Loan Default Prediction Challenge Dataset*. Kaggle.

---

## 📄 Research Paper

The complete academic study, including the research motivation, experimental methodology, empirical results, fairness analysis and policy implications, is available in the repository as a six-page research paper.

**Research Title:** Vehicle-Loan Default Prediction for First-Time Borrowers: A Credit-Risk Benchmark and Financial Inclusion Audit Using Classical Machine Learning.

---

*An academic exploration of how machine learning can contribute to credit-risk assessment while highlighting the importance of fairness and financial inclusion.*
