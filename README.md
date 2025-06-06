### 📁 Project Structure

```text
Loan-Approval-Machine-Learning-Project-Learning-Classification-using-XGBoost-and-SHAP/
│
├── README.md
├── requirements.txt
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── raw.csv
│
├── notebooks/
│   ├── Loan Approval Machine Learning Project Learning Classification-using-XGBoost and SHAP.ipynb
│
├── app/
│   ├── app.py
│   └── model.pkl
```
### Project Background
Access to financial services is a key driver of economic growth and personal development. However, traditional loan approval processes often involve lengthy paperwork, manual verification, and subjective decision-making, which can delay or even prevent credit access for many individuals. As demand for faster and fairer lending grows, financial institutions are looking for smarter, data-driven solutions.
This project introduces a Loan Approval System powered by machine learning to streamline and automate the loan evaluation process. By analyzing applicant data—such as income, credit history, employment status, and existing debts—the system predicts the likelihood of loan repayment and assists in making informed approval decisions. The goal is to reduce processing time, minimize human bias, and improve overall accuracy in credit risk assessment.
By leveraging predictive analytics, this system supports financial institutions in extending credit responsibly while enhancing customer experience through faster decisions and greater transparency.

Insights and recommendations are provided on the following key areas:

- **Feature Preprocessing**
- **Exploratory Data Analysis (EDA)** 
- **Feature Selection** 
- **Model Training** 

The full Data Science Notebook can be found here. [[link](https://github.com/nixon-quesada/Loan-Approval-Machine-Learning/blob/main/Loan%20Approval%20Machine%20Learning/notebook/Loan%20Approval%20System.ipynb)].

### Data Structure & Initial Checks

The dataset structure as seen below consists of 12 features, 1 id feature and 1 target column with a total row count of 4629 records. A description of each feature is as follows:

![features 1](https://github.com/user-attachments/assets/464cd1a5-8d27-491f-a3cc-8f26877f0628)
![features 2](https://github.com/user-attachments/assets/af3076a6-0d7e-40ab-a711-24463f0f51b4)

### Executive Summary

*[Visualization, including a graph of overall trends or snapshot of a dashboard]*

**Project Title:** *Loan Approval Machine Learning Project: Learning Classification using XGBoost and SHAP*

**Objective**: The goal of this project is to build a highly accurate and interpretable supervised machine learning model to predict loan approval status using historical loan application data.

**Approach**: I used Extreme Gradient Boosting (XGBoost) — a powerful tree-based ensemble method — to develop a binary classification model that predicts whether a loan application will be approved.
**Methodology**: 

1. Data Preprocessing & Exploration
- Cleaned and encoded categorical features.
- Explored feature distributions and correlations.
- Handled class balance (approx. 62% approved, 38% not approved).
- Performed EDA using seaborn and matplotlib visualizations.

2. Model Training
- Trained XGBClassifier on the training set only.
- Used Stratified K-Fold Cross-Validation (25 folds) for performance estimation.
- Tracked multiple metrics: Accuracy, F1-score, and ROC AUC.
- Achieved cross-validation mean accuracy ~99.5%, with minimal variance.

3. Evaluation on Unseen Test Set
- Separated a held-out test set before modeling.
- Final test metrics:
  - Accuracy: 99.77%
  - F1-Score: 99.82%
  - ROC AUC: 99.999%
- Indicates strong generalization and model confidence.

4. Robustness Checks
- Added label noise experiments to test overfitting resistance.
- Verified results are not memorizing artifacts.

6. Interpretability & Calibration
 - Applied SHAP (SHapley Additive exPlanations) for local/global feature importance.
   - Confirmed model decisions align with logical expectations (e.g., higher credit score → higher approval).
 - Generated calibration plots to ensure predicted probabilities are meaningful.

### Insights Deep Dive

**1. Feature Preprocessing:**

**Handling Missing Values:**
  
- The dataset is of a manageable size, consisting of 12 columns and 4,269 rows.
- It contains no duplicate entries or missing values, which simplifies initial cleaning.
- The primary issue identified was the presence of leading and trailing spaces in column names, as well as leading spaces in some categorical values.
  
**Encoding Categorical Variables:**
  
- Applied string manipulation methods (e.g., .strip()) to remove unwanted whitespace from column names and categorical entries.
- Encoded binary categorical variables into numerical format (0 and 1) to facilitate model training.
- Split the dataset into training and testing subsets to evaluate model performance and prevent data leakage.

### 2. Exploratory Data Analysis (EDA)

1. Target Imbalance Check:
- Moderate imbalance observed in loan_status (around 61% approved, 39% not).
- Confirmed via .value_counts(normalize=True) and visual bar plots.
  
2. Univariate & Bivariate Analysis:
- Categorical features like education, self_employed, and loan_term analyzed via approval rate bar plots.
- Numerical features like credit_score, bank_asset_value, and residential_assets_value explored using KDE + stacked histograms.
- Stratified comparisons revealed consistent relationships:
- Higher credit score and presence of collateral increased loan approval likelihood.
- Asset-rich individuals had a higher probability of loan approval.
  
3. Correlation Analysis:
- Generated heatmap using custom palette.
- Extreme multicollinearity found among features related to a person's asset, credit_score is the only feature associated with loan approvals.

![image](https://github.com/user-attachments/assets/9fb5ca64-ff0d-4243-b67a-a4f6ea0a78da)
![graph 1](https://github.com/user-attachments/assets/45a2439e-fe87-4194-a186-b7d6a7cca904)
![image](https://github.com/user-attachments/assets/8253faa6-39d7-4607-89ca-29e56f897ea8)

### Feature Selection & Model Interpretation:

1. **Feature Importance:**
- XGBoost's built-in importance plot highlighted credit_score, collateral, and installment_income_ratio as top contributors.
- Verified feature behavior using SHAP values for local/global interpretability.
2. **SHAP Insights:**
- Used force_plot and summary plot to visualize how specific features pushed predictions up/down.
- Found:
  - Collateral presence sharply boosted approval odds.
  - High installment-to-income ratio pulled predictions toward rejection.
  - Credit score and bank asset value had strong positive influence.
2. **Validation of Model Logic:**
- Tested on noise-injected labels to ensure model wasn’t memorizing artifacts.
- Ran calibration curves to verify confidence levels in probability predictions.

![image](https://github.com/user-attachments/assets/e5e77f47-d6ab-4b2e-b179-21074acf78ba)
![image](https://github.com/user-attachments/assets/40f635bd-b696-4be1-851e-e84f1500fe06)


### 4. Model Training & Evaluation:

1. Model Used:
- Final model: XGBClassifier(random_state=69)
- Hyperparameters kept near-default due to already high out-of-box performance.
2. Cross-Validation Strategy:
- Used StratifiedKFold(n_splits=10) to maintain label balance across folds.
- Captured metrics for each fold: Accuracy, F1-Score, and ROC-AUC.
- Consistently high results across all metrics (most folds above 0.99).
3. Unseen Test Performance:
- Held-out test set (never used during training or EDA) showed:
  - Accuracy: 99.77%
  - F1-Score: 99.82%
  - ROC-AUC: 99.99%
- Confirmed model generalizes extremely well on unseen data.

![image](https://github.com/user-attachments/assets/766de675-223f-429c-ae2b-125987be3525)
![image](https://github.com/user-attachments/assets/a3306f16-8646-4219-8dd8-e29fa3ce0b15)


# Recommendations:

Based on the insights and findings above, I would recommend to consider the following: 

**1. Operational Deployment**
- Deploy the model using Streamlit or a web API to integrate into internal systems for real-time decision support.
- Set input validation rules (e.g., valid credit score ranges, non-negative asset values) to ensure clean data intake.
- Log model predictions for auditability and future retraining needs.

**2. Business Usage**
- Prioritize credit score, collateral, and installment-to-income ratio when assessing applicants—these are the strongest predictors of loan approval.
- Introduce a “gray zone” threshold for borderline applicants (e.g., 0.4 < probability < 0.6) to be manually reviewed instead of auto-approved/rejected.
- Use SHAP insights to explain rejections to clients in a fair and interpretable way (e.g., “Low bank asset value contributed to rejection”).

**3. Model Monitoring & Maintenance**
- Establish a monitoring dashboard to track live prediction distributions and flag drifts in:
  - Approval rate trends
  - Feature distributions (e.g., credit scores dropping over time)
- Schedule quarterly retraining using new approved/rejected applications to keep the model current.
- Use calibration curves in production to adjust probability thresholds if the business wants to be more conservative or aggressive.

**4. Further Improvements**
- Explore ensemble models (e.g., blending XGBoost with LightGBM or Logistic Regression) for marginal performance gain.
- Consider adding external features like credit bureau scores, geographic risk factors, or employment industry data if available.
- Evaluate fairness metrics to ensure the model is not biased against specific demographic groups (e.g., age, gender, employment status).

**5. Communication & Transparency**
- Provide decision-makers with confidence scores and explanations, not just binary predictions.
- Train staff on interpreting SHAP plots so that model predictions can be communicated clearly to clients.
- Maintain a clear documentation trail covering data lineage, preprocessing logic, and model evaluation steps.
  
### Assumptions and Caveats:

**Assumptions**

**1. Training Data Representativeness**

The model assumes that the historical loan data provided is a reliable and accurate reflection of future applicant behavior and approval policies.
**2. Feature Stability Over Time**

It is assumed that the relationships between features (e.g., credit score, collateral value) and loan approval will remain stable over time.

**3. No Data Leakage**

The model development ensured no data from the test or future sets influenced training (e.g., via proper train-test splits and cross-validation).

**4. Binary Classification Scope**

The current model only supports binary loan decisions (approve vs. reject), assuming no intermediate decisions like "pending review" or "request more info."

**5. Label Accuracy**

It is assumed that the loan_status labels are correct and do not contain noise (e.g., mislabeling of rejections or approvals).

**Caveats**

**1. Class Imbalance**

Although stratified K-fold cross-validation was used, the slight imbalance in approved vs. rejected cases may still bias the model’s learning and generalization.

**2. Unseen External Factors**

The model does not account for external economic indicators (e.g., interest rate changes, inflation, regulatory shifts) that may affect loan approval decisions.

**3. SHAP Interpretability Limitations**

While SHAP offers valuable insights into feature importance for individual predictions, it is not foolproof and should be used alongside human judgment.

**4. Overfitting Risk**

Extremely high performance metrics suggest a potential risk of overfitting—even with validation in place. Performance should be reevaluated with completely new or live data.

**5. Feature Encoding Decisions**

Some categorical features were label encoded (e.g., education, employment) which may impose ordinal assumptions. This could affect interpretability or fairness.

**6. No Fairness or Bias Audit**

The model has not yet been evaluated for demographic fairness (e.g., gender, age, socioeconomic status). This should be addressed before production use.
