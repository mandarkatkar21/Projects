# Credit Risk Analysis Project

## Project Overview
This project focuses on credit risk analysis using machine learning techniques to predict loan approval based on various financial and demographic factors. The dataset consists of customer-related attributes that help in assessing creditworthiness.

## Objective
The goal of this project is to develop and evaluate different machine learning models to classify loan applications into approved and non-approved categories based on customer attributes.

## Dataset
The dataset contains various features related to customer demographics, credit history, and financial details. The target variable, `Approved_Flag`,classifies customers into predefined categories based on past trends (P1, P2, P3, P4).

## Data Preprocessing
1. **Handling Missing Values:** Missing values were treated using imputation techniques.
2. **Feature Encoding:**
   - Categorical variables (`MARITALSTATUS`, `GENDER`) were one-hot encoded.
   - Target variable `Approved_Flag` was label-encoded.
3. **Feature Selection:** Statistical techniques like Variance Inflation Factor (VIF) and ANOVA were used to retain important features.
4. **Scaling:** Numeric variables were standardized using Min-Max scaling.

## Model Training
Several machine learning models were trained and evaluated:
1. **Decision Tree Classifier**
2. **Random Forest Classifier**
3. **XGBoost Classifier**

### Model Evaluation Metrics:
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

## Hyperparameter Tuning
GridSearchCV was used for hyperparameter tuning of Random Forest and XGBoost to improve model performance.

## Results
- Random Forest and XGBoost outperformed the Decision Tree model.
- Feature importance analysis helped in identifying the key factors affecting loan approval.

## Installation & Usage
### Prerequisites:
- Python 3.7+
- Required Libraries: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `seaborn`

### Setup:
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/credit-risk-analysis.git
   cd credit-risk-analysis
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the main script:
   ```bash
   python main.py
   ```

## Conclusion
This project demonstrates how machine learning can be leveraged for credit risk assessment. Further improvements can be made by exploring deep learning models and incorporating more financial indicators.

## Authors
- Your Name
- Contact: your.email@example.com

## License
This project is licensed under the MIT License.

