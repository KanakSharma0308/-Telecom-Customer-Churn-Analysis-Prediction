Telecom Customer Churn: Analysis & Prediction
An end-to-end analysis of why telecom customers churn, and a machine learning model that flags at-risk customers before they leave. Framed as an analyst question: which customers are likely to leave, why, and what should the business do about it?

Business problem
Acquiring a new telecom customer costs far more than retaining an existing one. This project identifies the behavioral and contract-related factors that predict churn, so the business can target retention offers at the customers most likely to leave — instead of running blanket retention campaigns.

Dataset
IBM Telco Customer Churn (Kaggle): ~7,043 customers, target variable Churn (Yes/No).

Workflow
Data Cleaning → EDA → Feature Engineering → Handling Class Imbalance → Model Training & Comparison → Evaluation → Prediction Pipeline

1. Data cleaning
Converted TotalCharges from object to numeric, dropped rows with missing values
Removed customerID (non-informative)
2. Exploratory Data Analysis
Churn distribution, churn by contract type, monthly charges vs. churn, tenure vs. churn, correlation heatmap.

Key insights:
The large majority of churned customers were on month-to-month contracts, versus far lower churn on one- or two-year contracts
Senior citizens churn at a noticeably higher rate than non-senior customers
[Add 1-2 more from your EDA charts — e.g. churn vs. internet service type, or churn vs. tenure]
3. Feature engineering
Label encoding applied to all categorical columns; encoders saved to encoders.pkl for reuse at prediction time.

4. Handling class imbalance
Applied SMOTE (Synthetic Minority Oversampling Technique) on the training data, since churned customers are a minority class and a model trained on imbalanced data would under-predict churn.

5. Model training & comparison
Three models were evaluated using 5-fold cross-validation:

Model	CV Accuracy
Decision Tree	78%
Random Forest	84% ✅ Selected
XGBoost	84%

Random Forest and XGBoost tied on cross-validation accuracy; Random Forest was selected as the final model for its simpler tuning and interpretable feature importances.

6. Model evaluation (on the test set)
Metric	Score
Accuracy	76.5%
ROC-AUC	0.81
Precision (Churn class)	56%
Recall (Churn class)	57%

Recall on the churn class matters most here: missing an actual churner (false negative) costs more than flagging a loyal customer by mistake, since the cost of a retention offer is much lower than the cost of losing the customer. At 57% recall, the model catches just over half of actual churners — a reasonable starting point, but there's room to improve: SMOTE plus hyperparameter tuning (e.g. class weighting, threshold adjustment) could push recall higher, since the current 0.5 probability threshold trades off some recall for overall accuracy.

Top predictive features: tenure, MonthlyCharges, TotalCharges, and Contract type were the strongest predictors of churn.

Prediction pipeline

The saved model (customer_churn_model.pkl) takes a new customer's details (gender, tenure, contract type, monthly charges, internet service, etc.) and returns a churn prediction with probability, so it can be plugged into a retention-targeting workflow.

Recommendations
Target month-to-month customers first. They churn at the highest rate — offer a discount for switching to a 1-year contract.
Watch new customers closely. Low tenure is one of the top predictors of churn, so the first few months matter most for retention outreach.
Prioritize by predicted probability, not just the Yes/No flag. Use the model's probability score to rank customers so retention budget goes to the highest-risk, highest-value customers first.
Tech stack
Category	Libraries
Data manipulation	pandas, numpy
Visualization	matplotlib, seaborn
Machine learning	scikit-learn (Decision Tree, Random Forest)
Boosting	xgboost
Imbalance handling	imbalanced-learn (SMOTE)
Model persistence	pickle
Getting started
bash
git clone https://github.com/KanakSharma0308/telecom-customer-churn-analysis
cd telecom-customer-churn-analysis
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn

Place WA_Fn-UseC_-Telco-Customer-Churn.csv (from the Kaggle link above) in the project root, then run:

bash
jupyter notebook churn_analysis.ipynb
Project structure
telecom-customer-churn-analysis/
├── churn_analysis.ipynb        # full analysis and model training
├── customer_churn_model.pkl    # saved trained model
├── encoders.pkl                # saved label encoders
└── README.md
