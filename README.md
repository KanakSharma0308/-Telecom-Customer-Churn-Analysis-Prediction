📉 Telecom Customer Churn Analysis & Prediction
A complete end-to-end machine learning project to analyze and predict customer churn for a telecom company using the IBM Telco Customer Churn dataset.

📌 Problem Statement
Customer churn — when customers stop doing business with a company — is a major challenge in the telecom industry. This project identifies the key factors driving churn and builds a predictive model to flag at-risk customers before they leave.

📂 Dataset
Source: IBM Sample Dataset – Telco Customer Churn
File: WA_Fn-UseC_-Telco-Customer-Churn.csv
Records: ~7,043 customers
Target Variable: Churn (Yes / No)


🔍 Project Workflow
Data Loading → Cleaning → EDA → Feature Engineering → Modeling → Evaluation → Prediction
1. Data Cleaning
Converted TotalCharges from object to numeric
Dropped rows with missing values in TotalCharges
Removed customerID (non-informative column)

2. Exploratory Data Analysis (EDA)
Churn distribution plot
Churn by contract type
Monthly charges vs churn (KDE histogram)
Tenure by churn (boxplot)
Correlation heatmap

3. Key Insights
A large majority of churned customers were on month-to-month contracts
Senior citizens have a noticeably higher churn rate than non-senior customers

4. Feature Engineering

Label encoding applied to all categorical columns
Encoders saved to encoders.pkl for reuse in deployment

5. Handling Class Imbalance

Applied SMOTE (Synthetic Minority Oversampling Technique) on training data to balance churn vs non-churn classes

6. Model Training & Comparison
Three models were evaluated using 5-fold cross-validation:
ModelCV AccuracyDecision Tree—Random Forest✅ BestXGBoost—
Random Forest was selected as the final model.

7. Model Evaluation
Accuracy Score
Confusion Matrix
Classification Report
ROC-AUC Score
ROC Curve plot
Top 10 Feature Importances (bar chart)


🤖 Prediction Pipeline
The saved model (customer_churn_model.pkl) accepts a new customer's details and predicts:

Churn / No Churn
Prediction probability

Example input fields: gender, tenure, Contract, MonthlyCharges, InternetService, etc.

🛠️ Tech Stack
Category           | Libraries 
Data Manipulation  | pandas, numpy 
Visualization      | matplotlib, seaborn 
Machine Learning   | scikit-learn (Decision Tree, Random Forest)
Boosting           | xgboost 
Imbalance Handling | imbalanced-learn (SMOTE) 
Model Persistence  | pickle 


🚀 Getting Started
1. Clone the repository
bashgit clone https://github.com/KanakSharma0308/-Telecom-Customer-Churn-Analysis-Prediction
cd churn-analysis
2. Install dependencies
bashpip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn
3. Add the dataset
Place WA_Fn-UseC_-Telco-Customer-Churn.csv in the project root directory.
4. Run the notebook
bashjupyter notebook churn_analysis.ipynb

📁 Project Structure
churn-analysis/
│
├── churn_analysis.ipynb                      https://github.com/KanakSharma0308/-Telecom-Customer-Churn-Analysis-Prediction   
├── WA_Fn-UseC_-Telco-Customer-Churn.csv      https://www.kaggle.com/datasets/blastchar/telco-customer-churn
├── customer_churn_model.pkl                  
├── encoders.pkl                 
└── README.md

📊 Results

The Random Forest classifier achieved strong performance on the test set
High ROC-AUC score indicating good discrimination between churn and non-churn
tenure, MonthlyCharges, TotalCharges, and Contract were among the top predictive features

