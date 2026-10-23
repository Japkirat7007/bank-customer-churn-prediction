#Bank Customer Churn Prediction

An end-to-end machine learning project that predicts which bank customers are likely to leave, groups them into risk levels, and serves the results in an interactive web app.

Live app: ADD YOUR STREAMLIT LINK HERE

App screenshot

Problem

Losing customers is expensive for a bank. The goal is to identify customers at risk of leaving early, so the bank can spend its retention budget on the right people instead of contacting everyone.

Dataset
Public Churn Modelling dataset, 10,000 bank customers, 14 columns
Target: Exited (1 = customer left, 0 = stayed)
About 20.4% of customers churned, so the data is imbalanced
No missing values or duplicate rows
Key findings from the analysis
Age is the strongest driver: churn is about 7.5% for ages 18-30, 34% for 41-50 and 56% for 51-60.
Number of products: customers with 3 products churn about 83% of the time, and all customers with 4 products left. Customers with 2 products are the safest (about 8%).
Country: Germany churns at about 32%, double France (16%) and Spain (17%).
Activity: inactive members churn at about 27%, against 14% for active members.
Gender: female customers churn at about 25%, male customers at about 16%.
Approach
Data cleaning: dropped ID columns (RowNumber, CustomerId, Surname) that carry no signal.
Exploratory data analysis: churn rate by country, gender, age group, number of products and activity status.
Feature engineering: BalanceSalaryRatio, ProductsPerTenure, ZeroBalance flag and age groups; one-hot encoding for country and gender.
Train-test split: stratified 80/20 split to keep the same churn rate in both sets.
Modelling: Logistic Regression (baseline) and Random Forest.
Imbalance handling: class weights in the Random Forest.
Threshold tuning: lowered the decision threshold from 0.50 to 0.30 to catch more churners, because missing a churner costs the bank more than a wasted retention offer.
Explainability: feature importance and SHAP values.
Business layer: Low, Medium and High risk segments.
Deployment: Streamlit web app with a gauge chart, risk card and data insights.
Results
Model	Accuracy	Precision	Recall	F1	ROC-AUC
Logistic Regression	0.808	0.595	0.184	0.281	0.774
Random Forest	0.869	0.813	0.459	0.587	0.853

Final model: Random Forest with class weights and a 0.30 threshold, giving ROC-AUC of about 0.855, recall of about 0.66, precision of about 0.57 and F1 of about 0.61.

Accuracy alone is misleading here, because predicting "no churn" for every customer already scores about 80%.

Risk segments on the test set (2,000 customers)
Risk level	Customers	Real churn rate
Low (below 30%)	1,544	9%
Medium (30-60%)	296	42%
High (above 60%)	160	88%

Targeting the 467 Medium and High risk customers reaches 267 of the 407 customers who actually left.

Retention ideas
Personal outreach and offers for High-risk customers
Loyalty benefits for customers aged 41-60
Investigate why churn in Germany is double that of other countries
Review product bundles, since customers with 3-4 products almost always leave
Re-engagement campaigns for inactive members
Project structure
bank-customer-churn-prediction/
├── data/
│   └── Churn_Modelling.csv
├── notebooks/
│   └── churn_analysis.ipynb
├── app/
│   ├── app.py
│   ├── requirements.txt
│   ├── churn_model.pkl
│   └── model_columns.pkl
└── README.md
Run it locally
git clone https://github.com/YOUR-USERNAME/bank-customer-churn-prediction.git
cd bank-customer-churn-prediction/app
pip install -r requirements.txt
streamlit run app.py
Tech stack

Python, pandas, NumPy, scikit-learn, SHAP, Matplotlib, Plotly, Streamlit, Joblib

Limitations
The dataset is a public snapshot with no dates or transaction history, so time-based analysis is not possible.
It appears to be synthetic, so the findings should not be treated as real-world banking conclusions.
Account balance is used as a simple stand-in for customer value.
The "key factors" shown in the app are rules based on the data analysis, not per-customer model explanations.
Author
Japkirat kaur | B.Tech CSE | LinkedIn | GitHub
