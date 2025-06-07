# Healthcare Fraud Detection Project
# Covers: Descriptive, Predictive, and Prescriptive Triggers

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# ------------------ Step 1: Load Mock Data ------------------
# Simulated PMJAY claims data
data = pd.DataFrame({
    'claim_id': range(1, 101),
    'hospital_id': np.random.choice(['H001', 'H002', 'H003'], 100),
    'patient_id': np.random.randint(1000, 1100, 100),
    'admission_date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'claim_amount': np.random.normal(20000, 5000, 100).round(2),
    'procedure_code': np.random.choice(['PROC_A', 'PROC_B', 'PROC_C'], 100),
    'distance_to_hospital_km': np.random.uniform(1, 100, 100),
    'fraud_label': np.random.choice([0, 1], 100, p=[0.85, 0.15])  # 0 = Legit, 1 = Fraud
})

# ------------------ Step 2: Descriptive Triggers ------------------
# Rule 1: High number of claims in a single day by same hospital
rule1 = data.groupby(['hospital_id', 'admission_date']).size().reset_index(name='claim_count')
descriptive_alerts = rule1[rule1['claim_count'] > 3]  # Threshold set to 3

# Rule 2: High claim amount
rule2_alerts = data[data['claim_amount'] > 40000]  # Static threshold

# Combine descriptive alerts
print("\nDescriptive Alerts:")
print(pd.concat([descriptive_alerts, rule2_alerts[['claim_id', 'claim_amount']]], axis=0, ignore_index=True))

# ------------------ Step 3: Predictive Modeling ------------------
# Feature Engineering
features = data[['claim_amount', 'distance_to_hospital_km']]
labels = data['fraud_label']

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print("\nPredictive Model Performance:")
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Add fraud risk score to full dataset
risk_scores = model.predict_proba(features)[:, 1]
data['fraud_risk_score'] = risk_scores

# ------------------ Step 4: Prescriptive Prioritization ------------------
# Create a priority score = fraud risk × claim amount
data['priority_score'] = data['fraud_risk_score'] * data['claim_amount']

# Assume limited capacity: investigate top 10 only
top_cases = data.sort_values('priority_score', ascending=False).head(10)

print("\nTop 10 Prescriptive Priority Cases:")
print(top_cases[['claim_id', 'fraud_risk_score', 'claim_amount', 'priority_score']])
