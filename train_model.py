import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("dataset/train.csv")

# Select features
data = data[['ApplicantIncome','LoanAmount','Credit_History','Loan_Status']]

# Convert target
data['Loan_Status'] = data['Loan_Status'].map({'Y':1,'N':0})

# Remove missing values
data.dropna(inplace=True)

X = data[['ApplicantIncome','LoanAmount','Credit_History']]
y = data['Loan_Status']

# Train model
model = LogisticRegression()
model.fit(X, y)

# Save model
joblib.dump(model, "model.pkl")

print("Model trained successfully")