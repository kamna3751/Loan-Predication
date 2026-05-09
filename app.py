from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("model.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():

    income = float(request.form['income'])
    loan = float(request.form['loan'])
    credit = float(request.form['credit'])

    prediction = model.predict([[income, loan, credit]])

    if prediction[0] == 1:
        result = "Loan Approved ✅"
    else:
        result = "Loan Rejected ❌"

    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)