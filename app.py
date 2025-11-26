import os
import logging
import joblib
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash
from forms import TransactionForm

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Create the app
app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key")

# Load your trained model
try:
    model = joblib.load("fraud_detection_pipeline.pkl")
    print("✅ Model loaded successfully!")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def index():
    form = TransactionForm()
    prediction_result = None

    if form.validate_on_submit():
        # Extract form data
        transaction_type = form.transaction_type.data
        amount = float(form.amount.data)
        sender_old_balance = float(form.sender_old_balance.data)
        sender_new_balance = float(form.sender_new_balance.data)
        receiver_old_balance = float(form.receiver_old_balance.data)
        receiver_new_balance = float(form.receiver_new_balance.data)

        # Use your actual trained model
        prediction_result = predict_fraud_with_model(
            transaction_type, amount, sender_old_balance, 
            sender_new_balance, receiver_old_balance, receiver_new_balance
        )

        flash('Prediction completed successfully!', 'success')

    return render_template('index.html', form=form, prediction_result=prediction_result)

def predict_fraud_with_model(transaction_type, amount, sender_old_balance, sender_new_balance, receiver_old_balance, receiver_new_balance):
    """
    Use your trained ML model for fraud prediction
    """
    if model is None:
        # Fallback to mock prediction if model not loaded
        return predict_fraud_fallback(transaction_type, amount, sender_old_balance, sender_new_balance, receiver_old_balance, receiver_new_balance)

    try:
        # Create input data for your model (matching your Streamlit app format)
        input_data = pd.DataFrame([{
            "type": transaction_type,
            "amount": amount,
            "oldbalanceOrg": sender_old_balance,
            "newbalanceOrig": sender_new_balance,
            "oldbalanceDest": receiver_old_balance,
            "newbalanceDest": receiver_new_balance,
        }])

        # Get prediction
        prediction = model.predict(input_data)[0]

        # Get probability if available
        if hasattr(model, 'predict_proba'):
            probability = model.predict_proba(input_data)[0][1]  # Probability of fraud (class 1)
        else:
            probability = 0.8 if prediction == 1 else 0.2  # Default probabilities

        confidence = round(probability * 100, 1)

        return {
            'is_fraud': prediction == 1,
            'confidence': confidence,
            'risk_level': 'High' if probability > 0.7 else 'Medium' if probability > 0.3 else 'Low'
        }

    except Exception as e:
        print(f"Error during prediction: {e}")
        # Fallback to mock prediction
        return predict_fraud_fallback(transaction_type, amount, sender_old_balance, sender_new_balance, receiver_old_balance, receiver_new_balance)

def predict_fraud_fallback(transaction_type, amount, sender_old_balance, sender_new_balance, receiver_old_balance, receiver_new_balance):
    """
    Fallback fraud prediction logic if model fails
    """
    fraud_score = 0

    # Risk factors that increase fraud probability
    if transaction_type in ['CASH_OUT', 'TRANSFER']:
        fraud_score += 0.3

    if amount > 200000:  # Large transactions are riskier
        fraud_score += 0.4

    # Check for suspicious balance changes
    expected_sender_balance = sender_old_balance - amount
    if abs(sender_new_balance - expected_sender_balance) > 1:
        fraud_score += 0.3

    # Check if receiver balance change makes sense
    if receiver_old_balance > 0:  # Only check if receiver had a balance
        expected_receiver_balance = receiver_old_balance + amount
        if abs(receiver_new_balance - expected_receiver_balance) > 1:
            fraud_score += 0.2

    # Round amounts (common in legitimate transactions)
    if amount % 1000 == 0 and amount > 10000:
        fraud_score += 0.1

    # Determine result based on fraud score
    is_fraud = fraud_score > 0.5
    confidence = min(max(fraud_score * 100, 10), 90)  # Between 10-90%

    return {
        'is_fraud': is_fraud,
        'confidence': round(confidence, 1),
        'risk_level': 'High' if fraud_score > 0.7 else 'Medium' if fraud_score > 0.3 else 'Low'
    }

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
