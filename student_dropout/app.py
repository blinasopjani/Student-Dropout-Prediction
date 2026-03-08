from flask import Flask, request, jsonify
import joblib
import pandas as pd
from preprocessing import preprocess_new_data

app = Flask(__name__)

# Load models and scaler
rf_binary = joblib.load("models/rf_binary_dropout_model.pkl")
rf_multi = joblib.load("models/rf_multiclass_dropout_model.pkl")
scaler = joblib.load("models/scaler.pkl")

@app.route('/')
def home():
    return "Student Dropout Prediction API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data
        data = request.get_json()
        new_df = pd.DataFrame(data)  # convert to DataFrame

        # Preprocess
        processed_df = preprocess_new_data(new_df, scaler)

        # Predictions
        pred_binary = rf_binary.predict(processed_df)
        pred_multi = rf_multi.predict(processed_df)

        # Return results
        results = {
            "binary_prediction": pred_binary.tolist(),
            "multiclass_prediction": pred_multi.tolist()
        }
        return jsonify(results)
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)