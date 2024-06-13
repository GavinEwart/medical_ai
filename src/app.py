from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('../models/breast_cancer_model.pkl')
scaler = joblib.load('../models/scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    if request.content_type != 'application/json':
        return jsonify({"error": "Invalid Content-Type, must be application/json"}), 415
    
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No input data provided"}), 400
        
        df = pd.DataFrame(data, index=[0])
        df_scaled = scaler.transform(df)
        prediction = model.predict(df_scaled)[0]
        return jsonify({'prediction': int(prediction)})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/status', methods=['GET'])
def status():
    return jsonify({"status": "API is running"}), 200

if __name__ == '__main__':
    app.run(debug=True)
