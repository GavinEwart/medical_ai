import joblib
import pandas as pd

# Load the model
model = joblib.load('../models/breast_cancer_model.pkl')

# Load the fitted scaler
scaler = joblib.load('../models/scaler.pkl')

# Function to make predictions
def predict(input_data):
    df = pd.DataFrame(input_data, index=[0])
    df_scaled = scaler.transform(df)  # Scale the input data
    prediction = model.predict(df_scaled)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    new_patient_data = {
        'feature_1': 20.57, 'feature_2': 17.77, 'feature_3': 132.9, 'feature_4': 1326.0,
        'feature_5': 0.08474, 'feature_6': 0.07864, 'feature_7': 0.0869, 'feature_8': 0.07017,
        'feature_9': 0.1812, 'feature_10': 0.05667, 'feature_11': 0.5435, 'feature_12': 0.7339,
        'feature_13': 3.398, 'feature_14': 74.08, 'feature_15': 0.005225, 'feature_16': 0.01308,
        'feature_17': 0.0186, 'feature_18': 0.0134, 'feature_19': 0.01389, 'feature_20': 0.003532,
        'feature_21': 24.99, 'feature_22': 23.41, 'feature_23': 158.8, 'feature_24': 1956.0,
        'feature_25': 0.1238, 'feature_26': 0.1866, 'feature_27': 0.2416, 'feature_28': 0.1860,
        'feature_29': 0.2750, 'feature_30': 0.08902
    }
    print(f'Prediction: {predict(new_patient_data)}')
