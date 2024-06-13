import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv('../data/wdbc.data', header=None)

# Assign column names based on the dataset description
columns = ['ID', 'Diagnosis'] + [f'feature_{i}' for i in range(1, 31)]
df.columns = columns

# Drop the ID column
df.drop('ID', axis=1, inplace=True)

# Encode the 'Diagnosis' column
le = LabelEncoder()
df['Diagnosis'] = le.fit_transform(df['Diagnosis'])

# Split the data into features and target variable
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Initialize the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(model, X_scaled, y, cv=5)

# Print the cross-validation scores
print(f'Cross-validation scores: {cv_scores}')
print(f'Average cross-validation score: {cv_scores.mean():.2f}')

# Fit the model on the entire dataset
model.fit(X_scaled, y)

# Save the model and the scaler to a file
joblib.dump(model, '../models/breast_cancer_model.pkl')
joblib.dump(scaler, '../models/scaler.pkl')
