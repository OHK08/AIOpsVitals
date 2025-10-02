import matplotlib.pyplot as plt
import xgboost as xgb
from preprocessing import load_and_preprocess
from sklearn.preprocessing import LabelEncoder

# Load data
X, y, features = load_and_preprocess("ErWaitTime.csv")

# Convert categorical features to numeric using LabelEncoder
X_encoded = X.copy()
for col in X_encoded.select_dtypes(include=["object"]).columns:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))

# Train an XGBoost model (example: predicting total wait time)
target_col = "Total Wait Time (min)"
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    random_state=42
)

model.fit(X_encoded, y[target_col])

# Get feature importance
importances = model.feature_importances_

# Plot feature importance vertically
plt.figure(figsize=(12, 6))
plt.bar(features, importances, color="blue")
plt.ylabel("Importance Score")
plt.xlabel("Features")
plt.title(f"Feature Importance for predicting {target_col}")
plt.xticks(rotation=45, ha='right')  # Rotate x-axis labels for readability
plt.tight_layout()
plt.show()
