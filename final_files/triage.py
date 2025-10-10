import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
import joblib

# ----------------------------
# 1. Text Preprocessing
# ----------------------------
def preprocess_text(text):
    """Clean and standardize symptom text."""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    return text

# ----------------------------
# 2. Load and preprocess dataset
# ----------------------------
df = pd.read_csv("triage_dataset.csv")  # columns: symptom, department
df['symptom'] = df['symptom'].apply(preprocess_text)  # Apply preprocessing

# ----------------------------
# 3. Split dataset into training and testing sets
# ----------------------------
X = df["symptom"]
y = df["department"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# 4. Build Pipeline: TF-IDF + Logistic Regression
# ----------------------------
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs", class_weight="balanced")),
])

# Define parameter grid for hyperparameter tuning
param_grid = {
    'tfidf__max_features': [1000, 2000, 3000],
    'tfidf__ngram_range': [(1, 1), (1, 2)],  # Unigrams and bigrams
    'clf__C': [0.1, 1.0, 10.0]  # Regularization strength
}

# ----------------------------
# 5. Hyperparameter tuning with GridSearchCV
# ----------------------------
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1_weighted',  # Optimize for weighted F1-score
    n_jobs=-1,
    verbose=1
)

# Train model with grid search
grid_search.fit(X_train, y_train)

# Best model
model = grid_search.best_estimator_
print("\nBest Parameters:", grid_search.best_params_)
print("Best Cross-Validation F1-Score: {:.4f}".format(grid_search.best_score_))

# ----------------------------
# 6. Evaluate Model
# ----------------------------
y_pred = model.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print evaluation metrics
print("\nModel Evaluation Metrics:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))

# ----------------------------
# 7. Save Model
# ----------------------------
joblib.dump(model, "../hospitalAi/symptom_triage_model.pkl")
print("\nModel trained and saved as symptom_triage_model.pkl")

# ----------------------------
# 8. Rule-based mappings for Urgency & Tests
# ----------------------------
urgency_map = {
    "Cardiology": "High",
    "Neurology": "High",
    "Gastroenterology": "Medium",
    "Dermatology": "Low",
    "Pulmonology": "High",
    "Orthopedics": "Medium",
    "ENT": "Medium",
    "Dentistry": "Low",
    "General Medicine": "Medium"
}

tests_map = {
    "Cardiology": ["ECG", "Blood Pressure", "Echocardiogram"],
    "Neurology": ["MRI", "EEG", "CT Scan"],
    "Gastroenterology": ["Endoscopy", "Ultrasound", "Liver Function Test"],
    "Dermatology": ["Skin Biopsy", "Allergy Test"],
    "Pulmonology": ["X-Ray", "Pulmonary Function Test", "Spirometry"],
    "Orthopedics": ["X-Ray", "MRI", "CT Scan"],
    "ENT": ["Hearing Test", "Nasal Endoscopy", "Throat Examination"],
    "Dentistry": ["Dental X-Ray", "Oral Exam"],
    "General Medicine": ["Blood Test", "Physical Exam"]
}

# ----------------------------
# 9. Test prediction
# ----------------------------
def triage_predict(symptom_text):
    symptom_text = preprocess_text(symptom_text)  # Preprocess input
    dept = model.predict([symptom_text])[0]
    urgency = urgency_map.get(dept, "Medium")
    tests = tests_map.get(dept, [])
    return {
        "department": dept,
        "urgency": urgency,
        "suggested_tests": tests
    }

# Example
if __name__ == "__main__":
    sample = "chest pain and shortness of breath"
    prediction = triage_predict(sample)
    print("\nSample symptom:", sample)
    print("Prediction:", prediction)