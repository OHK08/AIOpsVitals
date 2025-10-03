import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("triage_dataset.csv")  # columns: symptom, department

# ----------------------------
# 2. Build Pipeline: TF-IDF + Logistic Regression
# ----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000, solver="lbfgs")),
])

# ----------------------------
# 3. Train Model
# ----------------------------
X = df["symptom"]
y = df["department"]
model.fit(X, y)

# ----------------------------
# 4. Save Model
# ----------------------------
joblib.dump(model, "../hospitalAi/symptom_triage_model.pkl")
print("✅ Model trained and saved as symptom_triage_model.pkl")

# # ----------------------------
# # 5. Rule-based mappings for Urgency & Tests
# # ----------------------------
# urgency_map = {
#     "Cardiology": "High",
#     "Neurology": "High",
#     "Gastroenterology": "Medium",
#     "Dermatology": "Low",
#     "Pulmonology": "High",
#     "Orthopedics": "Medium",
#     "ENT": "Medium",
#     "Dentistry": "Low",
#     "General Medicine": "Medium"
# }
#
# tests_map = {
#     "Cardiology": ["ECG", "Blood Pressure", "Echocardiogram"],
#     "Neurology": ["MRI", "EEG", "CT Scan"],
#     "Gastroenterology": ["Endoscopy", "Ultrasound", "Liver Function Test"],
#     "Dermatology": ["Skin Biopsy", "Allergy Test"],
#     "Pulmonology": ["X-Ray", "Pulmonary Function Test", "Spirometry"],
#     "Orthopedics": ["X-Ray", "MRI", "CT Scan"],
#     "ENT": ["Hearing Test", "Nasal Endoscopy"],
#     "Dentistry": ["Dental X-Ray", "Oral Exam"],
#     "General Medicine": ["Blood Test", "Physical Exam"]
# }
#
# # ----------------------------
# # 6. Test prediction
# # ----------------------------
# def triage_predict(symptom_text):
#     dept = model.predict([symptom_text])[0]
#     urgency = urgency_map.get(dept, "Medium")
#     tests = tests_map.get(dept, [])
#     return {
#         "department": dept,
#         "urgency": urgency,
#         "suggested_tests": tests
#     }
#
# # Example
# sample = "chest pain and shortness of breath"
# prediction = triage_predict(sample)
# print("Sample symptom:", sample)
# print("Prediction:", prediction)
