from flask import Flask, request, render_template, jsonify, session
import pandas as pd
import joblib
import numpy as np
import re
import os
from werkzeug.utils import secure_filename
from health_summary import generate_health_summary
from chatbot import get_bot_response
import logging

app = Flask(__name__)
app.secret_key = "secret_key"

# Set up logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Upload Configuration
# ==========================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


# ==========================
# Load Models
# ==========================
model = joblib.load("tuned_multioutput_xgboost.pkl")
triage_model = joblib.load("../hospitalAi/symptom_triage_model.pkl")

# ==========================
# Rule-based Mappings
# ==========================
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
    "ENT": ["Hearing Test", "Nasal Endoscopy"],
    "Dentistry": ["Dental X-Ray", "Oral Exam"],
    "General Medicine": ["Blood Test", "Physical Exam"]
}

# ==========================
# In-Memory Storage
# ==========================
patient_records = []


# ==========================
# Text Preprocessing
# ==========================
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text


# ==========================
# Triage Prediction
# ==========================
def triage_predict(symptom_text):
    symptom_text = preprocess_text(symptom_text)
    dept = triage_model.predict([symptom_text])[0]
    urgency = urgency_map.get(dept, "Medium")
    tests = tests_map.get(dept, [])
    return {
        "department": dept,
        "urgency": urgency,
        "suggested_tests": tests
    }


# ==========================
# Routes: Homepage & Form
# ==========================
@app.route("/")
def homepage():
    return render_template("homepage.html")


@app.route("/form")
def formpage():
    regions = ["Rural", "Urban"]
    urgencies = ["Low", "Medium", "High", "Critical"]
    return render_template("form.html", regions=regions, urgencies=urgencies)


# ==========================
# Wait Time Prediction
# ==========================
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = {
            "Region": request.form.get("Region"),
            "Day of Week": request.form.get("Day of Week"),
            "Season": request.form.get("Season"),
            "Time of Day": request.form.get("Time of Day"),
            "Urgency Level": request.form.get("Urgency Level"),
            "Nurse-to-Patient Ratio": float(request.form.get("Nurse-to-Patient Ratio")),
            "Specialist Availability": int(request.form.get("Specialist Availability")),
            "Facility Size (Beds)": int(request.form.get("Facility Size (Beds)"))
        }

        session["form_data"] = data
        input_df = pd.DataFrame({k: [v] for k, v in data.items()})
        predictions = model.predict(input_df)[0].tolist()

        bottleneck_steps = ["Registration", "Triage", "Medical Professional"]
        bottleneck_idx = int(np.argmax(predictions[1:4]))
        bottleneck = bottleneck_steps[bottleneck_idx]

        return render_template("predict.html",
                               predictions=predictions,
                               bottleneck=bottleneck,
                               error=None)
    except Exception as e:
        return render_template("predict.html", predictions=None, bottleneck=None, error=str(e))


# ==========================
# Triage Pages
# ==========================
@app.route("/triage")
def triage_page():
    return render_template("triage.html")


@app.route("/predict_triage", methods=["POST"])
def predict_triage_api():
    try:
        data = request.get_json()
        symptom_text = data.get("symptom", "")
        if not symptom_text:
            return jsonify({"error": "No symptom provided"}), 400
        return jsonify(triage_predict(symptom_text))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# Chatbot Pages
# ==========================
@app.route("/doctor_chat")
def doctor_chat():
    form_data = session.get("form_data", {})
    return render_template("doctor.html", form_data=form_data)


@app.route("/management_chat")
def management_chat():
    form_data = session.get("form_data", {})
    return render_template("management.html", form_data=form_data)


@app.route("/get_response", methods=["POST"])
def get_response():
    try:
        data = request.json
        user_input = data.get("message")
        role = data.get("role")
        if not user_input or not role:
            return jsonify({"error": "Message and role are required"}), 400
        response = get_bot_response(user_input, role)
        return jsonify({
            "topic": "General",
            "raw": response["raw"],
            "formatted": response["formatted"]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ==========================
# Health Summary
# ==========================
@app.route("/health_summary")
def health_summary_page():
    return render_template("health_summary.html")


@app.route("/submit_health_summary", methods=["POST"])
def submit_health_summary():
    try:
        patient_info = {
            "name": request.form.get("patient_name"),
            "age": request.form.get("patient_age"),
            "other_details": request.form.get("other_details", "")
        }

        if 'health_report' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['health_report']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_content = file.read()  # Read file as bytes
            summary_result = generate_health_summary(file_content, filename, patient_info)

            # Log the result for debugging
            logger.info(f"Generated summary result: {summary_result}")

            if 'error' in summary_result.get('summary', {}):
                return jsonify({"error": summary_result['summary']['error']}), 500

            patient_records.append({
                "patient_info": patient_info,
                "summary": summary_result["summary"],
                "pdf_name": summary_result["pdf_name"]
            })

            return jsonify({
                "patient_info": patient_info,
                "summary": summary_result["summary"],
                "pdf_name": summary_result["pdf_name"]
            })
        else:
            return jsonify({"error": "Invalid file format. Only PDFs are allowed."}), 400

    except Exception as e:
        logger.error(f"Error in submit_health_summary: {str(e)}")
        return jsonify({"error": f"Error processing request: {str(e)}"}), 500


# ==========================
# Patient Details
# ==========================
@app.route("/patient_details")
def patient_details():
    return render_template("patient_details.html", patients=patient_records)


@app.route("/api/patient_details", methods=["GET"])
def api_patient_details():
    try:
        return jsonify(patient_records)
    except Exception as e:
        return jsonify({"error": f"Error fetching patient data: {str(e)}"}), 500


# ==========================
# Medicine Lookup
# ==========================
MEDICINE_CSV = "medicine_dataset.csv"
if os.path.exists(MEDICINE_CSV):
    MEDICINES_DF = pd.read_csv(MEDICINE_CSV)
    MEDICINES_DF.columns = [c.strip().lower() for c in MEDICINES_DF.columns]
    MEDICINES_DF = MEDICINES_DF.fillna("")
else:
    MEDICINES_DF = pd.DataFrame()


@app.route("/medicine_lookup")
def medicine_lookup_page():
    return render_template("medicine_lookup.html")


@app.route("/api/medicine_lookup", methods=["POST"])
def api_medicine_lookup():
    try:
        data = request.get_json()
        name = data.get("name", "").strip().lower()
        category = data.get("category", "").strip().lower()
        indication = data.get("indication", "").strip().lower()

        if MEDICINES_DF.empty:
            return jsonify([])

        df_lower = MEDICINES_DF.applymap(lambda x: str(x).lower())

        mask = (
                df_lower["name"].str.contains(name, na=False) &
                df_lower["category"].str.contains(category, na=False) &
                df_lower["indication"].str.contains(indication, na=False)
        )

        results = MEDICINES_DF[mask]
        return jsonify(results.to_dict(orient="records") if not results.empty else [])

    except Exception as e:
        print("Error in medicine_lookup:", e)
        return jsonify({"error": str(e)}), 500


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)