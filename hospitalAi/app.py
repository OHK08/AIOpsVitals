from flask import Flask, request, render_template, jsonify, session
import pandas as pd
import joblib
import numpy as np
import re
import os
from werkzeug.utils import secure_filename
import google.generativeai as genai
import json
from chatbot import get_bot_response
import logging

app = Flask(__name__)
app.secret_key = "secret_key"

# Set up logging for the app
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ==========================
# Gemini Configuration
# ==========================
GEMINI_API_KEY = "AIzaSyBA1W2X0KprDS4AWHJ3D0oz1pZbV9_QJQ4"
GEMINI_MODEL = "gemini-2.5-flash"

# Configure Gemini API
if GEMINI_API_KEY and GEMINI_API_KEY != "your_api_key_here":
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel(GEMINI_MODEL)
    logger.info(f"Using Gemini model: {GEMINI_MODEL}")
else:
    gemini_model = None
    logger.warning("GEMINI_API_KEY not set. Health summary feature will be disabled.")

# ==========================
# Upload Configuration
# ==========================
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
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
# Health Summary Generator
# ==========================
def generate_health_summary(pdf_file, filename, patient_name, patient_age, patient_gender, other_details=None):
    """
    Generate a concise medical summary from a PDF report using Gemini API.
    """
    if not gemini_model:
        return {
            "status": "error",
            "pdf_name": filename,
            "error_message": "Gemini API not configured. Please set GEMINI_API_KEY.",
            "patient_info": {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "other_details": other_details or "None"
            }
        }

    try:
        # Secure filename
        filename = secure_filename(filename)

        # Write PDF bytes to a temporary file
        temp_pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_file)

        # Read PDF content and convert to base64 for Gemini
        logger.info(f"Processing PDF: {temp_pdf_path}")

        # For newer versions of google-generativeai, use direct content generation
        import base64
        with open(temp_pdf_path, 'rb') as pdf:
            pdf_data = base64.standard_b64encode(pdf.read()).decode('utf-8')

        # Create inline data for PDF
        pdf_part = {
            "mime_type": "application/pdf",
            "data": pdf_data
        }

        # Prepare patient context
        patient_context = (
            f"Patient Information:\n"
            f"- Name: {patient_name}\n"
            f"- Age: {patient_age}\n"
            f"- Gender: {patient_gender}\n"
        )
        if other_details:
            patient_context += f"- Medical History/Notes: {other_details}\n"

        # Create prompt for Gemini
        prompt = (
            f"{patient_context}\n"
            "You are a medical assistant helping doctors quickly understand patient reports. "
            "Analyze the medical report in this PDF and provide a CONCISE summary highlighting ONLY the most critical information for doctors.\n\n"
            "Respond with ONLY valid JSON (no markdown, no code blocks, no extra text) in this exact structure:\n"
            "{\n"
            '  "test_type": "string (e.g., HbA1c Test, Blood Panel, X-Ray)",\n'
            '  "test_date": "string or null",\n'
            '  "key_findings": ["critical finding 1", "critical finding 2", "..."],\n'
            '  "abnormal_values": [{"parameter": "string", "value": "string", "normal_range": "string", "status": "high/low/critical"}],\n'
            '  "diagnosis_impression": "string - brief diagnosis or clinical impression",\n'
            '  "risk_level": "low/moderate/high/critical",\n'
            '  "recommendations": ["recommendation 1", "recommendation 2"],\n'
            '  "follow_up_required": "yes/no",\n'
            '  "urgency": "routine/soon/urgent/immediate"\n'
            "}\n\n"
            "Focus on:\n"
            "1. Abnormal or concerning values\n"
            "2. Clinical significance\n"
            "3. Immediate action items\n"
            "Keep it brief and actionable for busy doctors."
        )

        logger.info("Sending prompt to Gemini API")
        response = gemini_model.generate_content([
            prompt,
            pdf_part
        ])
        summary_text = response.text.strip()

        # Clean up markdown code blocks if present
        if summary_text.startswith("```"):
            summary_text = summary_text.split("```")[1]
            if summary_text.startswith("json"):
                summary_text = summary_text[4:].strip()

        # Parse JSON response
        try:
            summary_data = json.loads(summary_text)

            # Ensure all required fields exist with defaults
            summary_data.setdefault("test_type", "Not specified")
            summary_data.setdefault("test_date", None)
            summary_data.setdefault("key_findings", [])
            summary_data.setdefault("abnormal_values", [])
            summary_data.setdefault("diagnosis_impression", "Not provided")
            summary_data.setdefault("risk_level", "unknown")
            summary_data.setdefault("recommendations", [])
            summary_data.setdefault("follow_up_required", "unknown")
            summary_data.setdefault("urgency", "routine")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            summary_data = {
                "test_type": "Unknown",
                "test_date": None,
                "key_findings": [summary_text[:500]],
                "abnormal_values": [],
                "diagnosis_impression": "Unable to parse structured data",
                "risk_level": "unknown",
                "recommendations": ["Review raw report"],
                "follow_up_required": "yes",
                "urgency": "routine",
                "parse_error": True,
                "raw_summary": summary_text
            }

        result = {
            "status": "success",
            "pdf_name": filename,
            "patient_info": {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "other_details": other_details or "None"
            },
            "summary": summary_data
        }

        # Clean up temporary file
        try:
            os.remove(temp_pdf_path)
            logger.info(f"Cleaned up temporary file: {temp_pdf_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file: {e}")

        return result

    except Exception as e:
        logger.error(f"Error processing PDF {filename}: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "pdf_name": filename,
            "error_message": f"Error processing PDF: {str(e)}",
            "patient_info": {
                "name": patient_name,
                "age": patient_age,
                "gender": patient_gender,
                "other_details": other_details or "None"
            }
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


@app.route("/api/generate_summary", methods=["POST"])
def api_generate_summary():
    """API endpoint to generate medical summary"""
    try:
        # Get form data
        patient_name = request.form.get('patient_name', '').strip()
        patient_age = request.form.get('patient_age', '').strip()
        patient_gender = request.form.get('patient_gender', '').strip()
        other_details = request.form.get('other_details', '').strip()

        # Validate required fields
        if not all([patient_name, patient_age, patient_gender]):
            return jsonify({
                "status": "error",
                "error_message": "Name, age, and gender are required"
            }), 400

        # Get uploaded file
        if 'health_report' not in request.files:
            return jsonify({
                "status": "error",
                "error_message": "No file uploaded"
            }), 400

        file = request.files['health_report']

        if file.filename == '':
            return jsonify({
                "status": "error",
                "error_message": "No file selected"
            }), 400

        if not allowed_file(file.filename):
            return jsonify({
                "status": "error",
                "error_message": "File must be a PDF"
            }), 400

        # Read file bytes
        pdf_bytes = file.read()

        # Log request
        logger.info(f"Processing report for patient: {patient_name}, Age: {patient_age}, Gender: {patient_gender}")

        # Generate summary
        result = generate_health_summary(
            pdf_file=pdf_bytes,
            filename=file.filename,
            patient_name=patient_name,
            patient_age=patient_age,
            patient_gender=patient_gender,
            other_details=other_details if other_details else None
        )

        # Store in patient records if successful
        if result.get("status") == "success":
            patient_records.append({
                "patient_info": result["patient_info"],
                "summary": result["summary"],
                "pdf_name": result["pdf_name"]
            })

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in api_generate_summary: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error_message": str(e)
        }), 500


# Legacy endpoint for backward compatibility
@app.route("/submit_health_summary", methods=["POST"])
def submit_health_summary():
    """Legacy endpoint - redirects to new API"""
    try:
        patient_info = {
            "name": request.form.get("patient_name"),
            "age": request.form.get("patient_age"),
            "gender": request.form.get("patient_gender", "Unknown"),
            "other_details": request.form.get("other_details", "")
        }

        if 'health_report' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400

        file = request.files['health_report']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        if file and allowed_file(file.filename):
            pdf_bytes = file.read()
            result = generate_health_summary(
                pdf_file=pdf_bytes,
                filename=file.filename,
                patient_name=patient_info["name"],
                patient_age=patient_info["age"],
                patient_gender=patient_info["gender"],
                other_details=patient_info["other_details"]
            )

            if result.get("status") == "error":
                return jsonify({"error": result.get("error_message")}), 500

            patient_records.append({
                "patient_info": result["patient_info"],
                "summary": result["summary"],
                "pdf_name": result["pdf_name"]
            })

            return jsonify({
                "patient_info": result["patient_info"],
                "summary": result["summary"],
                "pdf_name": result["pdf_name"]
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
MEDICINE_CSV = "medicines_with_chemical.csv"

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

        # 🔹 Updated fields
        chemical = data.get("chemical", "").strip().lower()
        category = data.get("category", "").strip().lower()
        indication = data.get("indication", "").strip().lower()

        if MEDICINES_DF.empty:
            return jsonify([])

        # Convert DF to lowercase for search
        df_lower = MEDICINES_DF.applymap(lambda x: str(x).lower())

        # Flexible filtering (ignore empty filters)
        mask = True

        if chemical:
            mask &= df_lower["chemical_name"].str.contains(chemical, na=False)
        if category:
            mask &= df_lower["category"].str.contains(category, na=False)
        if indication:
            mask &= df_lower["indication"].str.contains(indication, na=False)

        results = MEDICINES_DF[mask]

        return jsonify(results.to_dict(orient="records") if not results.empty else [])

    except Exception as e:
        print("❌ Error in medicine_lookup:", e)
        return jsonify({"error": str(e)}), 500


# ==========================
# Health Check
# ==========================
@app.route("/api/health", methods=["GET"])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "gemini_configured": gemini_model is not None,
        "model": GEMINI_MODEL if gemini_model else "Not configured",
        "version": "1.0.0"
    })


# ==========================
# Error Handlers
# ==========================
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "status": "error",
        "error_message": f"File too large. Maximum size is {app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024):.0f}MB"
    }), 413


# ==========================
# Run App
# ==========================
if __name__ == "__main__":
    app.run(debug=True)