import os
import google.generativeai as genai
import json
import logging
from typing import Dict, Optional
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from werkzeug.utils import secure_filename
from google.generativeai.types import Tool

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION - Edit these values directly
# ============================================================================
class Config:
    """Application configuration"""
    # ADD YOUR GEMINI API KEY HERE
    GEMINI_API_KEY = "AIzaSyBA1W2X0KprDS4AWHJ3D0oz1pZbV9_QJQ4"

    # Gemini model to use
    GEMINI_MODEL = "gemini-2.5-flash"  # Options: gemini-pro, gemini-1.5-pro, gemini-2.0-flash-exp

    # File upload settings
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    UPLOAD_FOLDER = "/tmp"
    ALLOWED_EXTENSIONS = {'pdf'}

    # Server configuration
    HOST = "0.0.0.0"  # Use "127.0.0.1" for localhost only
    PORT = 5000
    DEBUG = True  # Set to False in production


# ============================================================================


# Validate API key
if not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY == "your_api_key_here":
    raise ValueError(
        "Please set your GEMINI_API_KEY in the Config class above. "
        "Get your API key from: https://makersuite.google.com/app/apikey"
    )

# Configure Gemini API
genai.configure(api_key=Config.GEMINI_API_KEY)

# ============================================
# FILE SEARCH SETUP (RAG SYSTEM)
# ============================================

logger.info("Uploading medical knowledge file...")

knowledge_file = genai.upload_file(
    path="medical_knowledge.txt",
    mime_type="text/plain"
)

file_search_tool = Tool.from_file_search(
    files=[knowledge_file.name]
)

logger.info("File Search Tool created successfully")


model = genai.GenerativeModel(
    model_name=Config.GEMINI_MODEL,
    tools=[file_search_tool]
)

logger.info(f"Using Gemini model: {Config.GEMINI_MODEL}")


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def generate_health_summary(
        pdf_file: bytes,
        filename: str,
        patient_name: str,
        patient_age: str,
        patient_gender: str,
        other_details: Optional[str] = None
) -> Dict:
    """
    Generate a concise medical summary from a PDF report using Gemini API.

    Args:
        pdf_file: PDF file as bytes
        filename: Name of the PDF file
        patient_name: Patient's name
        patient_age: Patient's age
        patient_gender: Patient's gender
        other_details: Additional patient information

    Returns:
        Dictionary containing patient info and medical summary
    """
    try:
        # Validate PDF file
        if not allowed_file(filename):
            logger.error(f"Invalid file format for {filename}: Not a PDF")
            return {
                "status": "error",
                "pdf_name": filename,
                "error_message": "Invalid file: Not a valid PDF format.",
                "patient_info": {
                    "name": patient_name,
                    "age": patient_age,
                    "gender": patient_gender,
                    "other_details": other_details or "None"
                }
            }

        # Secure filename
        filename = secure_filename(filename)

        # Write PDF bytes to a temporary file
        temp_pdf_path = os.path.join(Config.UPLOAD_FOLDER, filename)
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_file)

        # Validate file existence
        if not os.path.exists(temp_pdf_path):
            logger.error(f"Temporary PDF file not created: {temp_pdf_path}")
            return {
                "status": "error",
                "pdf_name": filename,
                "error_message": "Error processing PDF: File could not be created.",
                "patient_info": {
                    "name": patient_name,
                    "age": patient_age,
                    "gender": patient_gender,
                    "other_details": other_details or "None"
                }
            }

        # Upload PDF to Gemini
        logger.info(f"Uploading PDF: {temp_pdf_path}")
        uploaded_file = genai.upload_file(path=temp_pdf_path, mime_type="application/pdf")

        # Prepare patient context
        patient_context = (
            f"Patient Information:\n"
            f"- Name: {patient_name}\n"
            f"- Age: {patient_age}\n"
            f"- Gender: {patient_gender}\n"
        )
        if other_details:
            patient_context += f"- Medical History/Notes: {other_details}\n"

        # Create a doctor-focused prompt for concise summary
        prompt = (
            f"{patient_context}\n"
            "You are an expert medical AI assistant.\n\n"

            "IMPORTANT:\n"
            "Use BOTH:\n"
            "1. The uploaded patient PDF report\n"
            "2. The medical reference knowledge base available via file search\n\n"

            "Compare patient values with reference ranges.\n"
            "Identify abnormalities accurately.\n\n"

            "Return ONLY valid JSON:\n"

            "{\n"
            '  "test_type": "string",\n'
            '  "test_date": "string or null",\n'
            '  "key_findings": [],\n'
            '  "abnormal_values": [],\n'
            '  "diagnosis_impression": "string",\n'
            '  "risk_level": "low/moderate/high/critical",\n'
            '  "recommendations": [],\n'
            '  "follow_up_required": "yes/no",\n'
            '  "urgency": "routine/soon/urgent/immediate"\n'
            "}\n"
        )

        logger.info("Sending prompt to Gemini API")
        response = model.generate_content([prompt, uploaded_file])
        summary_text = response.text.strip()

        # Clean up markdown code blocks if present
        if summary_text.startswith("```"):
            summary_text = summary_text.split("```")[1]
            if summary_text.startswith("json"):
                summary_text = summary_text[4:].strip()

        # Log the raw response for debugging
        logger.debug(f"Raw Gemini response: {summary_text}")

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
            # Fallback: Return raw text
            summary_data = {
                "test_type": "Unknown",
                "test_date": None,
                "key_findings": [summary_text[:500]],  # Truncate if too long
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

        # Clean up temporary file and uploaded file
        try:
            genai.delete_file(uploaded_file.name)
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


# Create Flask app
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH
CORS(app)


@app.route('/')
def index():
    """Redirect to health summary page"""
    return render_template('health_summary.html')


@app.route('/health_summary', methods=['GET'])
def health_summary_page():
    """Serve the health summary HTML page"""
    return render_template('health_summary.html')


@app.route('/patient_details', methods=['GET'])
def patient_details_page():
    """Serve the patient details HTML page"""
    return render_template('patient_details.html')


@app.route('/api/generate_summary', methods=['POST'])
def generate_summary_endpoint():
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

        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in generate_summary_endpoint: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "error_message": str(e)
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model": Config.GEMINI_MODEL,
        "version": "1.0.0"
    })


@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle file too large error"""
    return jsonify({
        "status": "error",
        "error_message": f"File too large. Maximum size is {Config.MAX_CONTENT_LENGTH / (1024 * 1024):.0f}MB"
    }), 413


if __name__ == "__main__":
    logger.info(f"Starting Medical Summary API Server")
    logger.info(f"Server will run on http://{Config.HOST}:{Config.PORT}")
    logger.info(f"Debug mode: {Config.DEBUG}")

    app.run(
        debug=Config.DEBUG,
        host=Config.HOST,
        port=Config.PORT
    )