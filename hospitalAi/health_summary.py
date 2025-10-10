import os
import google.generativeai as genai
import json
import logging
from typing import Dict, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Gemini API - Use environment variable for security
GEMINI_API_KEY = "Omshree-API-key"
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable not set. Please set it in your environment.")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-2.5-flash")

def generate_health_summary(
        pdf_file: bytes,
        filename: str,
        patient_info: Optional[Dict[str, str]] = None
) -> Dict:
    try:
        # Validate PDF file
        if not filename.lower().endswith('.pdf'):
            logger.error(f"Invalid file format for {filename}: Not a PDF")
            return {
                "pdf_name": filename,
                "summary": {"error": "Invalid file: Not a valid PDF format."},
                "patient_info": patient_info
            }

        # Write PDF bytes to a temporary file
        temp_pdf_path = f"/tmp/{filename}"
        with open(temp_pdf_path, 'wb') as f:
            f.write(pdf_file)

        # Validate file existence
        if not os.path.exists(temp_pdf_path):
            logger.error(f"Temporary PDF file not created: {temp_pdf_path}")
            return {
                "pdf_name": filename,
                "summary": {"error": "Error processing PDF: File could not be created."},
                "patient_info": patient_info
            }

        # Upload PDF to Gemini
        logger.info(f"Uploading PDF: {temp_pdf_path}")
        uploaded_file = genai.upload_file(path=temp_pdf_path, mime_type="application/pdf")

        # Prepare patient context for prompt if available
        patient_context = ""
        if patient_info:
            patient_context = (
                f"Patient: {patient_info.get('name', 'Unknown')}, "
                f"Age: {patient_info.get('age', 'Unknown')}. "
                f"Other details: {patient_info.get('other_details', 'None')}. "
            )

        # Generate structured summary using Gemini API - Stricter prompt for JSON only
        prompt = (
            f"{patient_context}Analyze the medical report in this PDF and provide a detailed summary. "
            "Respond with ONLY valid JSON in this exact structure (no extra text, no markdown, no explanations): "
            "{"
            '  "chief_complaint": "string or null",'
            '  "medical_history": "string or null",'
            '  "examination_findings": ["string", ...],'
            '  "diagnoses": ["string", ...],'
            '  "treatments": ["string", ...],'
            '  "other_details": ["string", ...]'
            "}"
        )
        logger.info("Sending prompt to Gemini API")
        response = model.generate_content([prompt, uploaded_file])
        summary_text = response.text.strip()

        # Log the raw response for debugging
        logger.info(f"Raw Gemini response: {summary_text}")

        # Parse JSON response
        summary_data = {}
        try:
            summary_data = json.loads(summary_text)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Gemini response as JSON: {e}")
            # Fallback: Return raw text for frontend to handle
            summary_data = {
                "raw_summary": summary_text,
                "parse_error": True
            }

        result = {
            "pdf_name": filename,
            "summary": summary_data,
            "patient_info": patient_info
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
        logger.error(f"Error processing PDF {filename}: {str(e)}")
        return {
            "pdf_name": filename,
            "summary": {"error": f"Error processing PDF: {str(e)}"},
            "patient_info": patient_info
        }