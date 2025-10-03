from flask import Flask, request, render_template, jsonify, session
import pandas as pd
import joblib
import numpy as np
import random

# -----------------------------
# Initialize Flask app
# -----------------------------
app = Flask(__name__)
app.secret_key = "secret_key"

# -----------------------------
# Load Models
# -----------------------------
model = joblib.load("tuned_multioutput_xgboost.pkl")   # wait time prediction model
triage_model = joblib.load("symptom_triage_model.pkl")  # symptom → department model

# -----------------------------
# Load knowledge base
# -----------------------------
def load_knowledge_base(file_path="hospital_dmaic.txt"):
    kb = {"Doctor": {}, "Management": {}}
    current_topic = None
    current_role = None
    variant_lines = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("# "):
                if variant_lines and current_role and current_topic:
                    kb[current_role][current_topic].append("\n".join(variant_lines))
                    variant_lines = []
                current_topic = line[2:].strip()
                kb["Doctor"].setdefault(current_topic, [])
                kb["Management"].setdefault(current_topic, [])
                current_role = None
                continue

            if line.startswith("## "):
                if variant_lines and current_role and current_topic:
                    kb[current_role][current_topic].append("\n".join(variant_lines))
                    variant_lines = []
                role_name = line[3:].strip()
                if role_name in kb:
                    current_role = role_name
                continue

            if line.startswith("### Variant"):
                if variant_lines and current_role and current_topic:
                    kb[current_role][current_topic].append("\n".join(variant_lines))
                variant_lines = []
                continue

            if current_topic and current_role is not None:
                variant_lines.append(line)

        if variant_lines and current_role and current_topic:
            kb[current_role][current_topic].append("\n".join(variant_lines))

    for role in kb:
        for topic in kb[role]:
            if not kb[role][topic]:
                kb[role][topic] = ["No data available."]

    return kb


knowledge_base = load_knowledge_base("hospital_dmaic.txt")

# -----------------------------
# Routes
# -----------------------------
@app.route("/")
def homepage():
    return render_template("homepage.html")

@app.route("/form")
def formpage():
    regions = ["Rural", "Urban"]
    urgencies = ["Low", "Medium", "High", "Critical"]
    return render_template("form.html", regions=regions, urgencies=urgencies)

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

        return render_template(
            "predict.html",
            predictions=predictions,
            bottleneck=bottleneck,
            error=None
        )
    except Exception as e:
        return render_template("predict.html", predictions=None, bottleneck=None, error=str(e))

# -----------------------------
# Triage Page + API
# -----------------------------
@app.route("/triage")
def triage_page():
    return render_template("triage.html")

@app.route("/predict_triage", methods=["POST"])
def predict_triage():
    try:
        data = request.get_json()
        symptom_text = data.get("symptom", "")

        if not symptom_text:
            return jsonify({"error": "No symptom provided"}), 400

        prediction = triage_model.predict([symptom_text])[0]

        return jsonify({"department": prediction})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# -----------------------------
# Chatbot pages
# -----------------------------
@app.route("/doctor_chat")
def doctor_chat():
    form_data = session.get("form_data", {})
    return render_template("doctor.html", form_data=form_data)

@app.route("/management_chat")
def management_chat():
    form_data = session.get("form_data", {})
    return render_template("management.html", form_data=form_data)

# -----------------------------
# Chatbot API
# -----------------------------
@app.route("/get_response", methods=["POST"])
def get_response():
    user_input = request.json.get("message")
    role = request.json.get("role")

    matched_topic = None
    for topic in knowledge_base[role]:
        if topic.lower() in user_input.lower():
            matched_topic = topic
            break

    if not matched_topic:
        matched_topic = session.get(f"last_topic_{role}", random.choice(list(knowledge_base[role].keys())))

    variant_list = knowledge_base[role][matched_topic]

    last_index = session.get(f"last_variant_index_{role}_{matched_topic}", -1)
    next_index = (last_index + 1) % len(variant_list)
    session[f"last_variant_index_{role}_{matched_topic}"] = next_index
    session[f"last_topic_{role}"] = matched_topic

    answer = variant_list[next_index]

    emoji_map = {
        "Define": "📝 Define",
        "Measure": "📊 Measure",
        "Analyze": "🔍 Analyze",
        "Improve": "🚀 Improve",
        "Control": "📈 Control",
        "Actionable": "✅ Actionable"
    }
    for key, val in emoji_map.items():
        answer = answer.replace(key, val)

    return jsonify({
        "topic": matched_topic,
        "raw": answer,
        "formatted": answer.replace("\n", "<br>")
    })


# -----------------------------
# Run app
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
