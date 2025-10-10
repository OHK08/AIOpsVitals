import os
import re
import threading
import warnings
import string
import random
from typing import Dict, Optional, List

# --- NLP deps ---
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# -------------------------------------------------------
# One-time NLTK setup (safe to call multiple times)
# -------------------------------------------------------
def _ensure_nltk():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")

_ensure_nltk()

# -------------------------------------------------------
# Config
# -------------------------------------------------------
KB_PATH = os.environ.get("HOSPITAL_KB_PATH", "hospital_dmaic.txt")

GREETINGS_INPUTS = {"hello", "hi", "greetings", "sup", "hey", "good morning", "good afternoon", "good evening"}

GREETINGS_RESPONSES = [
    "Greetings! I can help with DMAIC-based solutions in hospitals. What bottleneck are you facing?",
    "Hello! How can I assist with hospital optimization today?",
    "Hi there! Let’s work on improving your hospital process. What’s the issue?",
]

BYE_INPUTS = {"bye", "exit", "quit", "goodbye", "see you"}

BYE_RESPONSES = [
    "Goodbye! Apply DMAIC to sustain hospital process improvements.",
    "See you later! Keep optimizing those healthcare processes!",
    "Bye! Let me know if you need more help with hospital efficiency."
]

INTENT_KEYWORDS: Dict[str, List[str]] = {
    "registration bottleneck": [
        "register", "registration", "checkin", "check in", "front desk", "admission",
        "enroll", "enrollment", "kiosk", "id scan", "preregister", "preregistration",
        "token", "mrn", "counter", "queue at counter"
    ],
    "triage delays": [
        "triage", "assess", "assessment", "acuity", "priority", "fast track",
        "quick look", "ed triage", "emergency triage", "triage nurse"
    ],
    "lab delays": [
        "lab", "laboratory", "test", "blood", "blood work", "report", "result",
        "tat", "turnaround", "pathology", "sample", "cbc", "bmp", "culture",
        "pneumatic tube"
    ],
    "consultation delays": [
        "doctor", "consult", "consultation", "appointment", "opd", "clinic",
        "physician", "specialist", "token wait", "consult wait", "doctor wait"
    ],
    "discharge delays": [
        "discharge", "bill", "billing", "pharmacy", "exit", "summary", "checkout",
        "dc summary", "bed release", "bed turnover", "clearance", "lounge"
    ],
    "overall wait time": [
        "wait", "waiting", "overall wait", "reduce wait", "bottleneck",
        "process improvement", "dmaic", "lean", "patient flow", "vsm",
        "pareto", "kaizen"
    ],
    "patient care optimization": [
        "patient care", "care delivery", "care quality", "patient outcome",
        "patient satisfaction", "care coordination", "ehr", "patient trust"
    ],
    "handle patient load": [
        "patient load", "patient volume", "patient surge", "high load",
        "patient influx", "staff load", "resource strain", "patient flow"
    ],
}

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
_LEM = nltk.stem.WordNetLemmatizer()
_REMOVE_PUNCT = str.maketrans({p: " " for p in string.punctuation})
_LOCK = threading.RLock()

def _normalize_for_match(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    print(f"Normalized text: '{text}'")  # Debug
    return text

def _lem_tokens(tokens: List[str]) -> List[str]:
    return [_LEM.lemmatize(t) for t in tokens]

def _lem_normalize(text: str) -> List[str]:
    text = text.lower().translate(_REMOVE_PUNCT)
    return _lem_tokens(nltk.word_tokenize(text))

# -------------------------------------------------------
# Variant Rotation State
# -------------------------------------------------------
_VARIANT_COUNTERS: Dict[str, int] = {}  # Tracks variant index for each intent-role pair

def _get_next_variant_index(intent: str, role: str) -> int:
    key = f"{intent}:{role}"
    with _LOCK:
        current_index = _VARIANT_COUNTERS.get(key, -1)
        next_index = (current_index + 1) % 4  # Rotate 0,1,2,3,0,...
        _VARIANT_COUNTERS[key] = next_index
        print(f"Variant rotation: intent={intent}, role={role}, index={next_index}")  # Debug
        return next_index

# -------------------------------------------------------
# Knowledge Base loading/parsing
# -------------------------------------------------------
def _load_kb_by_headers(path: str) -> Dict[str, Dict[str, List[str]]]:
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        text = f.read()

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    kb: Dict[str, Dict[str, List[str]]] = {}
    current_intent: Optional[str] = None
    current_role: Optional[str] = None
    current_variant: Optional[str] = None
    current_lines: List[str] = []

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        if line.startswith("# "):  # Intent (e.g., # Registration Bottleneck)
            if current_intent and current_role and current_variant and current_lines:
                kb.setdefault(current_intent, {}).setdefault(current_role, []).append("\n".join(current_lines).strip())
            current_intent = line[2:].strip().lower()
            current_role = None
            current_variant = None
            current_lines = []
        elif line.startswith("## "):  # Role (e.g., ## Doctor, ## Management)
            if current_intent and current_role and current_variant and current_lines:
                kb.setdefault(current_intent, {}).setdefault(current_role, []).append("\n".join(current_lines).strip())
            current_role = line[2:].strip().lower()
            current_variant = None
            current_lines = []
        elif line.startswith("### "):  # Variant (e.g., ### Variant 1)
            if current_intent and current_role and current_variant and current_lines:
                kb.setdefault(current_intent, {}).setdefault(current_role, []).append("\n".join(current_lines).strip())
            current_variant = line[3:].strip().lower()
            current_lines = []  # Exclude the ### Variant X header
        else:
            current_lines.append(line)

    # Store the last variant
    if current_intent and current_role and current_variant and current_lines:
        kb.setdefault(current_intent, {}).setdefault(current_role, []).append("\n".join(current_lines).strip())

    # Cleanup: Remove empty entries
    kb = {intent: {role: variants for role, variants in roles.items() if variants}
          for intent, roles in kb.items() if roles}
    print(f"Loaded KB intents: {list(kb.keys())}")  # Debug
    return kb

# State
_KB: Dict[str, Dict[str, List[str]]] = {}
_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_MATRIX = None
_CHUNKS: List[str] = []
_CHUNK_METADATA: List[Dict[str, str]] = []

def _build_vector_index():
    global _VECTORIZER, _TFIDF_MATRIX, _CHUNKS, _CHUNK_METADATA
    _CHUNKS = []
    _CHUNK_METADATA = []
    for intent, roles in _KB.items():
        for role, variants in roles.items():
            for idx, variant in enumerate(variants):
                _CHUNKS.append(variant)
                _CHUNK_METADATA.append({"intent": intent, "role": role, "variant_idx": str(idx)})
    if not _CHUNKS:
        _VECTORIZER = None
        _TFIDF_MATRIX = None
        print("No chunks for TF-IDF indexing")  # Debug
        return
    _VECTORIZER = TfidfVectorizer(tokenizer=_lem_normalize, stop_words="english")
    _TFIDF_MATRIX = _VECTORIZER.fit_transform(_CHUNKS)
    print(f"TF-IDF index built with {len(_CHUNKS)} chunks")  # Debug

def reload_kb(path: Optional[str] = None) -> None:
    global _KB
    kb_path = path or KB_PATH
    with _LOCK:
        _KB = _load_kb_by_headers(kb_path)
        _build_vector_index()

# Initial load
reload_kb(KB_PATH)

# -------------------------------------------------------
# Routing + Retrieval
# -------------------------------------------------------
def _is_greeting(user_text: str) -> bool:
    normalized_text = _normalize_for_match(user_text)
    clean_text = normalized_text.translate(str.maketrans("", "", string.punctuation))
    print(f"Checking greeting: raw='{user_text}', normalized='{normalized_text}', clean='{clean_text}' against {GREETINGS_INPUTS}")  # Debug
    words = clean_text.split()
    is_greeting = (clean_text in GREETINGS_INPUTS or
                   any(word == greeting for word in words for greeting in GREETINGS_INPUTS) or
                   any(greeting in clean_text for greeting in GREETINGS_INPUTS))
    print(f"Greeting detected: {is_greeting}")  # Debug
    return is_greeting

def _is_bye(user_text: str) -> bool:
    normalized_text = _normalize_for_match(user_text)
    clean_text = normalized_text.translate(str.maketrans("", "", string.punctuation))
    print(f"Checking bye: raw='{user_text}', normalized='{normalized_text}', clean='{clean_text}' against {BYE_INPUTS}")  # Debug
    words = clean_text.split()
    is_bye = (clean_text in BYE_INPUTS or
              any(word == bye for word in words for bye in BYE_INPUTS) or
              any(bye in clean_text for bye in BYE_INPUTS))
    print(f"Bye detected: {is_bye}")  # Debug
    return is_bye

def _intent_route(user_text: str, role: str) -> Optional[str]:
    text = _normalize_for_match(user_text)
    tokens = set(_lem_normalize(text))  # Lemmatized tokens for matching
    print(f"Intent routing tokens: {tokens}, role: {role}")  # Debug

    best_intent, best_score = None, 0.0
    for intent, kws in INTENT_KEYWORDS.items():
        lem_kws = set(_LEM.lemmatize(kw) for kw in kws)
        score = sum(1 for token in tokens if any(token in lem_kw or lem_kw in token for lem_kw in lem_kws))
        score = score / max(len(lem_kws), 1)
        print(f"Intent '{intent}' score: {score}")  # Debug
        if score > best_score and score >= 0.2:
            best_intent, best_score = intent, score

    if best_intent and best_intent in _KB and role in _KB[best_intent]:
        print(f"Selected intent: {best_intent}, role: {role}")  # Debug
        variants = _KB[best_intent][role]
        variant_index = _get_next_variant_index(best_intent, role)
        if variant_index < len(variants):
            return variants[variant_index]
        return variants[0]  # fallback
    print("No intent matched, falling back to TF-IDF")  # Debug
    return None


def _retrieval_fallback(user_text: str, role: str) -> str:
    if not _CHUNKS or _VECTORIZER is None or _TFIDF_MATRIX is None:
        print("No knowledge base chunks available for TF-IDF")  # Debug
        return ("I’m sorry, I don’t fully understand. Could you clarify if this is about "
                "registration, triage, lab, consultation, discharge, wait times, patient care, or patient load?")

    user_vec = _VECTORIZER.transform([user_text])
    vals = cosine_similarity(user_vec, _TFIDF_MATRIX)
    idx = vals.argsort()[0][-1]
    best_score = vals[0, idx]
    selected_metadata = _CHUNK_METADATA[idx]
    intent = selected_metadata["intent"]
    role_in_chunk = selected_metadata["role"]
    print(f"TF-IDF fallback score: {best_score}, role: {role}, matched chunk: {selected_metadata}")  # Debug

    if best_score <= 0.3:
        print("TF-IDF score too low, returning clarification prompt")  # Debug
        return ("I’m sorry, I don’t fully understand. Could you clarify if this is about "
                "registration, triage, lab, consultation, discharge, wait times, patient care, or patient load?")

    # Apply variant rotation if role matches
    if intent in _KB and role_in_chunk.lower() == role.lower():
        variants = _KB[intent][role.lower()]
        variant_index = _get_next_variant_index(intent, role)
        if variant_index < len(variants):
            return variants[variant_index]
        return variants[0]  # fallback

    # Otherwise return TF-IDF chunk as-is
    return _CHUNKS[idx]

# -------------------------------------------------------
# Public API
# -------------------------------------------------------
def get_bot_response(user_text: str, role: Optional[str] = None) -> dict:
    if not user_text or not user_text.strip():
        raw_response = "Please share a brief description of the bottleneck you’re facing."
        print("Empty input received")  # Debug
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    user = user_text.strip()
    role = role.lower() if role else "doctor"
    print(f"Processing input: '{user}', role: {role}")  # Debug

    # Handle greetings
    if _is_greeting(user):
        raw_response = random.choice(GREETINGS_RESPONSES)
        print(f"Greeting detected, response: '{raw_response}'")  # Debug
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    # Handle bye
    if _is_bye(user):
        raw_response = random.choice(BYE_RESPONSES)
        print(f"Bye detected, response: '{raw_response}'")  # Debug
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    if user.lower() in {"thanks", "thank you", "thx"}:
        raw_response = "You’re welcome! Let’s keep improving healthcare efficiency with DMAIC."
        print(f"Thanks detected, response: '{raw_response}'")  # Debug
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    # 1) Intent route
    routed = _intent_route(user, role)
    if routed:
        print(f"Intent routed response: '{routed[:50]}...'")  # Debug
        return {"raw": routed, "formatted": format_response(routed)}

    # 2) TF-IDF fallback
    raw_response = _retrieval_fallback(user_text, role)
    print(f"Fallback response: '{raw_response[:50]}...'")  # Debug
    return {"raw": raw_response, "formatted": format_response(raw_response)}

def format_response(text):
    # Replace markdown-style headers with emojis + bold
    text = text.replace("# ", "📌 ").replace("## ", "💡 ").replace("### ", "🔹 ")
    text = text.replace("Problem:", "📌 Problem:")
    text = text.replace("DMAIC Steps:", "⚙️ DMAIC Steps:")
    text = text.replace("Define:", "📝 Define:")
    text = text.replace("Measure:", "⏱ Measure:")
    text = text.replace("Analyze:", "🔍 Analyze:")
    text = text.replace("Improve:", "🚀 Improve:")
    text = text.replace("Control:", "📊 Control:")
    text = text.replace("Actionable:", "✅ Actionable:")

    # Replace dashes with bullet emojis
    text = text.replace("- ", "• ")

    # Add <br> for newlines (for HTML chat rendering)
    text = text.replace("\n", "<br>")

    return text