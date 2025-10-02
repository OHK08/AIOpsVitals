import os
import re
import threading
import warnings
import string
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

GREETINGS_INPUTS = {"hello", "hi", "greetings", "sup", "hey"}

GREETINGS_RESPONSES = [
    "Greetings! I can help with Six Sigma DMAIC in hospitals.",
    "Hello, how can I assist with reducing hospital wait times?",
    "Hi, let’s work on improving your hospital process.",
]

# Expanded synonyms for routing
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "registration bottleneck": [
        "registration", "check-in", "check in", "front desk", "admission",
        "enrollment", "kiosk", "id scan", "pre-registration", "preregistration",
        "token", "mrn", "counter", "queue at counter"
    ],
    "triage delays": [
        "triage", "initial assessment", "acuity", "priority", "fast track",
        "quick look", "ed triage", "emergency triage", "triage nurse"
    ],
    "lab results delay": [
        "lab", "labs", "laboratory", "test", "tests", "bloods", "blood work",
        "report", "results", "tat", "turnaround", "pathology", "sample",
        "cbc", "bmp", "culture", "pneumatic tube"
    ],
    "doctor consultation wait": [
        "doctor", "consult", "consultation", "appointment", "opd", "clinic",
        "queue to see doctor", "physician", "specialist", "token wait",
        "consult wait", "doctor wait"
    ],
    "discharge delays": [
        "discharge", "billing", "pharmacy", "exit", "summary", "checkout",
        "dc summary", "bed release", "bed turnover", "clearance", "lounge"
    ],
    "overall wait time": [
        "overall wait", "reduce wait", "waiting time", "bottleneck",
        "process improvement", "dmaic", "lean", "patient flow",
        "vsm", "pareto", "kaizen"
    ],
}

# -------------------------------------------------------
# Utilities
# -------------------------------------------------------
_LEM = nltk.stem.WordNetLemmatizer()
_REMOVE_PUNCT = str.maketrans({p: " " for p in string.punctuation})
_LOCK = threading.RLock()

def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())

def _lem_tokens(tokens: List[str]) -> List[str]:
    return [_LEM.lemmatize(t) for t in tokens]

def _lem_normalize(text: str) -> List[str]:
    text = text.lower().translate(_REMOVE_PUNCT)
    return _lem_tokens(nltk.word_tokenize(text))

# -------------------------------------------------------
# Knowledge Base loading/parsing
# -------------------------------------------------------
def _load_kb_by_headers(path: str) -> Dict[str, str]:
    with open(path, "r", encoding="utf-8", errors="strict") as f:
        text = f.read()

    text = text.replace("\r\n", "\n").replace("\r", "\n")

    blocks: Dict[str, str] = {}
    current_header: Optional[str] = None
    current_lines: List[str] = []

    for line in text.split("\n"):
        if line.startswith("## "):  # header line
            if current_header is not None:
                # store previous block (keep header line inside for display context)
                blocks[current_header] = "\n".join(current_lines).strip()
            current_header = line[3:].strip().lower()  # header sans '## '
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_header is not None:
        blocks[current_header] = "\n".join(current_lines).strip()

    # cleanup
    blocks = {k: v for k, v in blocks.items() if v.strip()}
    return blocks

# State
_KB: Dict[str, str] = {}
_VECTORIZER: Optional[TfidfVectorizer] = None
_TFIDF_MATRIX = None
_CHUNKS: List[str] = []

def _build_vector_index():
    """Build TF-IDF index over full sections for retrieval fallback."""
    global _VECTORIZER, _TFIDF_MATRIX, _CHUNKS
    _CHUNKS = list(_KB.values())
    if not _CHUNKS:
        _VECTORIZER = None
        _TFIDF_MATRIX = None
        return
    _VECTORIZER = TfidfVectorizer(tokenizer=_lem_normalize, stop_words="english")
    _TFIDF_MATRIX = _VECTORIZER.fit_transform(_CHUNKS)

def reload_kb(path: Optional[str] = None) -> None:
    """Reload KB from disk and rebuild retrieval index."""
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
    words = _normalize_for_match(user_text).split()
    return any(w in GREETINGS_INPUTS for w in words)

def _intent_route(user_text: str) -> Optional[str]:
    text = _normalize_for_match(user_text)
    best_intent, best_score = None, 0
    for intent, kws in INTENT_KEYWORDS.items():
        score = 0
        for kw in kws:
            # word-boundary match to avoid partials
            if re.search(rf"\b{re.escape(kw)}\b", text):
                score += 1
        if score > best_score:
            best_intent, best_score = intent, score
    if best_intent and best_intent in _KB:
        return _KB[best_intent]
    return None

def _retrieval_fallback(user_text: str) -> str:
    if not _CHUNKS or _VECTORIZER is None or _TFIDF_MATRIX is None:
        return ("I’m sorry, I don’t fully understand. Could you clarify if this is about "
                "registration, triage, lab, consultation, discharge, or overall wait time?")
    user_vec = _VECTORIZER.transform([user_text])
    vals = cosine_similarity(user_vec, _TFIDF_MATRIX)  # shape (1, n_chunks)
    idx = vals.argsort()[0][-1]  # best match
    best_score = vals[0, idx]
    if best_score <= 0:
        return ("I’m sorry, I don’t fully understand. Could you clarify if this is about "
                "registration, triage, lab, consultation, discharge, or overall wait time?")
    return _CHUNKS[idx]

# -------------------------------------------------------
# Public API
# -------------------------------------------------------
def get_bot_response(user_text: str) -> dict:
    """
    Returns a dictionary with raw and formatted response strings for the given user input.
    Designed to be imported from Flask: from chatbot import get_bot_response
    """
    if not user_text or not user_text.strip():
        raw_response = "Please share a brief description of the bottleneck you’re facing."
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    user = user_text.strip().lower()

    if user in {"bye", "exit", "quit"}:
        raw_response = "Goodbye! Apply DMAIC to sustain hospital process improvements."
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    if user in {"thanks", "thank you", "thx"}:
        raw_response = "You’re welcome! Let’s keep improving healthcare efficiency with DMAIC."
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    if _is_greeting(user):
        raw_response = GREETINGS_RESPONSES[0]
        return {"raw": raw_response, "formatted": format_response(raw_response)}

    # 1) Intent route
    routed = _intent_route(user)
    if routed:
        return {"raw": routed, "formatted": format_response(routed)}

    # 2) TF-IDF fallback
    raw_response = _retrieval_fallback(user_text)
    return {"raw": raw_response, "formatted": format_response(raw_response)}

def format_response(text):
    # Replace markdown-style headings with emojis + bold
    text = text.replace("## ", "💡 ").replace("Problem:", "📌 Problem:")
    text = text.replace("DMAIC Steps:", "⚙️ DMAIC Steps:")
    text = text.replace("Define:", "📝 Define:")
    text = text.replace("Measure:", "⏱ Measure:")
    text = text.replace("Analyze:", "🔍 Analyze:")
    text = text.replace("Improve:", "🚀 Improve:")
    text = text.replace("Control:", "📊 Control:")
    text = text.replace("Sustain:", "🔄 Sustain:")

    # Replace dashes with bullet emojis
    text = text.replace("- ", "• ")

    # Add <br> for newlines (for HTML chat rendering)
    text = text.replace("\n", "<br>")

    return text