# ================= IMPORTS =================
from data_loader import download_files

import json
import os
import threading
from datetime import datetime

# ================= GLOBALS =================
domain_classifier = None
risk_detector = None
control_policy = None
llm = None
validator = None
hallucination_detector = None
orchestrator = None
logger = None

retriever = None

data_loaded = False
system_loaded = False
is_loading = False

current_domain = None


# ================= LAZY LLM =================
def get_llm():
    global llm

    if llm is None:
        print("🚀 Loading LLM...")

        from models.llm_interface import LLMInterface

        llm = LLMInterface()

        print("✅ LLM Loaded")

    return llm


# ================= CORE INIT =================
def initialize_core():
    global domain_classifier
    global risk_detector
    global control_policy
    global validator
    global hallucination_detector
    global orchestrator
    global logger
    global system_loaded

    if system_loaded:
        return

    print("⚡ Initializing core system...")

    from core.domain_classifier import DomainClassifier
    from core.risk_detector import RiskDetector
    from core.control_policy import ControlPolicy
    from core.prompt_orchestrator import PromptOrchestrator
    from core.response_validator import ResponseValidator
    from core.hallucination_detector import HallucinationDetector
    from evaluation.logger import Logger

    domain_classifier = DomainClassifier()
    risk_detector = RiskDetector()
    control_policy = ControlPolicy()

    validator = ResponseValidator()
    hallucination_detector = HallucinationDetector()
    orchestrator = PromptOrchestrator()

    logger = Logger("framework_results.csv")

    system_loaded = True

    print("✅ Core system ready")


# ================= RAG INIT =================
def initialize_rag():
    global retriever
    global data_loaded
    global is_loading

    if data_loaded or is_loading:
        return

    is_loading = True

    try:

        print("🚀 Loading Retriever...")

        from rag.retriever import Retriever

        if not os.path.exists("data/kombucha_index.faiss"):
            print("⬇️ Downloading FAISS files...")
            download_files()

        retriever = Retriever()

        data_loaded = True

        print("✅ Retriever ready")

    except Exception as e:
        print("❌ RAG INIT ERROR:", e)

    finally:
        is_loading = False


# ================= BACKGROUND STARTUP =================
def startup_background_loader():
    try:
        initialize_core()
        initialize_rag()

        print("🔥 FULL SYSTEM READY")

    except Exception as e:
        print("❌ STARTUP ERROR:", e)


threading.Thread(
    target=startup_background_loader,
    daemon=True
).start()


# ================= MEMORY =================
def save_conversation(query, response):

    os.makedirs("data/conversation_memory", exist_ok=True)

    filename = "data/conversation_memory/memory_log.json"

    entry = {
        "timestamp": str(datetime.now()),
        "query": query,
        "response": response
    }

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)

                if not isinstance(data, list):
                    data = []

        except:
            data = []

    else:
        data = []

    data.append(entry)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# ================= FEEDBACK =================
def save_feedback(query, response, feedback_type):

    os.makedirs("data/feedback", exist_ok=True)

    filename = "data/feedback/feedback_log.json"

    entry = {
        "timestamp": str(datetime.now()),
        "query": query,
        "response": response,
        "feedback": feedback_type
    }

    if os.path.exists(filename):
        try:
            with open(filename, "r") as f:
                data = json.load(f)

                if not isinstance(data, list):
                    data = []

        except:
            data = []

    else:
        data = []

    data.append(entry)

    with open(filename, "w") as f:
        json.dump(data, f, indent=4)


# ================= FOLLOWUPS =================
def generate_followups(response):

    initialize_core()

    prompt = f"""
Based on this kombucha explanation:

{response}

Suggest exactly 3 short relevant follow-up questions.

Return ONLY the questions.
"""

    try:

        result = get_llm().generate(prompt, temperature=0.7)

        suggestions = []

        for line in result.split("\n"):

            line = line.strip()

            if line:
                cleaned = line.split(".", 1)[-1].strip()
                suggestions.append(cleaned)

        return suggestions[:3]

    except:
        return []


# ================= FILE ANALYZER =================
def analyze_uploaded_file(file_text, user_query=None):

    initialize_core()

    prompt = f"""
You are K-GPT.

Answer ONLY kombucha-related queries.

Document:
{file_text}

Question:
{user_query}
"""

    return get_llm().generate(prompt, temperature=0.3)


# ================= MAIN QUERY =================
def process_query(query):

    global current_domain

    initialize_core()

    if not data_loaded:

        if is_loading:
            return (
                "⏳ AI system is still starting. Please wait 1 minute.",
                [],
                []
            )

        initialize_rag()

        return (
            "⏳ AI system is loading research data. Please retry shortly.",
            [],
            []
        )

    domain_info = domain_classifier.classify(query)

    detected_domain = domain_info["domain"]

    kombucha_domains = [
        "brewing",
        "health",
        "contamination"
    ]

    if detected_domain in kombucha_domains:
        current_domain = detected_domain

    elif detected_domain == "general":

        if current_domain is None:
            return (
                "This assistant only supports kombucha-related queries.",
                [],
                []
            )

        detected_domain = current_domain

    risk_info = risk_detector.detect(query)

    risk_score = risk_info["risk_score"]

    control = control_policy.adapt(risk_score)

    retrieved_context, sources = retriever.retrieve(query, k=5)

    prompt = orchestrator.build_prompt(
        query,
        domain_info,
        risk_score,
        control,
        retrieved_context
    )

    response = get_llm().generate(
        prompt,
        control["temperature"]
    )

    similarity_score = hallucination_detector.detect(
        response,
        retrieved_context
    )

    validated = validator.validate(response)

    if risk_score > 0.7:

        if similarity_score < 0.4:
            validated = False

        if not validated:
            response = (
                "⚠️ Potential uncertainty detected."
            )

    logger.log(
        query=query,
        domain=detected_domain,
        risk=risk_score,
        response=response,
        validated=validated
    )

    save_conversation(query, response)

    suggestions = generate_followups(response)

    return response, suggestions, sources