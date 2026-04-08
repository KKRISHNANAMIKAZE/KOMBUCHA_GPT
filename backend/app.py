# ================= IMPORTS =================
from core.domain_classifier import DomainClassifier
from core.risk_detector import RiskDetector
from core.control_policy import ControlPolicy
from core.prompt_orchestrator import PromptOrchestrator
from core.response_validator import ResponseValidator
from core.hallucination_detector import HallucinationDetector
from models.llm_interface import LLMInterface
from evaluation.logger import Logger

from rag.retriever import Retriever
from data_loader import download_files

import json
import os
from datetime import datetime


# ================= GLOBAL (CHANGED TO LAZY) =================
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

current_domain = None


# ================= 🔥 LAZY CORE INIT =================
def initialize_core():
    global domain_classifier, risk_detector, control_policy
    global llm, validator, hallucination_detector, orchestrator, logger
    global system_loaded

    if system_loaded:
        return

    print("⚡ Initializing core system...")

    domain_classifier = DomainClassifier()
    risk_detector = RiskDetector()
    control_policy = ControlPolicy()
    llm = LLMInterface()
    validator = ResponseValidator()
    hallucination_detector = HallucinationDetector()
    orchestrator = PromptOrchestrator()
    logger = Logger("framework_results.csv")

    system_loaded = True
    print("✅ Core system ready")


# ================= 🔥 LAZY RAG INIT =================
def initialize_rag():
    global retriever, data_loaded

    if data_loaded:
        return

    print("🚀 Loading FAISS + Retriever...")

    if not os.path.exists("data/kombucha_index.faiss"):
        download_files()

    retriever = Retriever()

    data_loaded = True
    print("✅ Retriever ready")


# ================= MEMORY SAVER =================
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


# ================= FEEDBACK SAVER =================
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


# ================= FOLLOW-UP GENERATOR =================
def generate_followups(response):

    initialize_core()

    follow_prompt = f"""
Based on this kombucha explanation:

{response}

Suggest exactly 3 short relevant follow-up questions.
Return ONLY the questions as a numbered list.
"""

    follow_text = llm.generate(follow_prompt, temperature=0.7)

    suggestions = []

    for line in follow_text.split("\n"):
        line = line.strip()
        if line and line[0].isdigit():
            cleaned = line.split(".", 1)[-1].strip()
            suggestions.append(cleaned)

    return suggestions[:3]


# ================= FILE ANALYZER =================
def analyze_uploaded_file(file_text, user_query=None):

    initialize_core()

    prompt = f"""
You are K-GPT, a kombucha-only research assistant.

You ONLY answer kombucha-related questions.

If the document is NOT related to kombucha, respond:
"I am K-GPT and only answer kombucha-related queries."

Document:
{file_text}

User Question:
{user_query}
"""

    return llm.generate(prompt, temperature=0.3)


# ================= CORE QUERY PIPELINE =================
def process_query(query):
    global current_domain

    initialize_core()
    initialize_rag()

    domain_info = domain_classifier.classify(query)
    detected_domain = domain_info["domain"]

    kombucha_domains = ["brewing", "health", "contamination"]

    if detected_domain in kombucha_domains:
        current_domain = detected_domain

    elif detected_domain == "general":
        if current_domain is None:
            return (
                "This assistant is specialized for kombucha-related queries only.",
                [],
                []
            )
        detected_domain = current_domain

    if current_domain is None:
        return (
            "This assistant is specialized for kombucha-related queries only.",
            [],
            []
        )

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

    response = llm.generate(prompt, control["temperature"])

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
                "⚠️ The system detected potential uncertainty or unsafe claims. "
                "Please consult a qualified expert."
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