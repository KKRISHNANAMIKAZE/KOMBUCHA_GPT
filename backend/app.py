# ================= IMPORTS =================
from core.domain_classifier import DomainClassifier
from core.risk_detector import RiskDetector
from core.control_policy import ControlPolicy
from core.prompt_orchestrator import PromptOrchestrator
from core.response_validator import ResponseValidator
from core.hallucination_detector import HallucinationDetector
from models.llm_interface import LLMInterface
from evaluation.logger import Logger

import json
import os
from datetime import datetime


# ================= INITIALIZATION =================
domain_classifier = DomainClassifier()
risk_detector = RiskDetector()
control_policy = ControlPolicy()
llm = LLMInterface()
validator = ResponseValidator()
hallucination_detector = HallucinationDetector()
orchestrator = PromptOrchestrator()
logger = Logger("framework_results.csv")

current_domain = None


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

    # 1️⃣ Domain Classification
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

    # 2️⃣ Risk Scoring
    risk_info = risk_detector.detect(query)
    risk_score = risk_info["risk_score"]

    # 3️⃣ Adaptive Control Policy
    control = control_policy.adapt(risk_score)

    # ================= SAFE RETRIEVAL =================
    retrieved_context = """
Kombucha is a fermented tea made using a symbiotic culture of bacteria and yeast (SCOBY).
It is known for probiotics, antioxidants, and potential health benefits such as improved digestion.
"""
    sources = [{"citation": "Kombucha Research Literature"}]

    # 5️⃣ Prompt Construction
    prompt = orchestrator.build_prompt(
        query,
        domain_info,
        risk_score,
        control,
        retrieved_context
    )

    # 6️⃣ LLM Generation
    response = llm.generate(prompt, control["temperature"])

    # 7️⃣ Hallucination Detection
    similarity_score = hallucination_detector.detect(
        response,
        retrieved_context
    )

    validated = validator.validate(response)

    # 🔥 Safety control
    if risk_score > 0.7:
        if similarity_score < 0.4:
            validated = False

        if not validated:
            response = (
                "⚠️ The system detected potential uncertainty or unsafe claims. "
                "Please consult a qualified expert."
            )

    # 8️⃣ Logging
    logger.log(
        query=query,
        domain=detected_domain,
        risk=risk_score,
        response=response,
        validated=validated
    )

    # 9️⃣ Save Conversation Memory
    save_conversation(query, response)

    # 🔟 Generate Follow-ups
    suggestions = generate_followups(response)

    return response, suggestions, sources