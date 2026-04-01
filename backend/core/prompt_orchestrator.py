class PromptOrchestrator:

    def build_prompt(self, query, domain, risk_score, control, context):

        if control["safety_strength"] == "strict":
            safety_instruction = (
                "Provide conservative advice. "
                "Do not claim medical effectiveness. "
                "Recommend consulting a professional."
            )

        elif control["safety_strength"] == "moderate":
            safety_instruction = (
                "Include caution and note limitations of evidence."
            )

        else:
            safety_instruction = "Provide clear explanation."

        prompt = f"""
You are K-GPT, a specialized kombucha domain expert AI.

You must answer ONLY kombucha-related questions.

If Past Conversation Memory is relevant, PRIORITIZE it for personalization.
If Knowledge Context is relevant, use it for evidence grounding.
Do NOT ignore relevant memory information.

Risk Score: {risk_score}
Safety Mode: {control['safety_strength']}

Safety Instruction:
{safety_instruction}

----------------------------------------
Knowledge & Memory Context:
{context}
----------------------------------------

User Question:
{query}

Provide an accurate, evidence-aware, domain-specific answer.
If previous conversation details are relevant, explicitly reference them.
"""

        return prompt
