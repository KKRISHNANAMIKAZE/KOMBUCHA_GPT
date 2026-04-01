class SafetyEngine:

    def apply(self, risk_level: str):
        if risk_level == "high":
            return "Provide a conservative answer. Add medical disclaimer. Do not claim cure."

        elif risk_level == "medium":
            return "Include cautionary advice and mention safety limits."

        else:
            return "Normal response."
