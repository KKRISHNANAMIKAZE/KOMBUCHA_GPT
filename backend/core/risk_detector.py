class RiskDetector:

    def detect(self, query: str):

        query = query.lower()
        score = 0.0

        high_risk_keywords = [
            "cure", "treat", "replace medicine",
            "antibiotic", "pregnant", "baby",
            "infection", "disease"
        ]

        medium_risk_keywords = [
            "pH", "alcohol", "safe", "mold"
        ]

        for word in high_risk_keywords:
            if word in query:
                score += 0.25

        for word in medium_risk_keywords:
            if word in query:
                score += 0.15

        score = min(score, 1.0)

        return {
            "risk_score": score
        }
