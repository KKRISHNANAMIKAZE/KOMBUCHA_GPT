class DomainClassifier:

    def classify(self, query: str):
        query = query.lower()

        brewing_keywords = ["kombucha", "ferment", "brew", "scoby", "tea"]
        health_keywords = ["health", "diabetes", "pregnant", "benefit"]
        contamination_keywords = ["mold", "contamination", "spoiled", "ph"]

        if any(word in query for word in brewing_keywords):
            return {"domain": "brewing", "confidence": 0.8}

        elif any(word in query for word in health_keywords):
            return {"domain": "health", "confidence": 0.85}

        elif any(word in query for word in contamination_keywords):
            return {"domain": "contamination", "confidence": 0.9}

        else:
            return {"domain": "general", "confidence": 0.6}
