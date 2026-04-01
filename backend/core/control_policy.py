class ControlPolicy:

    def adapt(self, risk_score: float):

        if risk_score > 0.7:
            return {
                "temperature": 0.1,
                "retrieval_k": 5,
                "safety_strength": "strict"
            }

        elif risk_score > 0.4:
            return {
                "temperature": 0.2,
                "retrieval_k": 4,
                "safety_strength": "moderate"
            }

        else:
            return {
                "temperature": 0.3,
                "retrieval_k": 3,
                "safety_strength": "normal"
            }
