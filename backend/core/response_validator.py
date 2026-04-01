class ResponseValidator:

    def validate(self, response: str):
        forbidden_phrases = ["cure", "guaranteed", "100% safe"]

        for phrase in forbidden_phrases:
            if phrase in response.lower():
                return False

        return True
