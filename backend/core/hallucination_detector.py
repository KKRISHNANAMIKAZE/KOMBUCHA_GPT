from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class HallucinationDetector:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def detect(self, response: str, context: str):

        emb_response = self.model.encode([response])
        emb_context = self.model.encode([context])

        similarity = cosine_similarity(emb_response, emb_context)[0][0]

        return similarity
