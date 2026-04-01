import csv
import os
from datetime import datetime


class Logger:

    def __init__(self, filename="results.csv"):
        self.file = f"evaluation/{filename}"

        if not os.path.exists("evaluation"):
            os.makedirs("evaluation")

        if not os.path.exists(self.file):
            with open(self.file, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp",
                    "query",
                    "domain",
                    "risk_level",
                    "response",
                    "validated"
                ])


    def log(self, query, domain, risk, response, validated):
        with open(self.file, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(),
                query,
                domain,
                risk,
                response,
                validated
            ])
