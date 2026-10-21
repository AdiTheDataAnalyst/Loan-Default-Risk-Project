"""Sigmoid (Platt) calibration: p = sigmoid(a * logit(score) + b).

Only two numbers (a, b) are learned, so it stays smooth even where few borrowers exist.
(An isotonic step function looked more flexible but overfit the extreme tails.)
The numbers are stored in JSON, so there is nothing fragile to unpickle.
"""
import json

import numpy as np

EPS = 1e-6


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


class SigmoidCalibrator:
    def __init__(self, a: float, b: float, score_p05: float = 0.0, score_p95: float = 1.0):
        if a <= 0:
            raise ValueError("Calibration slope must be positive so the ranking of borrowers is preserved.")
        self.a, self.b, self.score_p05, self.score_p95 = float(a), float(b), float(score_p05), float(score_p95)

    def predict(self, raw_scores):
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(raw_scores) + self.b)))

    def is_typical(self, raw_score: float) -> bool:
        """True when the score sits inside the range most validation borrowers fall in."""
        return self.score_p05 <= raw_score <= self.score_p95

    def to_json(self) -> str:
        return json.dumps({"a": self.a, "b": self.b, "score_p05": self.score_p05, "score_p95": self.score_p95}, indent=2)

    @classmethod
    def fit(cls, raw_scores, y_true):
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(C=1e6, max_iter=1000).fit(_logit(raw_scores).reshape(-1, 1), np.asarray(y_true))
        lo, hi = np.quantile(raw_scores, [0.05, 0.95])
        return cls(model.coef_[0, 0], model.intercept_[0], lo, hi)

    @classmethod
    def from_json(cls, path):
        d = json.loads(open(path, encoding="utf-8").read())
        return cls(d["a"], d["b"], d.get("score_p05", 0.0), d.get("score_p95", 1.0))
