"""Metrics and threshold selection."""
import numpy as np
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix, f1_score,
    fbeta_score, precision_recall_curve, precision_score, recall_score,
    roc_auc_score, roc_curve,
)

from src.config import F_BETA


def best_threshold(y_true, proba, beta=F_BETA) -> float:
    """Pick the probability cut-off that gives the best F-beta score on VALIDATION data.
    beta=2 means we care about catching defaults (recall) more than avoiding false alarms."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    precision, recall = precision[:-1], recall[:-1]          # align with thresholds
    denom = (beta**2 * precision) + recall
    fbeta = np.divide((1 + beta**2) * precision * recall, denom,
                      out=np.zeros_like(denom), where=denom > 0)
    return float(thresholds[int(np.argmax(fbeta))])


def compute_metrics(y_true, proba, threshold) -> dict:
    """All headline metrics at a given threshold."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": round(float(threshold), 4),
        "precision": round(precision_score(y_true, pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "f2": round(fbeta_score(y_true, pred, beta=F_BETA), 4),
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_true, pred, target_names=["No default", "Default"],
            output_dict=True, zero_division=0),
    }


def curve_points(y_true, proba, max_points=200) -> dict:
    """ROC and precision-recall curve points, thinned out so the app loads quickly."""
    def thin(a):
        idx = np.linspace(0, len(a) - 1, min(max_points, len(a))).astype(int)
        return [round(float(v), 5) for v in np.asarray(a)[idx]]
    fpr, tpr, _ = roc_curve(y_true, proba)
    prec, rec, _ = precision_recall_curve(y_true, proba)
    return {"fpr": thin(fpr), "tpr": thin(tpr), "precision": thin(prec), "recall": thin(rec)}
