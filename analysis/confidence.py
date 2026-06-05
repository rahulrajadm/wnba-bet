"""Confidence tier assignment based on edge size."""


def get_confidence_tier(edge: float, sample: int = 100) -> str:
    penalty = 0.03 if sample < 20 else 0.0
    adj = edge - penalty
    if adj >= 0.15:   return "STRONG"
    elif adj >= 0.10: return "HIGH"
    elif adj >= 0.05: return "MEDIUM"
    else:             return "LOW"


TIER_COLORS = {
    "STRONG": "#16a34a",
    "HIGH":   "#22c55e",
    "MEDIUM": "#ca8a04",
    "LOW":    "#374151",
}

TIER_RANK = {"STRONG": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1}
