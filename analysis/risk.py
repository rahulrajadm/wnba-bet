"""Risk profile assignment for WNBA picks."""

HIGH_VARIANCE_STATS  = {"Steals", "Blocked Shots", "Blks+Stls", "3-PT Made", "3-Pointers Made", "3-PT Made (Combo)"}
MEDIUM_VARIANCE_STATS = {"Rebounds", "Assists", "Pts+Rebs", "Pts+Asts", "Rebs+Asts", "Turnovers", "Free Throws Made"}
LOW_VARIANCE_STATS   = {"Points", "Pts+Rebs+Asts", "Fantasy Score", "Points (Combo)"}

# Game pick risk profiles
GAME_RISK = {
    "moneyline": "MEDIUM",
    "spread":    "MEDIUM",
    "totals":    "LOW",
}


def get_risk_profile(stat_type: str, edge: float = 0.0) -> str:
    if stat_type in HIGH_VARIANCE_STATS:
        return "HIGH"
    if stat_type in MEDIUM_VARIANCE_STATS:
        return "MEDIUM"
    if stat_type in LOW_VARIANCE_STATS:
        return "LOW" if edge >= 0.10 else "MEDIUM"
    if stat_type in GAME_RISK:
        return GAME_RISK[stat_type]
    return "MEDIUM"


RISK_COLORS = {
    "LOW":    "#16a34a",
    "MEDIUM": "#c2410c",
    "HIGH":   "#dc2626",
}
