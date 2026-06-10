"""EV calculations for traditional odds (game picks) and pick'em props."""

PRIZEPICKS_POWER = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0}
UNDERDOG_POWER   = {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0}

PLATFORM_MULTIPLIERS = {
    "prizepicks":       PRIZEPICKS_POWER,
    "underdog":         UNDERDOG_POWER,
    "draftkings_pick6": {2: 3.0, 3: 5.5, 4: 11.0, 5: 22.0},
    "chalkboard":       {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0, 6: 25.0},
    "sleeper":          {2: 3.0, 3: 5.0, 4: 10.0, 5: 20.0},
}


def breakeven_prob(platform: str, slip_size: int = 2) -> float:
    """Per-leg probability needed to break even on a pick'em slip.

    A 2-pick 3x slip breaks even at p² · 3 = 1 ⇒ p ≈ 0.577, NOT 0.50.
    Measuring prop edges against 0.50 systematically overstates value.
    """
    mult = PLATFORM_MULTIPLIERS.get(str(platform).lower(), PRIZEPICKS_POWER).get(slip_size, 3.0)
    return float((1.0 / mult) ** (1.0 / slip_size))


def american_to_implied(odds: float) -> float:
    """Convert American odds to implied probability (includes vig)."""
    if odds > 0:
        return 100 / (odds + 100)
    return abs(odds) / (abs(odds) + 100)


def remove_vig(home_odds: float, away_odds: float) -> tuple[float, float]:
    """Remove bookmaker vig to get true implied probabilities."""
    h = american_to_implied(home_odds)
    a = american_to_implied(away_odds)
    total = h + a
    return h / total, a / total


def ev_game(model_prob: float, american_odds: float) -> float:
    """EV per $100 on a traditional sportsbook game bet."""
    net_win = american_odds if american_odds > 0 else 100 / abs(american_odds) * 100
    return round(model_prob * net_win - (1 - model_prob) * 100, 2)


def ev_prop(model_prob: float, implied_prob: float = 0.50) -> float:
    """EV per $100 on a pick'em prop leg."""
    return round((model_prob - implied_prob) * 100, 2)


def ev_slip(model_probs: list[float], platform: str, slip_size: int) -> dict:
    """EV for a full multi-leg pick'em slip."""
    multipliers = PLATFORM_MULTIPLIERS.get(platform, PRIZEPICKS_POWER)
    multiplier  = multipliers.get(slip_size)
    if not multiplier:
        return {}
    p_all = 1.0
    for p in model_probs[:slip_size]:
        p_all *= p
    return {
        "slip_size":    slip_size,
        "p_all_hit":    round(p_all, 4),
        "multiplier":   multiplier,
        "ev_per_100":   round(p_all * multiplier * 100 - 100, 2),
    }
