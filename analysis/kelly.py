"""Fractional Kelly criterion stake sizing (0.25x)."""

KELLY_FRACTION = 0.25


def kelly_stake(model_prob: float, payout_multiplier: float) -> float:
    b = payout_multiplier - 1
    if b <= 0 or model_prob <= 0:
        return 0.0
    full = (b * model_prob - (1 - model_prob)) / b
    return round(min(max(full * KELLY_FRACTION, 0.0), 0.25), 4)


def stake_dollars(kelly_pct: float, bankroll: float) -> float:
    return round(kelly_pct * bankroll, 2)


def risk_reward(stake: float, multiplier: float) -> dict:
    win = round(stake * multiplier - stake, 2)
    return {"stake": stake, "potential_win": win, "ratio": round(win / stake, 2) if stake > 0 else 0}
