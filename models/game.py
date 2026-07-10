"""
WNBA game prediction engine.
Loads trained models and generates moneyline, spread, and totals predictions
for today's matchups using rolling team stats.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import functools
import pandas as pd
import numpy as np
import joblib
from scipy.stats import norm
from utils.db import get_conn
from pipeline.team_metrics import get_rest_days

MODELS_DIR     = os.path.join(os.path.dirname(__file__), "../data/models")
LEAGUE_AVG_PTS = 82.0   # WNBA league average points per team per game

ROLLING_WINDOWS = [5, 10]


@functools.lru_cache(maxsize=1)
def load_models():
    ml    = joblib.load(os.path.join(MODELS_DIR, "game_moneyline.pkl"))
    sprd  = joblib.load(os.path.join(MODELS_DIR, "game_spread.pkl"))
    total = joblib.load(os.path.join(MODELS_DIR, "game_totals.pkl"))
    feats = joblib.load(os.path.join(MODELS_DIR, "game_feature_cols.pkl"))
    try:
        calib = joblib.load(os.path.join(MODELS_DIR, "game_calibration.pkl"))
    except Exception:
        calib = {"spread_std": 12.0, "totals_std": 14.0}
    # Single-threaded predict: XGBoost's OpenMP pool spawns a thread per visible
    # CPU on every predict; on Streamlit Cloud's small container, predicts fired
    # from short-lived Streamlit script threads segfault the whole process.
    # Predictions here are a handful of single rows — threading buys nothing.
    for m in (ml, sprd, total):
        try:
            m.set_params(n_jobs=1)
        except Exception:
            pass
    return ml, sprd, total, feats, calib


def get_team_rolling_stats(team_name: str, n: int = 10, game_logs_df=None) -> dict:
    """Pull a team's last N game rolling averages from the DB or an in-memory DataFrame."""
    if game_logs_df is not None:
        try:
            last = team_name.strip().split()[-1].lower()
            df = game_logs_df[game_logs_df["team_name"].str.lower().str.contains(last, na=False)]
            df = df.sort_values("game_date", ascending=False).head(n)
        except (KeyError, AttributeError, TypeError):
            return {}
    else:
        conn = get_conn()
        df   = pd.read_sql(
            "SELECT * FROM team_game_logs WHERE team_name LIKE ? ORDER BY game_date DESC LIMIT ?",
            conn, params=(f"%{team_name.split()[-1]}%", n)
        )
        conn.close()

    if df.empty:
        return {}

    return {
        "pts":          df["pts"].mean(),
        "fg_pct":       df["fg_pct"].mean(),
        "fg3_pct":      df["fg3_pct"].mean(),
        "reb":          df["reb"].mean(),
        "ast":          df["ast"].mean(),
        "stl":          df["stl"].mean(),
        "blk":          df["blk"].mean(),
        "tov":          df["tov"].mean(),
        "plus_minus":   df["plus_minus"].mean(),
        "win_pct":      (df["wl"] == "W").mean(),
        "games_sample": len(df),
    }


def build_matchup_vector(home_team: str, away_team: str, feat_cols: list[str], game_logs_df=None) -> pd.DataFrame:
    """Build a single-row feature vector for a matchup."""
    home_s10 = get_team_rolling_stats(home_team, n=10, game_logs_df=game_logs_df)
    away_s10 = get_team_rolling_stats(away_team, n=10, game_logs_df=game_logs_df)

    if not home_s10 or not away_s10:
        return None

    home_s5 = get_team_rolling_stats(home_team, n=5, game_logs_df=game_logs_df) or home_s10
    away_s5 = get_team_rolling_stats(away_team, n=5, game_logs_df=game_logs_df) or away_s10
    _ws = {5: (home_s5, away_s5), 10: (home_s10, away_s10)}

    row = {}
    stat_cols = ["pts", "fg_pct", "fg3_pct", "reb", "ast", "stl", "blk", "tov", "plus_minus"]
    for col in stat_cols:
        for w in ROLLING_WINDOWS:
            h, a = _ws[w]
            row[f"home_{col}_last{w}"] = h.get(col, 0)
            row[f"away_{col}_last{w}"] = a.get(col, 0)
    row["home_win_pct_last10"] = home_s10.get("win_pct", 0.5)
    row["away_win_pct_last10"] = away_s10.get("win_pct", 0.5)
    # Cap at 10 to match the training-time clip (season gaps would otherwise blow this up)
    row["home_rest_days"]      = min(get_rest_days(home_team, game_logs_df=game_logs_df), 10.0)
    row["away_rest_days"]      = min(get_rest_days(away_team, game_logs_df=game_logs_df), 10.0)
    row["home_is_b2b"]         = int(row["home_rest_days"] <= 1)
    row["away_is_b2b"]         = int(row["away_rest_days"] <= 1)
    row["rest_advantage"]      = row["home_rest_days"] - row["away_rest_days"]
    row["home_court"]          = 1

    df = pd.DataFrame([row])
    for col in feat_cols:
        if col not in df.columns:
            df[col] = 0
    return df[feat_cols]


def predict_game(home_team: str, away_team: str, game_logs_df=None) -> dict | None:
    """Generate ML, spread, and totals predictions for a matchup."""
    try:
        ml_model, sprd_model, total_model, feat_cols, calib = load_models()
    except Exception:
        return None

    X = build_matchup_vector(home_team, away_team, feat_cols, game_logs_df=game_logs_df)
    if X is None:
        return None

    # Spread model is the single source of truth for all probabilities.
    # Deriving win probability from pred_diff guarantees P(win by N+) ≤ P(win).
    # Using a separate ML classifier for win prob caused the two numbers to
    # contradict each other (e.g., 83% cover -2.5 but only 72% win outright).
    # Stds come from holdout residuals saved at training time (game_calibration.pkl).
    SPREAD_STD = float(calib.get("spread_std", 12.0))
    TOTALS_STD = float(calib.get("totals_std", 14.0))
    pred_diff   = float(sprd_model.predict(X)[0])
    home_win_prob = float(norm.cdf(pred_diff / SPREAD_STD))
    away_win_prob = 1.0 - home_win_prob

    pred_total = float(total_model.predict(X)[0])

    return {
        "home_team":       home_team,
        "away_team":       away_team,
        "home_win_prob":   round(home_win_prob, 4),
        "away_win_prob":   round(away_win_prob, 4),
        "pred_diff":       round(pred_diff, 1),
        "pred_total":      round(pred_total, 1),
        "spread_std":      SPREAD_STD,
        "totals_std":      TOTALS_STD,
    }


def prob_cover_spread(pred_diff: float, spread_line: float, std: float = 12.0) -> float:
    """P(home team covers spread_line) given predicted point differential.

    Home covers if actual_diff > -spread_line (e.g. home -12.5 needs to win by 12.5+;
    home +6.5 covers as long as it doesn't lose by more than 6.5).
    P(actual_diff > -spread_line) = norm.cdf((pred_diff + spread_line) / std).
    """
    return float(norm.cdf(pred_diff + spread_line, 0, std))


def prob_over_total(pred_total: float, total_line: float, std: float = 14.0) -> float:
    """P(total goes over total_line) given predicted total."""
    return float(1 - norm.cdf(total_line, pred_total, std))


if __name__ == "__main__":
    from pipeline.schedule import get_today_games
    games = get_today_games()
    print(f"\nGame predictions for {len(games)} games:\n")
    for g in games:
        result = predict_game(g["home_team"], g["away_team"])
        if result:
            print(f"{g['away_team']} @ {g['home_team']}")
            print(f"  ML:    Home {result['home_win_prob']:.1%} | Away {result['away_win_prob']:.1%}")
            print(f"  Diff:  Home predicted to win by {result['pred_diff']:+.1f}")
            print(f"  Total: Predicted {result['pred_total']:.1f} pts")
        else:
            print(f"{g['away_team']} @ {g['home_team']} — insufficient data")
