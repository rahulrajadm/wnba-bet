"""
Trains WNBA game prediction models (moneyline, spread, totals) from historical data.
Features: team rolling averages, pace, rest days, home/away, H2H record.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor
from utils.db import get_conn

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../data/models")
os.makedirs(MODELS_DIR, exist_ok=True)

ROLLING_WINDOWS = [5, 10]   # last N games rolling averages


def load_game_logs() -> pd.DataFrame:
    conn = get_conn()
    df   = pd.read_sql("SELECT * FROM team_game_logs ORDER BY game_date ASC", conn)
    conn.close()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df["win"]       = (df["wl"] == "W").astype(int)
    return df


def build_team_rolling_features(logs: pd.DataFrame) -> pd.DataFrame:
    """Per team, compute rolling averages of key stats over last 5 and 10 games."""
    stat_cols = ["pts", "fg_pct", "fg3_pct", "reb", "ast", "stl", "blk", "tov", "plus_minus"]
    dfs = []

    for team_id, grp in logs.groupby("team_id"):
        grp = grp.sort_values("game_date").copy()
        for col in stat_cols:
            for w in ROLLING_WINDOWS:
                grp[f"{col}_last{w}"] = grp[col].shift(1).rolling(w, min_periods=1).mean()
        grp["win_pct_last10"] = grp["win"].shift(1).rolling(10, min_periods=1).mean()
        # Rest days since this team's previous game (any venue). Cap at 10 so
        # season boundaries don't produce 200-day outliers.
        grp["rest_days"] = grp["game_date"].diff().dt.days.clip(upper=10).fillna(3.0)
        grp["is_b2b"]   = (grp["rest_days"] <= 1).astype(int)
        dfs.append(grp)

    return pd.concat(dfs).sort_values("game_date")


def build_matchup_features(logs: pd.DataFrame) -> pd.DataFrame:
    """
    Join home and away team rolling features for each game.
    Each game appears once (home perspective).
    """
    # Identify home/away from matchup string (e.g. "LVA vs. SEA" = home, "LVA @ SEA" = away)
    logs["is_home"] = logs["matchup"].str.contains(" vs\.", regex=True).astype(int)

    home = logs[logs["is_home"] == 1].copy()
    away = logs[logs["is_home"] == 0].copy()

    # rest_days/is_b2b are computed per-team in build_team_rolling_features so they
    # measure days since the team's previous game at ANY venue — matching what
    # get_rest_days() computes at predict time. (A groupby on the merged matchups
    # would only see each team's previous HOME game.)
    rolling_cols = [c for c in logs.columns if "_last" in c or c == "win_pct_last10"]
    rolling_cols += ["rest_days", "is_b2b"]

    home_feat = home[["game_id", "game_date", "season", "team_name", "pts", "win"] + rolling_cols].copy()
    home_feat.columns = ["game_id", "game_date", "season", "home_team", "home_pts", "home_win"] + \
                        [f"home_{c}" for c in rolling_cols]

    away_feat = away[["game_id", "team_name", "pts"] + rolling_cols].copy()
    away_feat.columns = ["game_id", "away_team", "away_pts"] + [f"away_{c}" for c in rolling_cols]

    merged = home_feat.merge(away_feat, on="game_id", how="inner")

    merged["total_pts"]  = merged["home_pts"] + merged["away_pts"]
    merged["pt_diff"]    = merged["home_pts"] - merged["away_pts"]
    merged["home_cover"] = (merged["pt_diff"] > 0).astype(int)

    merged = merged.sort_values("game_date")
    merged["rest_advantage"] = merged["home_rest_days"] - merged["away_rest_days"]

    # Explicit home court indicator — gives model a direct signal
    merged["home_court"] = 1

    return merged


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"game_id", "game_date", "season", "home_team", "away_team",
               "home_pts", "away_pts", "home_win", "total_pts", "pt_diff", "home_cover",
               "home_game_date", "away_game_date"}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [float, int, "float64", "int64"]]


def train_models(matchups: pd.DataFrame):
    matchups = matchups.dropna(subset=["home_win", "total_pts", "pt_diff"])

    feat_cols = get_feature_cols(matchups)

    # Chronological split: train on the past, evaluate on the most recent 20%.
    # A random split lets games from the same week land on both sides, which
    # inflates holdout metrics and hides drift.
    matchups = matchups.sort_values("game_date")
    cut      = int(len(matchups) * 0.8)
    train_df, test_df = matchups.iloc[:cut], matchups.iloc[cut:]
    print(f"  Train: {len(train_df)} games through {train_df['game_date'].max().date()}")
    print(f"  Test:  {len(test_df)} games from {test_df['game_date'].min().date()}")

    fill_means = train_df[feat_cols].mean()
    X_tr = train_df[feat_cols].fillna(fill_means)
    X_te = test_df[feat_cols].fillna(fill_means)

    results = {}

    # 1. Moneyline (win/loss classifier)
    ml_model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                              eval_metric="logloss", random_state=42)
    ml_model.fit(X_tr, train_df["home_win"])
    acc = accuracy_score(test_df["home_win"], ml_model.predict(X_te))
    print(f"  Moneyline accuracy (holdout): {acc:.3f}")
    results["moneyline"] = ml_model

    # 2. Point differential regressor (for spread)
    diff_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    diff_model.fit(X_tr, train_df["pt_diff"])
    diff_pred = diff_model.predict(X_te)
    mae = mean_absolute_error(test_df["pt_diff"], diff_pred)
    spread_std = float(np.std(test_df["pt_diff"] - diff_pred))
    print(f"  Point diff MAE (holdout): {mae:.2f} pts | residual std: {spread_std:.2f}")
    results["spread"] = diff_model

    # 3. Total points regressor
    total_model = XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    total_model.fit(X_tr, train_df["total_pts"])
    total_pred = total_model.predict(X_te)
    mae = mean_absolute_error(test_df["total_pts"], total_pred)
    totals_std = float(np.std(test_df["total_pts"] - total_pred))
    print(f"  Totals MAE (holdout): {mae:.2f} pts | residual std: {totals_std:.2f}")
    results["totals"] = total_model

    # Residual stds drive the cdf-based bet probabilities in models/game.py —
    # calibrated values keep win/cover/over probs honest.
    calibration = {"spread_std": round(spread_std, 2), "totals_std": round(totals_std, 2)}

    # Refit on ALL data so the deployed models see the newest games.
    X_all = matchups[feat_cols].fillna(fill_means)
    results["moneyline"].fit(X_all, matchups["home_win"])
    results["spread"].fit(X_all, matchups["pt_diff"])
    results["totals"].fit(X_all, matchups["total_pts"])

    return results, feat_cols, calibration


def main():
    print("Loading game logs...")
    logs = load_game_logs()
    print(f"  {len(logs)} team-game rows across {logs['season'].nunique()} seasons")

    print("Building rolling features...")
    logs_with_rolling = build_team_rolling_features(logs)

    print("Building matchup features (with rest days + home court)...")
    matchups = build_matchup_features(logs_with_rolling)
    print(f"  {len(matchups)} matchups")

    print("Training models...")
    models, feat_cols, calibration = train_models(matchups)

    print("Saving models...")
    for name, model in models.items():
        path = os.path.join(MODELS_DIR, f"game_{name}.pkl")
        joblib.dump(model, path)
        print(f"  Saved {path}")

    joblib.dump(feat_cols, os.path.join(MODELS_DIR, "game_feature_cols.pkl"))
    joblib.dump(calibration, os.path.join(MODELS_DIR, "game_calibration.pkl"))
    print(f"  Saved calibration: {calibration}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
