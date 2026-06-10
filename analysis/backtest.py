"""
Walk-forward backtest for the game models.

Trains on all seasons BEFORE the test season, then predicts the test season's
games in order (rolling features are already point-in-time via shift(1), so
there is no lookahead). Reports accuracy, MAE, Brier score, and calibration —
this is the number to trust when judging a model change, not 2 days of picks.

Usage:
    python analysis/backtest.py            # test on the latest season in the DB
    python analysis/backtest.py 2024       # test on a specific season
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
from scipy.stats import norm
from xgboost import XGBRegressor
from models.train import load_game_logs, build_team_rolling_features, \
    build_matchup_features, get_feature_cols


def run_backtest(test_season: int | None = None) -> dict | None:
    logs = load_game_logs()
    logs["season_int"] = pd.to_numeric(logs["season"], errors="coerce")

    matchups = build_matchup_features(build_team_rolling_features(logs))
    matchups["season_int"] = pd.to_numeric(matchups["season"], errors="coerce")
    matchups = matchups.dropna(subset=["pt_diff", "total_pts", "season_int"])

    seasons = sorted(matchups["season_int"].unique())
    if test_season is None:
        test_season = int(seasons[-1])
    if test_season == seasons[0]:
        print(f"No seasons before {test_season} to train on.")
        return None

    train_df = matchups[matchups["season_int"] < test_season].sort_values("game_date")
    test_df  = matchups[matchups["season_int"] == test_season].sort_values("game_date")
    if test_df.empty:
        print(f"No games found for season {test_season}.")
        return None

    feat_cols  = get_feature_cols(matchups)
    feat_cols  = [c for c in feat_cols if c != "season_int"]
    fill_means = train_df[feat_cols].mean()
    X_tr = train_df[feat_cols].fillna(fill_means)
    X_te = test_df[feat_cols].fillna(fill_means)

    params = dict(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    diff_model  = XGBRegressor(**params).fit(X_tr, train_df["pt_diff"])
    total_model = XGBRegressor(**params).fit(X_tr, train_df["total_pts"])

    # Std from train residuals (in-sample is too tight; use a 20% tail holdout)
    tail = max(int(len(train_df) * 0.2), 50)
    tail_pred  = diff_model.predict(X_tr.tail(tail))
    spread_std = max(float(np.std(train_df["pt_diff"].tail(tail) - tail_pred)), 8.0)

    pred_diff  = diff_model.predict(X_te)
    pred_total = total_model.predict(X_te)
    actual_diff  = test_df["pt_diff"].values
    actual_total = test_df["total_pts"].values
    home_won     = (actual_diff > 0).astype(int)
    win_prob     = norm.cdf(pred_diff / spread_std)

    acc        = float(((pred_diff > 0).astype(int) == home_won).mean())
    brier      = float(np.mean((win_prob - home_won) ** 2))
    diff_mae   = float(np.mean(np.abs(pred_diff - actual_diff)))
    total_mae  = float(np.mean(np.abs(pred_total - actual_total)))

    print(f"\nBacktest — train: {seasons[0]:.0f}–{test_season - 1}  |  "
          f"test: {test_season} ({len(test_df)} games)")
    print("-" * 60)
    print(f"  Moneyline accuracy:  {acc:.3f}   (home team wins ~58% as baseline)")
    print(f"  Brier score:         {brier:.4f}  (0.25 = coin flip, lower is better)")
    print(f"  Point diff MAE:      {diff_mae:.2f} pts")
    print(f"  Totals MAE:          {total_mae:.2f} pts")

    buckets = pd.DataFrame({"prob": win_prob, "won": home_won})
    buckets["bucket"] = (buckets["prob"] * 10).astype(int).clip(0, 9) / 10
    calib = buckets.groupby("bucket").agg(games=("won", "size"), actual=("won", "mean"))
    print("\n  Calibration (predicted home-win prob vs actual):")
    for b, row in calib.iterrows():
        print(f"    {b:.0%}–{b + 0.1:.0%}: predicted ~{b + 0.05:.0%}, "
              f"actual {row['actual']:.1%}  ({row['games']:.0f} games)")

    return {"accuracy": acc, "brier": brier, "diff_mae": diff_mae,
            "total_mae": total_mae, "spread_std": spread_std}


if __name__ == "__main__":
    season = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_backtest(season)
