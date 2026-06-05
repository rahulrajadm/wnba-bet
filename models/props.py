"""
WNBA player prop prediction engine.
Predicts per-game stat values using season averages + recent 10-game form,
adjusted for opponent defensive rating and game pace.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from scipy.stats import poisson, norm
from utils.db import get_conn
from pipeline.team_metrics import get_opp_pts_allowed, get_game_pace_factor, get_def_rating_adj

RECENT_GAMES    = 10
RECENT_WEIGHT   = 0.55
SEASON_WEIGHT   = 0.45

LEAGUE_AVG_PACE = 95.0    # WNBA possessions per 40 min (approx)
LEAGUE_AVG_DRTG = 100.0   # league average defensive rating

# Maps platform stat names → internal stat column
STAT_MAP = {
    "Points":                  "pts",
    "Rebounds":                "reb",
    "Assists":                 "ast",
    "Steals":                  "stl",
    "Blocked Shots":           "blk",
    "3-PT Made":               "fg3m",
    "3-Pointers Made":         "fg3m",
    "Pts+Rebs+Asts":           "pra",
    "Pts+Rebs":                "pts_reb",
    "Pts+Asts":                "pts_ast",
    "Rebs+Asts":               "reb_ast",
    "Blks+Stls":               "blk_stl",
    "Fantasy Score":           "fantasy",
    "Points (Combo)":          "pts",
    "Rebounds (Combo)":        "reb",
    "Assists (Combo)":         "ast",
    "3-PT Made (Combo)":       "fg3m",
    "Turnovers":               "tov",
    "Free Throws Made":        "ftm",
}

# Stats where Poisson is appropriate (discrete counting stats)
POISSON_STATS  = {"pts", "reb", "ast", "stl", "blk", "fg3m", "tov", "ftm"}
# Stats where normal distribution fits better (combo stats, fantasy)
NORMAL_STATS   = {"pra", "pts_reb", "pts_ast", "reb_ast", "blk_stl", "fantasy"}


def load_player_logs() -> pd.DataFrame:
    conn = get_conn()
    df   = pd.read_sql("SELECT * FROM player_game_logs ORDER BY game_date ASC", conn)
    conn.close()
    if df.empty:
        return df
    # Derived combo stats
    df["pra"]      = df["pts"] + df["reb"] + df["ast"]
    df["pts_reb"]  = df["pts"] + df["reb"]
    df["pts_ast"]  = df["pts"] + df["ast"]
    df["reb_ast"]  = df["reb"] + df["ast"]
    df["blk_stl"]  = df["blk"] + df["stl"]
    df["fantasy"]  = df["pts"] + df["reb"] * 1.2 + df["ast"] * 1.5 + df["stl"] * 3 + df["blk"] * 3 - df["tov"]
    return df


def get_player_season_rate(player_name: str, stat_col: str, logs: pd.DataFrame) -> float | None:
    """Season average per game for a player."""
    player_logs = logs[logs["player_name"].str.lower() == player_name.lower()]
    if player_logs.empty:
        last = player_name.split()[-1].lower()
        player_logs = logs[logs["player_name"].str.lower().str.contains(last, na=False)]
    if player_logs.empty or stat_col not in player_logs.columns:
        return None
    vals = pd.to_numeric(player_logs[stat_col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) > 0 else None


def get_player_recent_rate(player_name: str, stat_col: str, logs: pd.DataFrame) -> float | None:
    """Last 10 games average for a player."""
    player_logs = logs[logs["player_name"].str.lower() == player_name.lower()]
    if player_logs.empty:
        last = player_name.split()[-1].lower()
        player_logs = logs[logs["player_name"].str.lower().str.contains(last, na=False)]
    if player_logs.empty or stat_col not in player_logs.columns:
        return None
    recent = player_logs.sort_values("game_date").tail(RECENT_GAMES)
    vals   = pd.to_numeric(recent[stat_col], errors="coerce").dropna()
    return float(vals.mean()) if len(vals) >= 3 else None




def prob_over_line(expected: float, line: float, stat_col: str) -> float:
    """P(stat > line) using appropriate distribution."""
    if stat_col in POISSON_STATS:
        if expected <= 0:
            return 0.0
        threshold = int(np.ceil(line))
        return float(1.0 - poisson.cdf(threshold - 1, mu=expected))
    else:
        # Normal distribution for combo/fantasy stats
        std = max(expected * 0.35, 2.0)
        return float(1 - norm.cdf(line + 0.5, loc=expected, scale=std))


def _build_team_opponent_map(games: list[dict]) -> dict[str, str]:
    """Map each team name → opponent team name for today's games."""
    opp_map = {}
    for g in games:
        opp_map[g["home_team"]] = g["away_team"]
        opp_map[g["away_team"]] = g["home_team"]
    return opp_map


def predict_props(
    lines_data: list[dict] | None = None,
    games: list[dict] | None = None,
    player_logs_df=None,
    team_logs_df=None,
) -> list[dict]:
    logs = player_logs_df if player_logs_df is not None else load_player_logs()
    if logs is None or (hasattr(logs, "empty") and logs.empty):
        return []

    if games is None:
        from pipeline.schedule import get_today_games
        games = get_today_games()
    opp_map = _build_team_opponent_map(games)

    if lines_data is not None:
        lines = pd.DataFrame(lines_data)
    else:
        conn  = get_conn()
        lines = pd.read_sql("SELECT * FROM prop_lines WHERE DATE(fetched_at) = DATE('now')", conn)
        conn.close()

    lines = lines.sort_values("fetched_at", ascending=False).drop_duplicates(
        subset=["platform", "player_name", "stat_type"]
    )

    predictions = []
    for _, row in lines.iterrows():
        stat_col = STAT_MAP.get(row["stat_type"])
        if stat_col is None or row["line"] is None:
            continue

        player_name = row["player_name"]
        line        = float(row["line"])

        season_rate = get_player_season_rate(player_name, stat_col, logs)
        if season_rate is None or season_rate < 0:
            continue

        recent_rate = get_player_recent_rate(player_name, stat_col, logs)

        if recent_rate is not None:
            blended = RECENT_WEIGHT * recent_rate + SEASON_WEIGHT * season_rate
            form    = "blended"
        else:
            blended = season_rate
            form    = "season_only"

        player_team = row.get("player_team", "")

        # Fix 2: Actual opponent defensive rating
        opp_team    = opp_map.get(player_team, "")
        opp_pts     = get_opp_pts_allowed(opp_team, game_logs_df=team_logs_df) if opp_team else LEAGUE_AVG_DRTG
        def_adj     = get_def_rating_adj(opp_pts)
        blended     = max(blended * def_adj, 0.0)

        # Fix 3: Pace adjustment — scale stats by expected game pace vs league avg
        if opp_team and player_team:
            pace_factor = get_game_pace_factor(player_team, opp_team, game_logs_df=team_logs_df)
        else:
            pace_factor = 1.0
        # Pace only affects counting stats (pts, reb, ast, combo stats)
        if stat_col in POISSON_STATS | NORMAL_STATS:
            blended = max(blended * pace_factor, 0.0)

        if blended <= 0:
            continue

        p_more = prob_over_line(blended, line, stat_col)
        p_less = 1.0 - p_more

        if p_more >= p_less:
            direction, model_prob, edge = "More", p_more, p_more - 0.50
        else:
            direction, model_prob, edge = "Less", p_less, p_less - 0.50

        if edge <= 0:
            continue

        predictions.append({
            "platform":      row["platform"],
            "player_name":   player_name,
            "player_team":   opp_team,
            "stat_type":     row["stat_type"],
            "line":          line,
            "direction":     direction,
            "model_prob":    round(model_prob, 4),
            "implied_prob":  0.50,
            "edge":          round(edge, 4),
            "expected_rate": round(blended, 3),
            "season_rate":   round(season_rate, 3),
            "recent_rate":   round(recent_rate, 3) if recent_rate is not None else None,
            "form_source":   form,
            "game_id":       row.get("game_id", ""),
        })

    return predictions


if __name__ == "__main__":
    preds = predict_props()
    preds.sort(key=lambda x: x["edge"], reverse=True)
    hi = [p for p in preds if 0.55 <= p["model_prob"] <= 0.85]
    print(f"Total prop predictions: {len(preds)}")
    print(f"High-interest (55-85% model prob): {len(hi)}")
    print(f"\n{'Player':<22} {'Stat':<20} {'Line':>5} {'Dir':>5} {'Model%':>7} {'Edge':>6}  {'Platform'}")
    print("-" * 85)
    for p in hi[:15]:
        print(f"{p['player_name']:<22} {p['stat_type']:<20} {p['line']:>5} "
              f"{p['direction']:>5} {p['model_prob']:>6.1%} {p['edge']:>+6.1%}  {p['platform']}")
