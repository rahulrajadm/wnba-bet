"""
Computes team metrics directly from team_game_logs.
Used as a fallback when the NBA Stats API advanced endpoint is unavailable.

- Defensive rating proxy: avg points opponent scored vs this team (last N games)
- Pace factor: team pts per game / league average (proxy for pace)
- Rest days: days since team's last game
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
import numpy as np
from datetime import date
from utils.db import get_conn

LEAGUE_AVG_TEAM_PTS = 82.0   # WNBA league avg pts per team per game
RECENT_N            = 20     # games to use for rolling metrics


def _find_team(team_name: str, df: pd.DataFrame, col: str = "team_name") -> pd.DataFrame:
    """Fuzzy team name match — tries exact then last-word fallback."""
    match = df[df[col].str.lower() == team_name.lower()]
    if match.empty:
        last = team_name.strip().split()[-1].lower()
        match = df[df[col].str.lower().str.contains(last, na=False)]
    return match


def get_rest_days(team_name: str, game_logs_df: pd.DataFrame | None = None) -> float:
    """Days since this team last played. Defaults to 2.5 if unknown."""
    if game_logs_df is not None:
        team_rows = _find_team(team_name, game_logs_df)
        if team_rows.empty:
            return 2.5
        last_game = team_rows["game_date"].max()
    else:
        conn = get_conn()
        df   = pd.read_sql(
            "SELECT MAX(game_date) as last_game FROM team_game_logs WHERE team_name LIKE ?",
            conn, params=(f"%{team_name.strip().split()[-1]}%",)
        )
        conn.close()
        if df.empty or pd.isna(df["last_game"].iloc[0]):
            return 2.5
        last_game = df["last_game"].iloc[0]
    try:
        last = pd.to_datetime(last_game).date()
        return float((date.today() - last).days)
    except Exception:
        return 2.5


def get_opp_pts_allowed(team_name: str, n: int = RECENT_N, game_logs_df: pd.DataFrame | None = None) -> float:
    """
    Average points this team ALLOWED per game over last N games.
    Computed by joining game_logs: find the opponent's pts in each of this team's games.
    """
    if game_logs_df is not None:
        team_rows = _find_team(team_name, game_logs_df).sort_values("game_date", ascending=False).head(n)
        if team_rows.empty:
            return LEAGUE_AVG_TEAM_PTS
        game_ids = team_rows["game_id"].tolist()
        team_ids = team_rows["team_id"].astype(str).tolist()
        opp_rows = game_logs_df[
            game_logs_df["game_id"].isin(game_ids) &
            (~game_logs_df["team_id"].astype(str).isin(team_ids))
        ]
        return round(float(opp_rows["pts"].mean()), 2) if not opp_rows.empty else LEAGUE_AVG_TEAM_PTS
    conn = get_conn()
    df   = pd.read_sql(f"""
        SELECT tgl2.pts AS opp_pts
        FROM team_game_logs tgl1
        JOIN team_game_logs tgl2
          ON tgl1.game_id = tgl2.game_id
         AND tgl1.team_id != tgl2.team_id
        WHERE tgl1.team_name LIKE ?
        ORDER BY tgl1.game_date DESC
        LIMIT {n}
    """, conn, params=(f"%{team_name.strip().split()[-1]}%",))
    conn.close()
    if df.empty:
        return LEAGUE_AVG_TEAM_PTS
    return round(float(df["opp_pts"].mean()), 2)


def get_pace_factor(team_name: str, n: int = RECENT_N, game_logs_df: pd.DataFrame | None = None) -> float:
    """
    Pace factor relative to league average.
    Proxy: team's avg pts per game / league average (82).
    > 1.0 = fast pace, more possessions, more stat opportunities.
    < 1.0 = slow pace, fewer possessions.
    """
    if game_logs_df is not None:
        team_rows = _find_team(team_name, game_logs_df).sort_values("game_date", ascending=False).head(n)
        if team_rows.empty:
            return 1.0
        avg_pts = float(team_rows["pts"].mean())
        return round(avg_pts / LEAGUE_AVG_TEAM_PTS, 4)
    conn = get_conn()
    df   = pd.read_sql(
        f"SELECT pts FROM team_game_logs WHERE team_name LIKE ? ORDER BY game_date DESC LIMIT {n}",
        conn, params=(f"%{team_name.strip().split()[-1]}%",)
    )
    conn.close()
    if df.empty:
        return 1.0
    avg_pts = float(df["pts"].mean())
    return round(avg_pts / LEAGUE_AVG_TEAM_PTS, 4)


def get_game_pace_factor(home_team: str, away_team: str, game_logs_df: pd.DataFrame | None = None) -> float:
    """
    Expected pace for a specific matchup — average of both teams' pace factors.
    Applied to adjust player prop counting stats.
    """
    home_pf = get_pace_factor(home_team, game_logs_df=game_logs_df)
    away_pf = get_pace_factor(away_team, game_logs_df=game_logs_df)
    return round((home_pf + away_pf) / 2, 4)


def get_def_rating_adj(opp_pts_allowed: float) -> float:
    """
    Convert opponent's pts-allowed into an adjustment multiplier.
    Stronger defense (fewer pts allowed) = harder environment for scoring.
    Dampened to 35% effect.
    """
    adj = 1.0 + 0.35 * ((opp_pts_allowed - LEAGUE_AVG_TEAM_PTS) / LEAGUE_AVG_TEAM_PTS)
    return float(np.clip(adj, 0.70, 1.30))


if __name__ == "__main__":
    teams = ["Las Vegas Aces", "Minnesota Lynx", "Connecticut Sun", "Chicago Sky"]
    print(f"{'Team':<25} {'Rest days':>10} {'Opp pts allowed':>16} {'Pace factor':>12} {'Def adj':>8}")
    print("-" * 75)
    for t in teams:
        rest    = get_rest_days(t)
        opp_pts = get_opp_pts_allowed(t)
        pace    = get_pace_factor(t)
        def_adj = get_def_rating_adj(opp_pts)
        print(f"{t:<25} {rest:>10.1f} {opp_pts:>16.1f} {pace:>12.4f} {def_adj:>8.3f}")
