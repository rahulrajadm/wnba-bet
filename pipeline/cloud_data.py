"""
Fetch WNBA team and player game logs in-memory from NBA Stats API.
Used by the Streamlit Cloud app (no SQLite).
"""
import time
import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playergamelogs

LEAGUE_ID = "10"
SEASON    = "2026"
TIMEOUT   = 30


def fetch_team_game_logs(season: str = SEASON) -> pd.DataFrame:
    try:
        log = leaguegamelog.LeagueGameLog(
            league_id=LEAGUE_ID, season=season,
            season_type_all_star="Regular Season", timeout=TIMEOUT,
        )
        df = log.get_data_frames()[0]
        rows = []
        for _, r in df.iterrows():
            rows.append({
                "game_id":    r["GAME_ID"],
                "season":     season,
                "team_id":    str(r["TEAM_ID"]),
                "team_name":  r["TEAM_NAME"],
                "team_abbr":  r["TEAM_ABBREVIATION"],
                "game_date":  r["GAME_DATE"],
                "matchup":    r["MATCHUP"],
                "wl":         r["WL"],
                "pts":        r["PTS"],
                "fg_pct":     r["FG_PCT"],
                "fg3_pct":    r["FG3_PCT"],
                "ft_pct":     r["FT_PCT"],
                "reb":        r["REB"],
                "ast":        r["AST"],
                "stl":        r["STL"],
                "blk":        r["BLK"],
                "tov":        r["TOV"],
                "plus_minus": r["PLUS_MINUS"],
            })
        return pd.DataFrame(rows)
    except Exception:
        return pd.DataFrame()


def fetch_player_game_logs(season: str = SEASON) -> pd.DataFrame:
    try:
        logs = playergamelogs.PlayerGameLogs(
            league_id_nullable=LEAGUE_ID,
            season_nullable=season,
            season_type_nullable="Regular Season",
            timeout=TIMEOUT,
        )
        df = logs.get_data_frames()[0]
        df["season"] = season
        df.rename(columns={
            "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
            "TEAM_ABBREVIATION": "team_abbr", "GAME_ID": "game_id",
            "GAME_DATE": "game_date", "MATCHUP": "matchup", "WL": "wl",
            "MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast",
            "STL": "stl", "BLK": "blk", "TOV": "tov", "FG3M": "fg3m",
            "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct",
            "FTM": "ftm", "FTA": "fta", "PLUS_MINUS": "plus_minus",
        }, inplace=True)
        keep = ["season", "player_id", "player_name", "team_abbr", "game_id",
                "game_date", "matchup", "wl", "min", "pts", "reb", "ast",
                "stl", "blk", "tov", "fg3m", "fgm", "fga", "fg_pct",
                "ftm", "fta", "plus_minus"]
        return df[[c for c in keep if c in df.columns]]
    except Exception:
        return pd.DataFrame()
