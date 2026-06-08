"""
Fetch WNBA team and player game logs in-memory from NBA Stats API.
Used by the Streamlit Cloud app (no SQLite).
"""
import pandas as pd
import requests

LEAGUE_ID = "10"
SEASON    = "2026"
TIMEOUT   = 30

# stats.nba.com blocks requests without browser-like headers on cloud IPs
_HEADERS = {
    "Accept":               "*/*",
    "Accept-Language":      "en-US,en;q=0.9",
    "Host":                 "stats.nba.com",
    "Origin":               "https://www.nba.com",
    "Referer":              "https://www.nba.com/",
    "User-Agent":           (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "x-nba-stats-origin":   "stats",
    "x-nba-stats-token":    "true",
}


def _get(url: str, params: dict) -> dict:
    resp = requests.get(url, headers=_HEADERS, params=params, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp.json()


def fetch_team_game_logs(season: str = SEASON) -> pd.DataFrame:
    try:
        data = _get(
            "https://stats.nba.com/stats/leaguegamelog",
            {
                "Counter": "0", "DateFrom": "", "DateTo": "",
                "Direction": "DESC", "LeagueID": LEAGUE_ID,
                "PlayerOrTeam": "T", "Season": season,
                "SeasonType": "Regular Season", "Sorter": "DATE",
            },
        )
        rs   = data["resultSets"][0]
        cols = [c.lower() for c in rs["headers"]]
        df   = pd.DataFrame(rs["rowSet"], columns=cols)
        if df.empty:
            return pd.DataFrame()
        rename = {
            "game_id": "game_id", "team_id": "team_id",
            "team_name": "team_name", "team_abbreviation": "team_abbr",
            "game_date": "game_date", "matchup": "matchup", "wl": "wl",
            "pts": "pts", "fg_pct": "fg_pct", "fg3_pct": "fg3_pct",
            "ft_pct": "ft_pct", "reb": "reb", "ast": "ast",
            "stl": "stl", "blk": "blk", "tov": "tov",
            "plus_minus": "plus_minus",
        }
        keep = [c for c in rename if c in df.columns]
        df   = df[keep].rename(columns=rename)
        df["team_id"] = df["team_id"].astype(str)
        df["season"]  = season
        return df
    except Exception:
        return pd.DataFrame()


def _fetch_player_logs_for_season(season: str) -> pd.DataFrame:
    data = _get(
        "https://stats.nba.com/stats/playergamelogs",
        {
            "DateFrom": "", "DateTo": "", "GameSegment": "",
            "LastNGames": "0", "LeagueID": LEAGUE_ID,
            "Location": "", "MeasureType": "Base", "Month": "0",
            "OpponentTeamID": "0", "Outcome": "", "PORound": "0",
            "PerMode": "PerGame", "Period": "0", "PlayerID": "",
            "Season": season, "SeasonSegment": "",
            "SeasonType": "Regular Season", "ShotClockRange": "",
            "TeamID": "0", "VsConference": "", "VsDivision": "",
        },
    )
    rs   = data["resultSets"][0]
    cols = [c.lower() for c in rs["headers"]]
    df   = pd.DataFrame(rs["rowSet"], columns=cols)
    if df.empty:
        return pd.DataFrame()
    rename = {
        "player_id": "player_id", "player_name": "player_name",
        "team_abbreviation": "team_abbr", "game_id": "game_id",
        "game_date": "game_date", "matchup": "matchup", "wl": "wl",
        "min": "min", "pts": "pts", "reb": "reb", "ast": "ast",
        "stl": "stl", "blk": "blk", "tov": "tov", "fg3m": "fg3m",
        "fgm": "fgm", "fga": "fga", "fg_pct": "fg_pct",
        "ftm": "ftm", "fta": "fta", "plus_minus": "plus_minus",
    }
    keep = [c for c in rename if c in df.columns]
    df   = df[keep].rename(columns=rename)
    df["season"] = season
    return df


def fetch_player_game_logs(season: str = SEASON) -> pd.DataFrame:
    # Try current season first; fall back to prior season if not indexed yet
    for s in [season, str(int(season) - 1)]:
        try:
            df = _fetch_player_logs_for_season(s)
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()
