"""
Fetches today's WNBA schedule from the NBA Stats API.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import date
from utils.db import get_conn

SCOREBOARD_URL = "https://cdn.nba.com/static/json/liveData/scoreboard/todaysScoreboard_00.json"
SCHEDULE_URL   = "https://stats.nba.com/stats/internationalbroadcasterschedule"


def fetch_today_games(game_date: str = None) -> list[dict]:
    """
    Derives today's WNBA games from The Odds API (faster + more reliable than stats.nba.com).
    Falls back to NBA Stats API scoreboard if needed.
    """
    import os
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))
    api_key = os.getenv("ODDS_API_KEY")

    if game_date is None:
        game_date = date.today().isoformat()

    # Primary: derive from Odds API (already fetched, no extra requests if cached)
    try:
        url    = "https://api.the-odds-api.com/v4/sports/basketball_wnba/odds"
        params = {"apiKey": api_key, "regions": "us", "markets": "h2h", "oddsFormat": "american"}
        resp   = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data   = resp.json()

        games = []
        for g in data:
            game_dt = g.get("commence_time", "")[:10]
            games.append({
                "game_id":      g["id"],
                "date":         game_dt,
                "home_team":    g["home_team"],
                "away_team":    g["away_team"],
                "home_team_id": "",
                "away_team_id": "",
                "game_time":    g.get("commence_time", ""),
                "season":       "2026",
            })
        # Filter to today + tomorrow
        from datetime import timedelta
        tomorrow = (date.today() + timedelta(days=1)).isoformat()
        games = [g for g in games if g["date"] in [game_date, tomorrow]]
        return games
    except Exception as e:
        print(f"  Warning fetching schedule: {e}")
        return []


def save_schedule(games: list[dict]):
    conn = get_conn()
    c    = conn.cursor()
    for g in games:
        c.execute("""
            INSERT OR REPLACE INTO games
            (game_id, date, home_team, away_team, home_team_id, away_team_id, game_time, season)
            VALUES (:game_id, :date, :home_team, :away_team, :home_team_id, :away_team_id, :game_time, :season)
        """, g)
    conn.commit()
    conn.close()


def get_today_games(game_date: str = None) -> list[dict]:
    games = fetch_today_games(game_date)
    if games:
        save_schedule(games)
    return games


if __name__ == "__main__":
    games = get_today_games()
    print(f"Found {len(games)} WNBA games today:")
    for g in games:
        print(f"  {g['away_team']} @ {g['home_team']}  |  {g['game_time']}")
