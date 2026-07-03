"""
Fetches WNBA moneyline, spread, and totals odds from The Odds API.
Sport key: basketball_wnba
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import datetime, timezone
from dotenv import load_dotenv
from utils.db import get_conn, set_meta

load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

API_KEY  = os.getenv("ODDS_API_KEY")
BASE_URL = "https://api.the-odds-api.com/v4"
SPORT    = "basketball_wnba"
REGION   = "us"
MARKETS  = ["h2h", "spreads", "totals"]


def fetch_odds(market: str) -> list[dict]:
    url    = f"{BASE_URL}/sports/{SPORT}/odds"
    params = {
        "apiKey":      API_KEY,
        "regions":     REGION,
        "markets":     market,
        "oddsFormat":  "american",
        "dateFormat":  "iso",
    }
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    remaining = resp.headers.get("x-requests-remaining")
    print(f"  Odds API requests remaining: {remaining or 'N/A'}")
    if remaining is not None:
        set_meta("odds_api_remaining", remaining)
    return resp.json()


def parse_and_save(games: list[dict], market: str):
    conn       = get_conn()
    c          = conn.cursor()
    fetched_at = datetime.now(timezone.utc).isoformat()

    for game in games:
        game_id    = game["id"]
        home_team  = game["home_team"]
        away_team  = game["away_team"]

        for bookmaker in game.get("bookmakers", []):
            book = bookmaker["key"]
            for mkt in bookmaker.get("markets", []):
                if mkt["key"] != market:
                    continue

                outcomes = {o["name"]: o for o in mkt["outcomes"]}

                if market == "h2h":
                    c.execute("""
                        INSERT INTO game_odds
                        (fetched_at, platform, game_id, home_team, away_team, market,
                         home_odds, away_odds, home_spread, away_spread, over_odds, under_odds, total_line)
                        VALUES (?,?,?,?,?,?,?,?,NULL,NULL,NULL,NULL,NULL)
                    """, (fetched_at, book, game_id, home_team, away_team, "moneyline",
                          outcomes.get(home_team, {}).get("price"),
                          outcomes.get(away_team, {}).get("price")))

                elif market == "spreads":
                    home_o = outcomes.get(home_team, {})
                    away_o = outcomes.get(away_team, {})
                    c.execute("""
                        INSERT INTO game_odds
                        (fetched_at, platform, game_id, home_team, away_team, market,
                         home_odds, away_odds, home_spread, away_spread, over_odds, under_odds, total_line)
                        VALUES (?,?,?,?,?,?,?,?,?,?,NULL,NULL,NULL)
                    """, (fetched_at, book, game_id, home_team, away_team, "spread",
                          home_o.get("price"), away_o.get("price"),
                          home_o.get("point"), away_o.get("point")))

                elif market == "totals":
                    over  = outcomes.get("Over",  {})
                    under = outcomes.get("Under", {})
                    c.execute("""
                        INSERT INTO game_odds
                        (fetched_at, platform, game_id, home_team, away_team, market,
                         home_odds, away_odds, home_spread, away_spread, over_odds, under_odds, total_line)
                        VALUES (?,?,?,?,?,?,NULL,NULL,NULL,NULL,?,?,?)
                    """, (fetched_at, book, game_id, home_team, away_team, "totals",
                          over.get("price"), under.get("price"), over.get("point")))

    conn.commit()
    conn.close()


def get_all_odds():
    for market in MARKETS:
        print(f"  Fetching WNBA {market} odds...")
        games = fetch_odds(market)
        parse_and_save(games, market)
        print(f"    Saved {len(games)} games")


if __name__ == "__main__":
    get_all_odds()
