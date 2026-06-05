"""
Fetches live WNBA player prop lines from PrizePicks.
WNBA league_id = 3
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import datetime, timezone
from utils.db import get_conn

PRIZEPICKS_URL  = "https://api.prizepicks.com/projections"
WNBA_LEAGUE_ID  = 3

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json",
    "Referer":    "https://app.prizepicks.com/",
}


def fetch_wnba_lines() -> list[dict]:
    params = {"league_id": WNBA_LEAGUE_ID, "per_page": 250, "single_stat": "true"}
    resp   = requests.get(PRIZEPICKS_URL, headers=HEADERS, params=params, timeout=15)
    resp.raise_for_status()
    data   = resp.json()

    projections = data.get("data", [])
    included    = {item["id"]: item for item in data.get("included", [])}

    props = []
    for proj in projections:
        attrs    = proj.get("attributes", {})
        rel      = proj.get("relationships", {})
        pid      = rel.get("new_player", {}).get("data", {}).get("id")
        pinfo    = included.get(pid, {}).get("attributes", {}) if pid else {}

        props.append({
            "platform":    "prizepicks",
            "fetched_at":  datetime.now(timezone.utc).isoformat(),
            "player_name": pinfo.get("display_name", attrs.get("description", "")),
            "player_team": pinfo.get("team", ""),
            "stat_type":   attrs.get("stat_type", ""),
            "line":        attrs.get("line_score"),
            "game_id":     attrs.get("game_id", ""),
            "more_odds":   None,
            "less_odds":   None,
        })
    return props


def save_lines(props: list[dict]):
    conn = get_conn()
    c    = conn.cursor()
    for p in props:
        c.execute("""
            INSERT INTO prop_lines
            (fetched_at, platform, game_id, player_name, player_team, stat_type, line, more_odds, less_odds)
            VALUES (:fetched_at, :platform, :game_id, :player_name, :player_team, :stat_type, :line, :more_odds, :less_odds)
        """, p)
    conn.commit()
    conn.close()


def get_prizepicks_lines() -> list[dict]:
    props = fetch_wnba_lines()
    save_lines(props)
    return props


if __name__ == "__main__":
    props = get_prizepicks_lines()
    print(f"Fetched {len(props)} PrizePicks WNBA props:")
    stats = {}
    for p in props:
        stats[p["stat_type"]] = stats.get(p["stat_type"], 0) + 1
    for stat, cnt in sorted(stats.items(), key=lambda x: -x[1])[:10]:
        print(f"  {stat}: {cnt}")
