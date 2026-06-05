"""
Fetches live WNBA player prop lines from Underdog Fantasy.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
from datetime import datetime, timezone
from utils.db import get_conn

UNDERDOG_URL = "https://api.underdogfantasy.com/beta/v5/over_under_lines"
HEADERS      = {"User-Agent": "Mozilla/5.0", "Accept": "application/json"}


def fetch_wnba_lines() -> list[dict]:
    resp = requests.get(UNDERDOG_URL, headers=HEADERS, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    games       = {g["id"]: g for g in data.get("games", [])}
    appearances = {a["id"]: a for a in data.get("appearances", [])}
    players     = {p["id"]: p for p in data.get("players", [])}

    wnba_match_ids = {gid for gid, g in games.items() if g.get("sport_id") == "WNBA"}

    props = []
    for line in data.get("over_under_lines", []):
        app_id     = line.get("over_under", {}).get("appearance_stat", {}).get("appearance_id")
        appearance = appearances.get(app_id, {})
        match_id   = appearance.get("match_id")

        if match_id not in wnba_match_ids:
            continue

        player_id = appearance.get("player_id")
        player    = players.get(player_id, {})
        stat      = line.get("over_under", {}).get("appearance_stat", {}).get("display_stat", "")
        ou_line   = line.get("stat_value")

        props.append({
            "platform":    "underdog",
            "fetched_at":  datetime.now(timezone.utc).isoformat(),
            "player_name": f"{player.get('first_name', '')} {player.get('last_name', '')}".strip(),
            "player_team": appearance.get("team_id", ""),
            "stat_type":   stat,
            "line":        float(ou_line) if ou_line is not None else None,
            "game_id":     str(match_id),
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


def get_underdog_lines() -> list[dict]:
    props = fetch_wnba_lines()
    save_lines(props)
    return props


if __name__ == "__main__":
    props = get_underdog_lines()
    print(f"Fetched {len(props)} Underdog WNBA props:")
    for p in props[:8]:
        print(f"  {p['player_name']} | {p['stat_type']} | {p['line']}")
