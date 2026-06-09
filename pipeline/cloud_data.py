"""
Fetch WNBA game logs from the ESPN unofficial API.
Replaces stats.nba.com (unreliable on cloud IPs). No API key required.
Covers all teams including 2026 expansion franchises.
"""
import pandas as pd
import requests
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT          = 15
LOOKBACK         = 35   # days for team logs — current-season form only
PLAYER_SEASONS   = 2    # calendar years of player data (current + N-1 prior seasons)
WORKERS          = 6    # parallel box-score requests — keeps ESPN from rate-limiting cloud IPs

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_LABEL_MAP = {
    "PTS": "pts", "REB": "reb", "AST": "ast",
    "STL": "stl", "BLK": "blk", "TO": "tov",
}


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, headers=_HEADERS, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _completed_game_ids(lookback: int = LOOKBACK) -> list[str]:
    """Completed game IDs from the last `lookback` days (single range query)."""
    end   = date.today()
    start = end - timedelta(days=lookback)
    data  = _get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
        {"dates": f"{start.strftime('%Y%m%d')}-{end.strftime('%Y%m%d')}", "limit": 999},
    )
    return [
        e["id"] for e in data.get("events", [])
        if e.get("competitions", [{}])[0]
            .get("status", {}).get("type", {}).get("completed")
    ]


def _completed_game_ids_multiseason(num_seasons: int = PLAYER_SEASONS) -> list[str]:
    """
    Completed game IDs spanning the current and prior calendar years.
    ESPN rejects cross-year date ranges, so each year is queried separately.
    """
    global _season_game_counts
    _season_game_counts = {}
    current_year = date.today().year
    seen: set[str] = set()
    ids: list[str] = []

    for yr in range(current_year - num_seasons + 1, current_year + 1):
        try:
            data = _get(
                "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
                {"dates": f"{yr}0101-{yr}1231", "limit": 999},
            )
            yr_ids = [
                e["id"] for e in data.get("events", [])
                if e.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed")
                and e["id"] not in seen
            ]
            for gid in yr_ids:
                seen.add(gid)
                ids.append(gid)
            _season_game_counts[yr] = len(yr_ids)
            import sys
            print(f"[cloud_data] ESPN {yr}: {len(yr_ids)} completed games", file=sys.stderr)
        except Exception as _e:
            import sys
            print(f"[cloud_data] ESPN scoreboard query {yr} failed: {_e}", file=sys.stderr)

    return ids


def _box_score_team_rows(event_id: str) -> list[dict]:
    data = _get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary",
        {"event": event_id},
    )
    header = data.get("header", {}).get("competitions", [{}])[0]
    game_date = header.get("date", "")[:10]

    # Score by team_id from the header competitors
    scores = {}
    for c in header.get("competitors", []):
        tid = str(c.get("team", {}).get("id", ""))
        try:
            scores[tid] = float(c.get("score", 0) or 0)
        except (ValueError, TypeError):
            scores[tid] = 0.0

    rows = []
    for team_entry in data.get("boxscore", {}).get("players", []):
        team     = team_entry.get("team", {})
        team_id  = str(team.get("id", ""))
        groups   = team_entry.get("statistics", [])
        if not groups:
            continue
        labels   = groups[0].get("labels", [])
        athletes = groups[0].get("athletes", [])

        # Aggregate counting stats across all players on this team
        agg = {col: 0.0 for col in _LABEL_MAP.values()}
        fgm = fga = fg3m = fg3a = 0.0

        for ae in athletes:
            raw = ae.get("stats", [])
            if not raw or set(raw) == {"--"}:
                continue
            for lbl, col in _LABEL_MAP.items():
                if lbl in labels:
                    try:
                        agg[col] += float(raw[labels.index(lbl)])
                    except (ValueError, TypeError):
                        pass
            if "FG" in labels:
                try:
                    m, a = raw[labels.index("FG")].split("-")
                    fgm += float(m); fga += float(a)
                except Exception:
                    pass
            if "3PT" in labels:
                try:
                    m, a = raw[labels.index("3PT")].split("-")
                    fg3m += float(m); fg3a += float(a)
                except Exception:
                    pass

        pts = scores.get(team_id, agg["pts"])
        rows.append({
            "game_id":    event_id,
            "season":     str(date.today().year),
            "team_id":    team_id,
            "team_name":  team.get("displayName", ""),
            "team_abbr":  team.get("abbreviation", ""),
            "game_date":  game_date,
            "wl":         "",   # filled after both teams are known
            "pts":        pts,
            "fg_pct":     round(fgm / fga, 3) if fga > 0 else 0.0,
            "fg3_pct":    round(fg3m / fg3a, 3) if fg3a > 0 else 0.0,
            "ft_pct":     0.0,
            "reb":        agg["reb"],
            "ast":        agg["ast"],
            "stl":        agg["stl"],
            "blk":        agg["blk"],
            "tov":        agg["tov"],
            "plus_minus": 0.0,
        })

    # Assign W / L
    if len(rows) == 2:
        rows[0]["wl"] = "W" if rows[0]["pts"] > rows[1]["pts"] else "L"
        rows[1]["wl"] = "W" if rows[1]["pts"] > rows[0]["pts"] else "L"

    return rows


def _box_score_player_rows(event_id: str) -> list[dict]:
    data = _get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/summary",
        {"event": event_id},
    )
    header    = data.get("header", {}).get("competitions", [{}])[0]
    game_date = header.get("date", "")[:10]
    rows = []

    for team_entry in data.get("boxscore", {}).get("players", []):
        team     = team_entry.get("team", {})
        team_abbr = team.get("abbreviation", "")
        groups   = team_entry.get("statistics", [])
        if not groups:
            continue
        labels   = groups[0].get("labels", [])

        for ae in groups[0].get("athletes", []):
            athlete = ae.get("athlete", {})
            raw     = ae.get("stats", [])
            if not raw or set(raw) == {"--"}:
                continue

            row = {
                "game_id":     event_id,
                "game_date":   game_date,
                "player_id":   str(athlete.get("id", "")),
                "player_name": athlete.get("displayName", ""),
                "team_abbr":   team_abbr,
                "pts": 0.0, "reb": 0.0, "ast": 0.0,
                "stl": 0.0, "blk": 0.0, "tov": 0.0,
                "fg3m": 0.0, "fgm": 0.0, "fga": 0.0,
                "fg_pct": 0.0, "ftm": 0.0, "plus_minus": 0.0,
            }
            for lbl, col in _LABEL_MAP.items():
                if lbl in labels:
                    try:
                        row[col] = float(raw[labels.index(lbl)])
                    except (ValueError, TypeError):
                        pass
            if "FG" in labels:
                try:
                    m, a = raw[labels.index("FG")].split("-")
                    row["fgm"] = float(m); row["fga"] = float(a)
                    row["fg_pct"] = round(float(m) / float(a), 3) if float(a) > 0 else 0.0
                except Exception:
                    pass
            if "3PT" in labels:
                try:
                    row["fg3m"] = float(raw[labels.index("3PT")].split("-")[0])
                except (ValueError, TypeError):
                    pass
            if "FT" in labels:
                try:
                    row["ftm"] = float(raw[labels.index("FT")].split("-")[0])
                except (ValueError, TypeError):
                    pass
            if "+/-" in labels:
                try:
                    row["plus_minus"] = float(raw[labels.index("+/-")])
                except (ValueError, TypeError):
                    pass
            rows.append(row)

    return rows


_last_fetch_failures: int = 0
_season_game_counts: dict[int, int] = {}  # populated by _completed_game_ids_multiseason


def _fetch_all(game_ids: list[str], parse_fn) -> pd.DataFrame:
    global _last_fetch_failures
    _last_fetch_failures = 0
    try:
        all_rows = []
        failed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            futures = {ex.submit(parse_fn, gid): gid for gid in game_ids}
            for fut in as_completed(futures):
                try:
                    all_rows.extend(fut.result())
                except Exception:
                    failed += 1
        _last_fetch_failures = failed
        return pd.DataFrame(all_rows) if all_rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def fetch_team_game_logs(season: str = str(date.today().year)) -> pd.DataFrame:
    """Current-season rolling form only (LOOKBACK days). Used by game prediction model."""
    return _fetch_all(_completed_game_ids(lookback=LOOKBACK), _box_score_team_rows)


def fetch_player_game_logs(season: str = str(date.today().year)) -> pd.DataFrame:
    """Multi-season player history (PLAYER_SEASONS years). Used by props model.
    Covers the full prior season + current season for a stable baseline average,
    while team logs remain current-season only so game predictions aren't polluted.
    """
    return _fetch_all(_completed_game_ids_multiseason(PLAYER_SEASONS), _box_score_player_rows)
