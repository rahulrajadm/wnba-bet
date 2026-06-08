"""
Fetch WNBA player injury / availability status from ESPN for today's games.
No API key required. Returns non-active players per team.
"""
import requests
from datetime import date
from concurrent.futures import ThreadPoolExecutor, as_completed

TIMEOUT = 10

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_ACTIVE = {"active"}


def _get(url: str, params: dict | None = None) -> dict:
    r = requests.get(url, headers=_HEADERS, params=params or {}, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def _today_team_ids() -> dict[str, str]:
    """Return {ESPN displayName: team_id} for every team playing today."""
    today = date.today().strftime("%Y%m%d")
    data  = _get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
        {"dates": today},
    )
    team_map = {}
    for event in data.get("events", []):
        comp = event.get("competitions", [{}])[0]
        for c in comp.get("competitors", []):
            team = c.get("team", {})
            name = team.get("displayName", "")
            tid  = str(team.get("id", ""))
            if name and tid:
                team_map[name] = tid
    return team_map


def _roster_flags(team_id: str) -> list[dict]:
    """Fetch a team's roster and return any non-active players."""
    try:
        data = _get(
            f"https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/teams/{team_id}/roster"
        )
    except Exception:
        return []

    flags = []
    for athlete in data.get("athletes", []):
        s    = athlete.get("status", {})
        stype = s.get("type", "active").lower()
        sname = s.get("name", "Active")
        if stype not in _ACTIVE:
            flags.append({
                "name":   athlete.get("fullName", "Unknown"),
                "status": sname,
            })
    return flags


def fetch_injury_flags(team_names: list[str]) -> dict[str, list[dict]]:
    """
    Return {team_name: [{"name": player, "status": "Out"/"Questionable"/...}]}
    for teams that have at least one non-active player today.

    team_names: list of team names as they appear in the Odds API schedule.
    """
    try:
        today_map = _today_team_ids()
    except Exception:
        return {}

    # Build name → team_id, fuzzy-matching Odds API names to ESPN display names
    resolved: dict[str, str] = {}
    for name in team_names:
        tid = today_map.get(name)
        if not tid:
            last = name.strip().split()[-1].lower()
            for espn_name, tid2 in today_map.items():
                if last in espn_name.lower():
                    tid = tid2
                    break
        if tid:
            resolved[name] = tid

    if not resolved:
        return {}

    result: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=len(resolved)) as ex:
        futures = {ex.submit(_roster_flags, tid): name for name, tid in resolved.items()}
        for fut in as_completed(futures):
            name  = futures[fut]
            flags = fut.result()
            if flags:
                result[name] = flags

    return result
