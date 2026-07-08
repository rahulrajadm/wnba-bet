"""
Fetches live WNBA player prop lines from PrizePicks.
WNBA league_id = 3

PrizePicks' API is behind DataDome bot protection, so a plain requests call returns 403.
Two code paths deliberately diverge here (see CLAUDE.md's two-entry-points split):

  * fetch_wnba_lines()   — plain requests. Used by the CLOUD app (ui/app_cloud.py). It
                           still 403s on Streamlit Cloud's datacenter IP; the cloud app
                           already treats that as "no PrizePicks" and falls back to Underdog.
                           Left untouched on purpose so the cloud build stays lean.
  * get_prizepicks_lines() — used by the LOCAL app (ui/app.py). Mints a DataDome cookie in a
                           real (headed) browser once, caches it, and reuses it via curl_cffi.
                           Needs the local-only deps in requirements-local.txt
                           (curl_cffi, playwright + `playwright install chromium`).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import requests
from datetime import datetime, timezone
from utils.db import get_conn, ensure_schema

PRIZEPICKS_URL  = "https://api.prizepicks.com/projections"
WNBA_LEAGUE_ID  = 3
_PARAMS         = {"league_id": WNBA_LEAGUE_ID, "per_page": 250, "single_stat": "true"}
_PROJ_URL       = f"{PRIZEPICKS_URL}?league_id={WNBA_LEAGUE_ID}&per_page=250&single_stat=true"
_COOKIE_PATH    = os.path.join(os.path.dirname(__file__), "..", "data", ".pp_cookie.json")

_TEAM_ABBR: dict[str, str] = {
    "ATL": "Atlanta Dream",
    "CHI": "Chicago Sky",
    "CON": "Connecticut Sun",
    "DAL": "Dallas Wings",
    "GSV": "Golden State Valkyries",
    "IND": "Indiana Fever",
    "LAS": "Las Vegas Aces",
    "LVA": "Las Vegas Aces",
    "MIN": "Minnesota Lynx",
    "NY":  "New York Liberty",
    "NYL": "New York Liberty",
    "PHX": "Phoenix Mercury",
    "POR": "Portland Fire",
    "SEA": "Seattle Storm",
    "TOR": "Toronto Tempo",
    "WAS": "Washington Mystics",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept":     "application/json",
    "Referer":    "https://app.prizepicks.com/",
}


def fetch_wnba_lines() -> list[dict]:
    """Plain-requests fetch used by the cloud app. 403s behind DataDome — callers must
    handle failure (the cloud app does). Local callers use get_prizepicks_lines()."""
    resp = requests.get(PRIZEPICKS_URL, headers=HEADERS, params=_PARAMS, timeout=15)
    resp.raise_for_status()
    return _parse_projections(resp.json())


def _parse_projections(data: dict) -> list[dict]:
    projections = data.get("data", [])
    included    = {item["id"]: item for item in data.get("included", [])}

    props = []
    for proj in projections:
        attrs    = proj.get("attributes", {})
        rel      = proj.get("relationships", {})
        pid      = rel.get("new_player", {}).get("data", {}).get("id")
        pinfo    = included.get(pid, {}).get("attributes", {}) if pid else {}

        # odds_type controls which directions are available on this line:
        # "standard" → both More and Less
        # "demon"    → More only (elevated line, harder to hit)
        # "goblin"   → More only (lowered line, easier to hit — still More only on PrizePicks)
        odds_type = attrs.get("odds_type", "standard")
        if odds_type in ("demon", "goblin"):
            allowed = "More"
        else:
            allowed = None   # both directions available

        raw_abbr = pinfo.get("team", "")
        player_team = _TEAM_ABBR.get(raw_abbr, raw_abbr)

        props.append({
            "platform":          "prizepicks",
            "fetched_at":        datetime.now(timezone.utc).isoformat(),
            "player_name":       pinfo.get("display_name", attrs.get("description", "")),
            "player_team":       player_team,
            "stat_type":         attrs.get("stat_type", ""),
            "line":              attrs.get("line_score"),
            "game_id":           attrs.get("game_id", ""),
            "allowed_direction": allowed,
            "odds_type":         odds_type,
            "more_odds":         None,
            "less_odds":         None,
        })
    return props


def save_lines(props: list[dict]):
    conn = get_conn()
    ensure_schema(conn)
    c    = conn.cursor()
    for p in props:
        c.execute("""
            INSERT INTO prop_lines
            (fetched_at, platform, game_id, player_name, player_team, stat_type, line, more_odds, less_odds,
             odds_type, allowed_direction)
            VALUES (:fetched_at, :platform, :game_id, :player_name, :player_team, :stat_type, :line, :more_odds, :less_odds,
                    :odds_type, :allowed_direction)
        """, p)
    conn.commit()
    conn.close()


# ── Local DataDome workaround ─────────────────────────────────────────────────────────
# DataDome fingerprints the TLS/JS client and the IP. A headless browser is detected and
# handed a "challenge" cookie that still 403s, so we mint the cookie in a HEADED browser
# (works from a residential IP), cache it, and reuse it with curl_cffi — which impersonates
# Chrome's TLS fingerprint. The browser only launches when the cached cookie is missing or
# stale, so day-to-day refreshes are a plain HTTP call with no window popping up.

def _load_cached_cookie() -> tuple[str | None, str | None]:
    try:
        with open(_COOKIE_PATH) as f:
            d = json.load(f)
        return d.get("datadome"), d.get("user_agent")
    except Exception:
        return None, None


def _save_cached_cookie(datadome: str, user_agent: str) -> None:
    try:
        with open(_COOKIE_PATH, "w") as f:
            json.dump({"datadome": datadome, "user_agent": user_agent}, f)
    except Exception:
        pass


def _mint_datadome_cookie() -> tuple[str, str]:
    """Open a real (headed) browser so DataDome's JS challenge runs and sets a valid
    datadome cookie, then return (cookie, user_agent). Raises if minting fails."""
    from playwright.sync_api import sync_playwright  # local-only dep; imported lazily

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=False, args=["--disable-blink-features=AutomationControlled"]
        )
        try:
            ctx  = browser.new_context()
            page = ctx.new_page()
            page.goto("https://app.prizepicks.com/", wait_until="domcontentloaded", timeout=30000)
            # Fire the projections request from inside the page so DataDome runs its
            # challenge and commits the cookie before we read it back.
            page.evaluate(
                "async (u) => { try { await fetch(u, {credentials:'include'}); } catch(e) {} }",
                _PROJ_URL,
            )
            page.wait_for_timeout(2500)
            user_agent = page.evaluate("() => navigator.userAgent")
            cookies    = ctx.cookies()
        finally:
            browser.close()

    datadome = next((c["value"] for c in cookies if c["name"] == "datadome"), None)
    if not datadome:
        raise RuntimeError("PrizePicks: browser did not produce a datadome cookie")
    return datadome, user_agent


def _fetch_with_cookie(datadome: str, user_agent: str):
    from curl_cffi import requests as creq  # local-only dep; imported lazily

    return creq.get(
        PRIZEPICKS_URL,
        params=_PARAMS,
        impersonate="chrome124",
        headers={"User-Agent": user_agent, "Accept": "application/json",
                 "Referer": "https://app.prizepicks.com/"},
        cookies={"datadome": datadome},
        timeout=20,
    )


def _fetch_projections_local() -> dict:
    """Fetch raw projections JSON, minting/refreshing the DataDome cookie as needed."""
    datadome, user_agent = _load_cached_cookie()
    if datadome and user_agent:
        resp = _fetch_with_cookie(datadome, user_agent)
        if resp.status_code == 200:
            return resp.json()
    # Cached cookie missing or stale (403) → mint a fresh one and retry once.
    datadome, user_agent = _mint_datadome_cookie()
    _save_cached_cookie(datadome, user_agent)
    resp = _fetch_with_cookie(datadome, user_agent)
    resp.raise_for_status()
    return resp.json()


def get_prizepicks_lines() -> list[dict]:
    """Local fetch that gets past DataDome (see module docstring). Cloud uses
    fetch_wnba_lines() instead and is unaffected by this path."""
    props = _parse_projections(_fetch_projections_local())
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
