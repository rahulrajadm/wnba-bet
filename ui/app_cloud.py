"""
WNBA Bet — Streamlit Community Cloud version.
Fetches all data in-memory (no SQLite). Refresh is passcode-gated.
"""
import sys, os, shutil, pathlib

# Clear stale .pyc bytecode on first startup so Streamlit Cloud always
# runs the current source — without this, cached .pyc files survive
# deployments and execute old code even after the .py files are updated.
if "picks.engine" not in sys.modules:
    for _cache in pathlib.Path(__file__).resolve().parent.parent.rglob("__pycache__"):
        shutil.rmtree(_cache, ignore_errors=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
from datetime import datetime, date
from zoneinfo import ZoneInfo
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "../.env"))

from pipeline.prizepicks import fetch_wnba_lines as pp_fetch
from pipeline.underdog import fetch_wnba_lines as ud_fetch
from pipeline.cloud_data import fetch_team_game_logs, fetch_player_game_logs
from picks.engine import build_picks, best_props_per_player, is_high_interest
from analysis.confidence import TIER_COLORS, TIER_RANK
from analysis.risk import RISK_COLORS
from analysis.ev import ev_slip

st.set_page_config(page_title="WNBA Bet", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")


# ── In-memory data loading ─────────────────────────────────────────────────────

def fetch_odds_and_schedule() -> tuple[list[dict], list[dict]]:
    """
    Single Odds API call (3 credits: h2h + spreads + totals).
    Returns (odds_rows, games) — schedule is derived for free from the same response.
    """
    import requests
    from datetime import timezone

    api_key = os.getenv("ODDS_API_KEY")
    if not api_key:
        return [], []

    fetched_at = datetime.now(timezone.utc).isoformat()
    now_utc    = datetime.now(timezone.utc)
    rows, seen_games = [], {}

    try:
        resp = requests.get(
            "https://api.the-odds-api.com/v4/sports/basketball_wnba/odds",
            params={
                "apiKey":     api_key,
                "regions":    "us",
                "markets":    "h2h,spreads,totals",
                "oddsFormat": "american",
                "dateFormat": "iso",
            },
            timeout=15,
        )
        resp.raise_for_status()
        for game in resp.json():
            game_id    = game["id"]
            home_team  = game["home_team"]
            away_team  = game["away_team"]
            commence   = game.get("commence_time", "")

            # Parse full datetime to avoid UTC/US-timezone date mismatch:
            # date[:10] can give the wrong calendar day for evening US games.
            try:
                game_dt = datetime.fromisoformat(commence.replace("Z", "+00:00"))
            except Exception:
                continue

            # Only include games that haven't started yet
            if game_dt <= now_utc:
                continue

            game_date = game_dt.strftime("%Y-%m-%d")

            if game_id not in seen_games:
                seen_games[game_id] = {
                    "game_id": game_id, "date": game_date,
                    "home_team": home_team, "away_team": away_team,
                    "home_team_id": "", "away_team_id": "",
                    "game_time": commence, "season": "2026",
                }

            for bm in game.get("bookmakers", []):
                book = bm["key"]
                mkts = {m["key"]: m for m in bm.get("markets", [])}

                if "h2h" in mkts:
                    outcomes = {o["name"]: o for o in mkts["h2h"]["outcomes"]}
                    rows.append({"fetched_at": fetched_at, "platform": book,
                                 "game_id": game_id, "home_team": home_team, "away_team": away_team,
                                 "market": "moneyline",
                                 "home_odds": outcomes.get(home_team, {}).get("price"),
                                 "away_odds": outcomes.get(away_team, {}).get("price"),
                                 "home_spread": None, "away_spread": None,
                                 "over_odds": None, "under_odds": None, "total_line": None})
                if "spreads" in mkts:
                    outcomes = {o["name"]: o for o in mkts["spreads"]["outcomes"]}
                    ho = outcomes.get(home_team, {}); ao = outcomes.get(away_team, {})
                    rows.append({"fetched_at": fetched_at, "platform": book,
                                 "game_id": game_id, "home_team": home_team, "away_team": away_team,
                                 "market": "spread",
                                 "home_odds": ho.get("price"), "away_odds": ao.get("price"),
                                 "home_spread": ho.get("point"), "away_spread": ao.get("point"),
                                 "over_odds": None, "under_odds": None, "total_line": None})
                if "totals" in mkts:
                    outcomes = {o["name"]: o for o in mkts["totals"]["outcomes"]}
                    ov = outcomes.get("Over", {}); un = outcomes.get("Under", {})
                    rows.append({"fetched_at": fetched_at, "platform": book,
                                 "game_id": game_id, "home_team": home_team, "away_team": away_team,
                                 "market": "totals",
                                 "home_odds": None, "away_odds": None,
                                 "home_spread": None, "away_spread": None,
                                 "over_odds": ov.get("price"), "under_odds": un.get("price"),
                                 "total_line": ov.get("point")})
    except Exception as e:
        st.warning(f"Odds fetch warning: {e}")

    return rows, list(seen_games.values())


@st.cache_data(show_spinner=False, ttl=14400)  # 4-hour TTL
def load_all_data():
    # Odds API: single call for schedule + all markets (3 credits total, not 4)
    odds, games = fetch_odds_and_schedule()

    # Everything else in parallel (no Odds API calls)
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_pp          = ex.submit(pp_fetch)
        f_ud          = ex.submit(ud_fetch)
        f_team_logs   = ex.submit(fetch_team_game_logs)
        f_player_logs = ex.submit(fetch_player_game_logs)
        pp_lines    = f_pp.result(timeout=20)
        ud_lines    = f_ud.result(timeout=20)
        # NBA Stats API may be slow/blocked on cloud — hard 15s deadline
        try:
            team_logs = f_team_logs.result(timeout=15)
        except Exception:
            team_logs = pd.DataFrame()
        try:
            player_logs = f_player_logs.result(timeout=15)
        except Exception:
            player_logs = pd.DataFrame()

    # Derive combo stats for props model
    if not player_logs.empty and "pts" in player_logs.columns:
        player_logs["pra"]     = player_logs["pts"] + player_logs["reb"] + player_logs["ast"]
        player_logs["pts_reb"] = player_logs["pts"] + player_logs["reb"]
        player_logs["pts_ast"] = player_logs["pts"] + player_logs["ast"]
        player_logs["reb_ast"] = player_logs["reb"] + player_logs["ast"]
        player_logs["blk_stl"] = player_logs["blk"] + player_logs["stl"]
        player_logs["fantasy"] = (player_logs["pts"] + player_logs["reb"] * 1.2
                                  + player_logs["ast"] * 1.5 + player_logs["stl"] * 3
                                  + player_logs["blk"] * 3 - player_logs["tov"])

    all_lines = []
    from datetime import timezone
    ts = datetime.now(timezone.utc).isoformat()
    for p in pp_lines:
        p.setdefault("fetched_at", ts)
        all_lines.append(p)
    for p in ud_lines:
        p.setdefault("fetched_at", ts)
        all_lines.append(p)

    return {
        "games":       games,
        "lines":       all_lines,
        "odds":        odds,
        "team_logs":   team_logs,
        "player_logs": player_logs,
        "fetched_at":  datetime.now(ZoneInfo("America/Chicago")).strftime("%b %d %Y, %I:%M %p"),
    }


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("🏀 WNBA Bet")
    st.caption("AI-powered WNBA betting decisions")
    st.divider()

    bankroll  = st.number_input("My Bankroll ($)", min_value=10.0, value=500.0, step=10.0)
    unit_size = st.number_input("1 Unit = ($)",    min_value=1.0,  value=10.0,  step=1.0)

    st.divider()
    with st.expander("🔄 Refresh Data"):
        code = st.text_input("Passcode", type="password", key="refresh_code")
        if st.button("Refresh All Data", use_container_width=True):
            correct = st.secrets.get("REFRESH_CODE", "")
            if code == correct and correct != "":
                st.cache_data.clear()
                st.success("Cache cleared — reloading…")
                st.rerun()
            else:
                st.error("Invalid passcode")

    # Data is loaded after this point — timestamp shown in second sidebar block
    st.divider()
    min_conf  = st.selectbox("Min Confidence", ["LOW", "MEDIUM", "HIGH", "STRONG"], index=1)
    platforms = st.multiselect("Prop Platforms", ["prizepicks", "underdog"], default=["prizepicks", "underdog"])
    show_game = st.toggle("Show game picks", value=True)
    show_prop = st.toggle("Show player props", value=True)


# ── Load data ──────────────────────────────────────────────────────────────────

with st.spinner("Loading today's picks…"):
    data = load_all_data()

with st.sidebar:
    st.caption(f"🕐 Last updated: **{data['fetched_at']}** CT")
    st.caption("Data: PrizePicks · Underdog · The Odds API")
    tl = data.get("team_logs")
    pl = data.get("player_logs")
    st.caption(
        f"Games: {len(data['games'])} · "
        f"Lines: {len(data['lines'])} · "
        f"Team logs: {len(tl) if tl is not None and not tl.empty else 0} rows · "
        f"Player logs: {len(pl) if pl is not None and not pl.empty else 0} rows"
    )


# ── Build picks ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_picks_cloud(bankroll, unit_size, _cache_key):
    return build_picks(
        games=data["games"],
        bankroll=bankroll,
        unit_size=unit_size,
        lines_data=data["lines"],
        odds_data=data["odds"],
        game_logs_df=data.get("team_logs"),
        player_logs_df=data.get("player_logs"),
        team_logs_df=data.get("team_logs"),
    )

try:
    all_picks = load_picks_cloud(bankroll, unit_size, data["fetched_at"])
except Exception as _e:
    st.exception(_e)
    all_picks = []

min_rank = TIER_RANK[min_conf]
filtered = [p for p in all_picks if TIER_RANK.get(p["confidence_tier"], 0) >= min_rank]
if platforms:
    filtered = [p for p in filtered if p["pick_type"] == "game" or p.get("platform") in platforms]
if not show_game:
    filtered = [p for p in filtered if p["pick_type"] != "game"]
if not show_prop:
    filtered = [p for p in filtered if p["pick_type"] != "prop"]

best = best_props_per_player(filtered)
hi   = [p for p in best if is_high_interest(p)]


# ── Helpers ────────────────────────────────────────────────────────────────────

def timestamp_bar(fetched_at: str):
    st.markdown(
        f"<div style='background:#1a1d27;border-left:3px solid #22c55e;padding:8px 14px;"
        f"border-radius:4px;font-size:0.85rem;color:#9ca3af;margin-bottom:8px'>"
        f"🕐 Data last updated: <strong style='color:#e8eaf0'>{fetched_at} CT</strong>"
        f" &nbsp;·&nbsp; Refresh in the sidebar to update</div>",
        unsafe_allow_html=True,
    )


def style_df(df):
    def color_conf(val):
        colors = {"STRONG": "background-color:#16a34a;color:#fff;font-weight:700",
                  "HIGH":   "background-color:#22c55e;color:#000;font-weight:700",
                  "MEDIUM": "background-color:#ca8a04;color:#fff;font-weight:700",
                  "LOW":    "background-color:#374151;color:#9ca3af;font-weight:700"}
        return colors.get(val, "")
    def color_risk(val):
        colors = {"LOW":    "background-color:#16a34a;color:#fff;font-weight:700",
                  "MEDIUM": "background-color:#c2410c;color:#fff;font-weight:700",
                  "HIGH":   "background-color:#dc2626;color:#fff;font-weight:700"}
        return colors.get(val, "")
    style_fn = df.style.map if hasattr(df.style, "map") else df.style.applymap
    styled = style_fn(color_conf, subset=["Confidence"])
    style_fn2 = styled.map if hasattr(styled, "map") else styled.applymap
    return style_fn2(color_risk, subset=["Risk"])


def picks_to_df(picks, show_context=False):
    rows = []
    for p in picks:
        if p["pick_type"] == "game":
            row = {
                "Type":       p["market"],
                "Selection":  p["selection"],
                "Platform":   p["best_platform"],
                "Odds":       f"{int(p['best_odds']):+d}" if p.get("best_odds") else "—",
                "Model %":    f"{p['model_prob']:.1%}",
                "Edge":       f"{p['edge']:+.1%}",
                "EV / $100":  f"${p['ev_per_100']:+.1f}",
                "Confidence": p["confidence_tier"],
                "Risk":       p["risk_profile"],
                "Units":      f"{p.get('units', 0):.1f}u",
                "Stake ($)":  f"${p['stake_dollars']:.0f}",
                "Win ($)":    f"${p['potential_win']:.0f}",
            }
        else:
            row = {
                "Type":       "Prop",
                "Selection":  f"{p['player_name']} {p['stat_type']} {p['direction']} {p['line']}",
                "Platform":   p["platform"],
                "Odds":       "—",
                "Model %":    f"{p['model_prob']:.1%}",
                "Edge":       f"{p['edge']:+.1%}",
                "EV / $100":  f"${p['ev_per_100']:+.1f}",
                "Confidence": p["confidence_tier"],
                "Risk":       p["risk_profile"],
                "Units":      f"{p.get('units', 0):.1f}u",
                "Stake ($)":  f"${p['stake_dollars']:.0f}",
                "Win ($)":    f"${p['potential_win']:.0f}",
            }
        if show_context and p["pick_type"] == "prop":
            row["Season"] = f"{p.get('season_rate', 0):.2f}" if p.get("season_rate") is not None else "—"
            row["Recent"] = f"{p.get('recent_rate', 0):.2f}" if p.get("recent_rate") is not None else "—"
        rows.append(row)
    if not rows:
        return None
    return pd.DataFrame(rows)


# ── Tabs ───────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔥 Top Picks",
    "🏀 Game Predictions",
    "🎯 Player Props",
    "📊 Platform Comparison",
    "💰 Bankroll Tracker",
])

# ── Tab 1: Top Picks ───────────────────────────────────────────────────────────

with tab1:
    timestamp_bar(data["fetched_at"])
    st.header("🔥 Top Picks")
    st.caption("Game picks (ML/Spread/Totals) + high-interest player props, ranked by EV.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total",      len(hi))
    c2.metric("Game Picks", sum(1 for p in hi if p["pick_type"] == "game"))
    c3.metric("Props",      sum(1 for p in hi if p["pick_type"] == "prop"))
    c4.metric("STRONG",     sum(1 for p in hi if p["confidence_tier"] == "STRONG"))

    st.divider()
    show_ctx = st.toggle("Show season/recent context", value=False)
    df = picks_to_df(hi[:75], show_context=show_ctx)
    if df is not None:
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
    else:
        st.info("No picks match current filters.")

# ── Tab 2: Game Predictions ────────────────────────────────────────────────────

with tab2:
    timestamp_bar(data["fetched_at"])
    st.header("🏀 Game Predictions")

    game_picks = [p for p in best if p["pick_type"] == "game"]
    games_list = data["games"]

    if not games_list:
        st.info("No games found for today.")
    else:
        for g in games_list:
            home, away = g["home_team"], g["away_team"]
            g_picks = [p for p in game_picks if p["home_team"] == home]

            with st.expander(f"**{away}** @ **{home}**", expanded=len(g_picks) > 0):
                ml = [p for p in g_picks if p["market"] == "Moneyline"]
                sp = [p for p in g_picks if p["market"] == "Spread"]
                to = [p for p in g_picks if p["market"] == "Totals"]

                if g_picks:
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown("**Moneyline**")
                        for p in ml:
                            st.markdown(f"- {p['selection']} `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']} `{int(p['best_odds']):+d}`")
                    with c2:
                        st.markdown("**Spread**")
                        for p in sp:
                            st.markdown(f"- {p['selection']} `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                    with c3:
                        st.markdown("**Totals**")
                        for p in to:
                            st.markdown(f"- {p['selection']} `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                else:
                    st.caption("No +EV picks for this game.")

# ── Tab 3: Player Props ────────────────────────────────────────────────────────

with tab3:
    timestamp_bar(data["fetched_at"])
    st.header("🎯 Player Props")

    prop_picks = [p for p in hi if p["pick_type"] == "prop"]
    stat_types = sorted(set(p["stat_type"] for p in prop_picks))
    sel_stats  = st.multiselect("Filter by stat:", stat_types, default=[], key="prop_stats")
    filtered_props = [p for p in prop_picks if not sel_stats or p["stat_type"] in sel_stats]

    df = picks_to_df(filtered_props[:75], show_context=True)
    if df is not None:
        df = df.drop(columns=["Type", "Odds"], errors="ignore")
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
    else:
        st.info("No prop picks match current filters.")

# ── Tab 4: Platform Comparison ─────────────────────────────────────────────────

with tab4:
    timestamp_bar(data["fetched_at"])
    st.header("📊 Platform Comparison")
    st.caption("Same player-prop across PrizePicks and Underdog side-by-side.")

    search     = st.text_input("Search player…", "")
    prop_all   = [p for p in filtered if p["pick_type"] == "prop"]
    grouped    = defaultdict(list)
    for p in prop_all:
        grouped[(p["player_name"], p["stat_type"])].append(p)

    items = [(k, v) for k, v in grouped.items() if len(v) > 1]
    if search:
        items = [i for i in items if search.lower() in i[0][0].lower()]

    if not items:
        st.info("No cross-platform comparisons available.")
    else:
        for (player, stat), picks_list in items[:40]:
            picks_list.sort(key=lambda x: x["edge"], reverse=True)
            with st.expander(f"**{player}** — {stat}", expanded=False):
                rows = [{
                    "Platform": p["platform"], "Line": p["line"],
                    "Pick": p["direction"], "Model %": f"{p['model_prob']:.1%}",
                    "Edge": f"{p['edge']:+.1%}", "EV/$100": f"${p['ev_per_100']:+.1f}",
                    "Confidence": p["confidence_tier"], "Risk": p["risk_profile"],
                } for p in picks_list]
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ── Tab 5: Bankroll Tracker ────────────────────────────────────────────────────

with tab5:
    st.header("💰 Bankroll Tracker")
    c1, c2 = st.columns(2)

    with c1:
        st.metric("Bankroll", f"${bankroll:,.2f}")
        st.caption(f"1 unit = ${unit_size:.0f}")
        st.divider()
        st.subheader("Recommended Stakes")
        rows = [{
            "Selection":  p["selection"],
            "Type":       p["pick_type"].title(),
            "Confidence": p["confidence_tier"],
            "Units":      f"{p.get('units', 0):.1f}u",
            "Stake ($)":  f"${p['stake_dollars']:.2f}",
            "Win ($)":    f"${p['potential_win']:.2f}",
            "R/R":        f"{p['risk_reward_ratio']:.1f}x",
        } for p in hi[:20]]
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with c2:
        st.subheader("PrizePicks Slip Builder")
        st.caption("Select 2–6 prop picks to calculate slip EV.")
        prop_hi   = [p for p in hi if p["pick_type"] == "prop"]
        slip_opts = [p["selection"] for p in prop_hi[:30]]
        selected  = st.multiselect("Select picks:", slip_opts, max_selections=6)
        if len(selected) >= 2:
            probs  = [p["model_prob"] for sel in selected for p in prop_hi if p["selection"] == sel]
            result = ev_slip(probs, "prizepicks", len(selected))
            if result:
                st.metric("Slip Size",   f"{len(selected)}-pick Power Play")
                st.metric("Multiplier",  f"{result['multiplier']}x")
                st.metric("P(all hit)",  f"{result['p_all_hit']:.1%}")
                ev_val = result["ev_per_100"]
                st.metric("EV per $100", f"${ev_val:+.2f}",
                          delta="Positive" if ev_val > 0 else "Negative",
                          delta_color="normal" if ev_val > 0 else "inverse")
        elif len(selected) == 1:
            st.info("Select at least 2 picks.")
        else:
            st.info("Select picks above to build a slip.")
