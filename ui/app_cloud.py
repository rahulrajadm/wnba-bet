"""
WNBA Bet — Streamlit Community Cloud version.
Fetches all data in-memory (no SQLite). Refresh is passcode-gated.
"""
import sys, os, shutil, pathlib

# The app has been segfaulting on Streamlit Cloud (Linux-only; identical
# versions render fine on macOS). faulthandler prints every thread's Python
# stack to stderr on SIGSEGV, which surfaces in the Cloud logs — turning a
# bare "Segmentation fault" into an exact crash location.
import faulthandler
try:
    faulthandler.enable(file=sys.stderr, all_threads=True)
except Exception:
    pass

# Cap native thread pools BEFORE numpy/scipy/xgboost load their runtimes.
# Streamlit Cloud's container advertises many CPUs but allots little memory;
# OpenMP/BLAS spawning a thread per visible CPU from Streamlit's short-lived
# script threads segfaults the process (no traceback, just SIGSEGV in the logs).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# The post-refresh segfault (see faulthandler stack, 2026-07-10) died inside
# libarrow's jemalloc allocator (AllocateResizableBuffer via NdarrayToArrow)
# while st.dataframe serialized the styled picks table. Arrow's Linux wheels
# default to jemalloc, macOS wheels to mimalloc — which is why the crash never
# reproduced locally on identical versions. ARROW_DEFAULT_MEMORY_POOL is read
# at pyarrow import, which streamlit has already done by the time this script
# runs, so switch the default pool at runtime instead.
try:
    import pyarrow as _pa
    _pa.set_memory_pool(_pa.system_memory_pool())
    del _pa
except Exception:
    pass

# Clear stale .pyc bytecode on first startup so Streamlit Cloud always
# runs the current source — without this, cached .pyc files survive
# deployments and execute old code even after the .py files are updated.
if "picks.engine" not in sys.modules:
    for _cache in pathlib.Path(__file__).resolve().parent.parent.rglob("__pycache__"):
        shutil.rmtree(_cache, ignore_errors=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# Streamlit Cloud can hot-swap updated source into a RUNNING Python process,
# which leaves stale project modules in sys.modules — the sibling of the .pyc
# problem above, and one the .pyc sweep can't fix. If the already-loaded
# shared code predates what this file needs, purge it all and re-import fresh.
_REQUIRED_SCHEMA = 3
import analysis.explain as _explain_probe
if getattr(_explain_probe, "SCHEMA_VERSION", 0) < _REQUIRED_SCHEMA:
    for _name in [n for n in list(sys.modules)
                  if n.split(".")[0] in ("pipeline", "picks", "models", "analysis", "utils")]:
        sys.modules.pop(_name, None)
del _explain_probe

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
from pipeline import cloud_data as _cloud_data
from pipeline.cloud_data import fetch_team_game_logs, fetch_player_game_logs
from pipeline.injuries import fetch_injury_flags
from picks.engine import build_picks, best_props_per_player, MODEL_WEIGHT
from models.game import predict_game
from analysis.confidence import TIER_COLORS, TIER_RANK
from analysis.risk import RISK_COLORS
from analysis.ev import ev_slip
from analysis.explain import answer_question
from scipy.stats import norm as _norm

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

            # Store date in ET so evening games (tip-off ~8 PM ET = midnight UTC)
            # aren't mis-labelled as the next calendar day.
            game_date = game_dt.astimezone(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

            if game_id not in seen_games:
                seen_games[game_id] = {
                    "game_id": game_id, "date": game_date,
                    "home_team": home_team, "away_team": away_team,
                    "home_team_id": "", "away_team_id": "",
                    "game_time": commence, "season": str(date.today().year),
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


def load_all_data():
    # Odds API: single call for schedule + all markets (3 credits total, not 4)
    odds, games = fetch_odds_and_schedule()

    # Injury flags: fetch now that we know which teams are playing today
    team_names = list({g["home_team"] for g in games} | {g["away_team"] for g in games})
    try:
        injuries = fetch_injury_flags(team_names)
    except Exception:
        injuries = {}

    # Everything else in parallel (no Odds API calls)
    with ThreadPoolExecutor(max_workers=4) as ex:
        f_pp          = ex.submit(pp_fetch)
        f_ud          = ex.submit(ud_fetch)
        f_team_logs   = ex.submit(fetch_team_game_logs)
        f_player_logs = ex.submit(fetch_player_game_logs)
        try:
            pp_lines = f_pp.result(timeout=20)
        except Exception:
            pp_lines = []
        try:
            ud_lines = f_ud.result(timeout=20)
        except Exception:
            ud_lines = []
        try:
            team_logs = f_team_logs.result(timeout=45)
            if _cloud_data._last_fetch_failures:
                st.warning(f"ESPN: {_cloud_data._last_fetch_failures} team box-score fetch(es) failed — team stats may be incomplete.")
        except Exception:
            team_logs = pd.DataFrame()
        try:
            player_logs = f_player_logs.result(timeout=120)
            if _cloud_data._last_fetch_failures:
                st.warning(f"ESPN: {_cloud_data._last_fetch_failures} player box-score fetch(es) failed — prop stats may be incomplete.")
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
        "injuries":    injuries,
        "fetched_at":  datetime.now(ZoneInfo("America/Chicago")).strftime("%b %d %Y, %I:%M %p"),
    }


# ── Persistent data store (survives page refresh, shared across sessions) ──────
# st.cache_resource holds its value as long as the server process is alive —
# unlike st.session_state which is cleared on every browser refresh.

@st.cache_resource
def _data_store():
    return {"payload": None}

store = _data_store()

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
                with st.spinner("Fetching data…"):
                    fresh = load_all_data()
                if fresh["games"]:
                    store["payload"] = fresh
                    st.rerun()
                else:
                    st.warning("No upcoming games found — Odds API has no lines posted yet. "
                               "Keeping existing data. Try again later.")
            else:
                st.error("Invalid passcode")


# ── Gate: first-ever load before any refresh has been triggered ────────────────

data = store["payload"]

if data is None:
    st.info("No data loaded. Enter your passcode in the sidebar and press **Refresh All Data**.")
    st.stop()

# ── Abbreviation map + date filter data (needed before sidebar renders) ────────
from pipeline.prizepicks import _TEAM_ABBR as _PP_ABBR

_team_date_map: dict[str, str] = {}
for _g in data.get("games", []):
    _d = _g.get("date", "")
    if _d:
        _team_date_map[_g["home_team"]] = _d
        _team_date_map[_g["away_team"]] = _d
        for _abbr, _full in _PP_ABBR.items():
            if _full in (_g["home_team"], _g["away_team"]):
                _team_date_map[_abbr] = _d
_available_dates = sorted(set(_team_date_map.values()))

with st.sidebar:
    st.caption(f"🕐 Last updated: **{data['fetched_at']}** CT")
    st.caption("Data: PrizePicks · Underdog · The Odds API")
    tl = data.get("team_logs")
    pl = data.get("player_logs")
    pl_rows = len(pl) if pl is not None and not pl.empty else 0
    season_detail = "  ·  ".join(
        f"{yr}: {cnt}g" for yr, cnt in sorted(_cloud_data._season_game_counts.items())
    )
    st.caption(
        f"Games: {len(data['games'])} · "
        f"Lines: {len(data['lines'])} · "
        f"Team logs: {len(tl) if tl is not None and not tl.empty else 0} rows · "
        f"Player logs: {pl_rows} rows"
        + (f" ({season_detail})" if season_detail else "")
    )
    st.divider()
    min_conf   = st.selectbox("Min Confidence", ["LOW", "MEDIUM", "HIGH", "STRONG"], index=1)
    platforms  = st.multiselect("Prop Platforms", ["prizepicks", "underdog"], default=["prizepicks", "underdog"])
    line_types = st.multiselect(
        "Line Types",
        ["standard", "goblin", "demon"],
        default=["standard", "goblin", "demon"],
        help="goblin = lowered line (easy More, lower payout). demon = elevated line (hard More). standard = normal line (More or Less).",
    )
    if len(_available_dates) > 1:
        sel_dates = st.multiselect(
            "Game Date",
            options=_available_dates,
            default=_available_dates,
            format_func=lambda d: datetime.strptime(d, "%Y-%m-%d").strftime("%a, %b %-d"),
        )
    else:
        sel_dates = list(_available_dates)
    show_game = st.toggle("Show game picks", value=True)
    show_prop = st.toggle("Show player props", value=True)


# ── Build picks ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_picks_cloud(bankroll, unit_size, cache_key):
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

# Log picks for later grading (analysis/tracking.py). Without a graded record
# there's no way to tell whether model changes actually help.
try:
    from analysis.tracking import save_picks
    save_picks(all_picks)
except Exception:
    pass  # ephemeral storage on Streamlit Cloud — logging is best-effort there

min_rank = TIER_RANK[min_conf]
filtered = [p for p in all_picks if TIER_RANK.get(p["confidence_tier"], 0) >= min_rank]
if platforms:
    filtered = [p for p in filtered if p["pick_type"] == "game" or p.get("platform") in platforms]
if line_types and len(line_types) < 3:
    filtered = [p for p in filtered if p["pick_type"] == "game" or p.get("odds_type", "standard") in line_types]
if not show_game:
    filtered = [p for p in filtered if p["pick_type"] != "game"]
if not show_prop:
    filtered = [p for p in filtered if p["pick_type"] != "prop"]
if sel_dates and len(sel_dates) < len(_available_dates):
    def _get_pick_date(p):
        if p["pick_type"] == "game":
            return _team_date_map.get(p.get("home_team", ""), "")
        pt = p.get("player_team", "")
        return _team_date_map.get(pt) or _team_date_map.get(_PP_ABBR.get(pt, ""), "")
    filtered = [p for p in filtered if _get_pick_date(p) in sel_dates]

best = best_props_per_player(filtered)
hi   = best


# ── Game-time lookup: team name → {date, time, opponent label} ────────────────
_team_game_map: dict[str, dict] = {}
for _g in data.get("games", []):
    try:
        _ct   = datetime.fromisoformat(_g["game_time"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/Chicago"))
        _date = _ct.strftime("%b %-d")
        _time = _ct.strftime("%-I:%M %p")
    except Exception:
        _date = _g.get("date", "—")
        _time = "—"
    _h, _a = _g["home_team"], _g["away_team"]
    # "sort" = ISO UTC tip-off, used to order tables chronologically — the
    # display strings ("Jul 5", "9:00 PM CT") don't sort correctly as text.
    _sort_key = _g.get("game_time", "") or "~"
    _team_game_map[_h] = {"date": _date, "time": _time, "opp": f"vs {_a}", "sort": _sort_key}
    _team_game_map[_a] = {"date": _date, "time": _time, "opp": f"@ {_h}", "sort": _sort_key}

# Index by PrizePicks abbreviation so stale cached player_team values resolve.
for _abbr, _full in _PP_ABBR.items():
    if _full in _team_game_map:
        _team_game_map.setdefault(_abbr, _team_game_map[_full])

# Lowercase fallback for any remaining mismatches
_team_game_map_lc: dict[str, dict] = {k.lower(): v for k, v in _team_game_map.items()}


# ── Helpers ────────────────────────────────────────────────────────────────────

def timestamp_bar(fetched_at: str):
    st.markdown(
        f"<div style='background:#1a1d27;border-left:3px solid #22c55e;padding:8px 14px;"
        f"border-radius:4px;font-size:0.85rem;color:#9ca3af;margin-bottom:8px'>"
        f"🕐 Data last updated: <strong style='color:#e8eaf0'>{fetched_at} CT</strong>"
        f" &nbsp;·&nbsp; Refresh in the sidebar to update</div>",
        unsafe_allow_html=True,
    )


def metric_chips(items: list[tuple[str, str]]):
    """Compact inline stat chips — unlike st.metric they don't stack into a
    full screen of giant numbers on mobile."""
    chips = "".join(
        f"<div style='background:#1a1d27;border:1px solid #2a2e3d;border-radius:8px;"
        f"padding:6px 14px;display:flex;gap:8px;align-items:baseline'>"
        f"<span style='color:#9ca3af;font-size:0.8rem'>{label}</span>"
        f"<strong style='color:#e8eaf0;font-size:1.15rem'>{value}</strong></div>"
        for label, value in items
    )
    st.markdown(
        f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin:6px 0 10px 0'>{chips}</div>",
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
    def color_edge(val):
        try:
            v = float(str(val).replace("%", "").replace("+", ""))
        except ValueError:
            return ""
        if v >= 15:
            return "color:#4ade80;font-weight:700"
        if v >= 8:
            return "color:#86efac;font-weight:600"
        return "color:#bbf7d0"
    styled = df.style
    style_fn = styled.map if hasattr(styled, "map") else styled.applymap
    for col, fn in (("Conf", color_conf), ("Risk", color_risk), ("Edge", color_edge)):
        if col in df.columns:
            styled = style_fn(fn, subset=[col])
            style_fn = styled.map if hasattr(styled, "map") else styled.applymap
    return styled


def _prop_game_info(p: dict) -> dict:
    _pt      = p.get("player_team", "")
    _pt_full = _PP_ABBR.get(_pt, _pt)   # expand abbrev → full name
    return (_team_game_map.get(_pt)
            or _team_game_map.get(_pt_full)
            or _team_game_map_lc.get(_pt_full.lower(), {}))


def picks_to_df(picks, show_context=False):
    rows = []
    for p in picks:
        if p["pick_type"] == "game":
            _gi = _team_game_map.get(p.get("home_team", "")) \
                  or _team_game_map_lc.get(p.get("home_team", "").lower(), {})
            row = {
                "_sort":     _gi.get("sort", "~"),
                "Game":      f"{p.get('away_team','')} @ {p.get('home_team','')}",
                "Tip":       f"{_gi.get('date', '—')} {_gi.get('time', '—')}",
                "Selection": f"{p['market']}: {p['selection']}",
                "Platform":  p["best_platform"],
                "Odds":      f"{int(p['best_odds']):+d}" if p.get("best_odds") else "—",
                "Model %":   f"{p['model_prob']:.1%}",
                "Edge":      f"{p['edge']:+.1%}",
                "EV/$100":   f"${p['ev_per_100']:+.1f}",
                "Conf":      p["confidence_tier"],
                "Risk":      p["risk_profile"],
                "Stake":     f"${p['stake_dollars']:.0f}",
                "Win":       f"${p['potential_win']:.0f}",
            }
        else:
            _gi = _prop_game_info(p)
            _ot = p.get("odds_type", "standard")
            sel = f"{p['player_name']} {p['stat_type']} {p['direction']} {p['line']}"
            if _ot in ("goblin", "demon"):
                sel += {"goblin": " 🐸", "demon": " 😈"}[_ot]
            row = {
                "_sort":     _gi.get("sort", "~"),
                "Game":      _gi.get("opp", "—"),
                "Tip":       f"{_gi.get('date', '—')} {_gi.get('time', '—')}",
                "Selection": sel,
                "Platform":  p["platform"],
                "Odds":      None,   # props have no American odds — dropped below
                "Model %":   f"{p['model_prob']:.1%}",
                "Edge":      f"{p['edge']:+.1%}",
                "EV/$100":   f"${p['ev_per_100']:+.1f}",
                "Conf":      p["confidence_tier"],
                "Risk":      p["risk_profile"],
                "Stake":     f"${p['stake_dollars']:.0f}",
                "Win":       f"${p['potential_win']:.0f}",
            }
        if show_context and p["pick_type"] == "prop":
            row["Season"] = f"{p.get('season_rate', 0):.2f}" if p.get("season_rate") is not None else "—"
            row["Recent"] = f"{p.get('recent_rate', 0):.2f}" if p.get("recent_rate") is not None else "—"
        rows.append(row)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("_sort", kind="stable").drop(columns=["_sort"])
    # Odds only exists for game picks; drop the column when it would be all "—"
    if df["Odds"].isna().all():
        df = df.drop(columns=["Odds"])
    else:
        df["Odds"] = df["Odds"].fillna("—")
    return df


# ── Tabs ───────────────────────────────────────────────────────────────────────

timestamp_bar(data["fetched_at"])

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔥 Top Picks",
    "🏀 Game Predictions",
    "🎯 Player Props",
    "📊 Platform Comparison",
    "💰 Bankroll Tracker",
    "💬 Ask Why",
])

# ── Tab 1: Top Picks ───────────────────────────────────────────────────────────

with tab1:
    st.caption("Game picks (ML/Spread/Totals) + high-interest player props, ranked by EV. Tip times in CT.")

    metric_chips([
        ("Total",      str(len(hi))),
        ("Game Picks", str(sum(1 for p in hi if p["pick_type"] == "game"))),
        ("Prop Picks", str(sum(1 for p in hi if p["pick_type"] == "prop"))),
        ("STRONG",     str(sum(1 for p in hi if p["confidence_tier"] == "STRONG"))),
    ])

    show_ctx = st.toggle("Show season/recent context", value=False)
    df = picks_to_df(hi[:75], show_context=show_ctx)
    if df is not None:
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
    elif not data["games"]:
        st.warning("No upcoming games found. The Odds API hasn't posted lines yet — try refreshing later in the day.")
    elif not all_picks:
        st.warning("Games were found but no picks cleared the edge threshold.")
    else:
        st.warning("Picks exist but none match the current confidence or platform filters. Try lowering Min Confidence.")

# ── Tab 2: Game Predictions ────────────────────────────────────────────────────

with tab2:
    game_picks = [p for p in best if p["pick_type"] == "game"]
    games_list = data["games"]

    # Model's view for every game, even where there's no bet. Shown with an
    # explicit "pass" so an empty market reads as a decision, not missing data.
    @st.cache_data(show_spinner=False)
    def _model_views(cache_key):
        odds_df = pd.DataFrame(data["odds"]) if data.get("odds") else pd.DataFrame()
        views = {}
        for _g in data["games"]:
            _h, _a = _g["home_team"], _g["away_team"]
            try:
                _pred = predict_game(_h, _a, game_logs_df=data.get("team_logs"))
            except Exception:
                _pred = None
            if not _pred:
                continue
            sl = tl = None
            if not odds_df.empty:
                _go = odds_df[odds_df["home_team"].str.lower() == _h.lower()]
                _s  = _go[_go["market"] == "spread"]["home_spread"].dropna()
                _t  = _go[_go["market"] == "totals"]["total_line"].dropna()
                sl  = float(_s.iloc[0]) if len(_s) else None
                tl  = float(_t.iloc[0]) if len(_t) else None
            # Same anchoring as picks/engine.py so the displayed view matches
            # the probabilities the pick filter actually used.
            diff  = _pred["pred_diff"]
            total = _pred["pred_total"]
            if sl is not None:
                diff = MODEL_WEIGHT * diff + (1 - MODEL_WEIGHT) * (-sl)
            if tl is not None:
                total = MODEL_WEIGHT * total + (1 - MODEL_WEIGHT) * tl
            views[_h] = {
                "diff": diff, "total": total, "spread_line": sl, "total_line": tl,
                "home_p": float(_norm.cdf(diff / _pred["spread_std"])),
                "spread_std": _pred["spread_std"], "totals_std": _pred["totals_std"],
                "raw_diff": _pred["pred_diff"], "raw_total": _pred["pred_total"],
            }
        return views

    views = _model_views(data["fetched_at"])

    if not games_list:
        st.warning("No upcoming games found. The Odds API hasn't posted lines yet — try refreshing later in the day.")
    else:
        st.caption(
            "**pass** = the model's probability doesn't beat the market price by enough to overcome the vig "
            "(spread/totals at -110 need ~56%+). A confident pass is still a pass — the payout already reflects "
            "the market's matching confidence."
        )
        for g in games_list:
            home, away = g["home_team"], g["away_team"]
            g_picks = [p for p in game_picks if p["home_team"] == home]

            # Show tip-off time in CT so users can verify games are upcoming
            try:
                gt = datetime.fromisoformat(g["game_time"].replace("Z", "+00:00"))
                gt_ct = gt.astimezone(ZoneInfo("America/Chicago"))
                tip_label = gt_ct.strftime("%a %b %-d · %-I:%M %p CT")
            except Exception:
                tip_label = ""

            injuries    = data.get("injuries", {})
            home_flags  = injuries.get(home, [])
            away_flags  = injuries.get(away, [])
            has_injury  = bool(home_flags or away_flags)

            # Append injury indicator to expander header
            inj_badge = "  ·  ⚠️ injury alert" if has_injury else ""
            header = f"**{away}** @ **{home}**" + (f"  ·  {tip_label}" if tip_label else "") + inj_badge
            with st.expander(header, expanded=len(g_picks) > 0):
                # Injury alerts
                if has_injury:
                    parts = []
                    for flag in away_flags:
                        parts.append(f"{flag['name']} ({away}) — {flag['status']}")
                    for flag in home_flags:
                        parts.append(f"{flag['name']} ({home}) — {flag['status']}")
                    st.warning("⚠️ " + "  ·  ".join(parts))

                ml = [p for p in g_picks if p["market"] == "Moneyline"]
                sp = [p for p in g_picks if p["market"] == "Spread"]
                to = [p for p in g_picks if p["market"] == "Totals"]
                v  = views.get(home)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.markdown("**Moneyline**")
                    for p in ml:
                        st.markdown(f"- {p['selection']} `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']} `{int(p['best_odds']):+d}`")
                    if not ml and v:
                        fav, fp = (home, v["home_p"]) if v["home_p"] >= 0.5 else (away, 1 - v["home_p"])
                        st.caption(f"Model: {fav} `{fp:.0%}` — market agrees → **pass**")
                with c2:
                    st.markdown("**Spread**")
                    for p in sp:
                        st.markdown(f"- {p['selection']} `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                    if not sp and v:
                        fav_t, fav_m = (home, v["diff"]) if v["diff"] >= 0 else (away, -v["diff"])
                        if v["spread_line"] is not None:
                            p_cov = float(_norm.cdf((v["diff"] + v["spread_line"]) / v["spread_std"]))
                            side_lbl, pc = ((f"{home} {v['spread_line']:+.1f}", p_cov) if p_cov >= 0.5
                                            else (f"{away} {-v['spread_line']:+.1f}", 1 - p_cov))
                            st.caption(f"Model: {fav_t} by `{fav_m:.1f}` · {side_lbl} covers `{pc:.0%}` → **pass**")
                        else:
                            st.caption(f"Model: {fav_t} by `{fav_m:.1f}` — no line posted")
                with c3:
                    st.markdown("**Totals**")
                    for p in to:
                        st.markdown(f"- {p['selection']} `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                    if not to and v:
                        if v["total_line"] is not None:
                            p_over = float(1 - _norm.cdf(v["total_line"], v["total"], v["totals_std"]))
                            ou_lbl, po = ("Over", p_over) if p_over >= 0.5 else ("Under", 1 - p_over)
                            st.caption(f"Model: `{v['total']:.1f}` vs line `{v['total_line']:.1f}` · "
                                       f"{ou_lbl} `{po:.0%}` → **pass**")
                        else:
                            st.caption(f"Model: `{v['total']:.1f}` total — no line posted")
                if not g_picks and not v:
                    st.caption("Not enough data to model this game yet.")

# ── Tab 3: Player Props ────────────────────────────────────────────────────────

with tab3:
    prop_picks = [p for p in hi if p["pick_type"] == "prop"]
    stat_types = sorted(set(p["stat_type"] for p in prop_picks))
    sel_stats  = st.multiselect("Filter by stat:", stat_types, default=[], key="prop_stats")
    filtered_props = [p for p in prop_picks if not sel_stats or p["stat_type"] in sel_stats]

    df = picks_to_df(filtered_props[:75], show_context=True)
    if df is not None:
        # Analysis view — staking columns live in Top Picks / Bankroll, dropping
        # them here keeps the Season/Recent context on screen.
        df = df.drop(columns=["Tip", "EV/$100", "Stake", "Win"], errors="ignore")
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
        st.caption("**Season/Recent** = the player's per-game average for this stat (full season vs last 5).")
    elif not data["games"]:
        st.warning("No upcoming games found — no opponent context to generate props.")
    elif not [p for p in all_picks if p["pick_type"] == "prop"]:
        st.warning("No prop picks cleared the edge threshold.")
    else:
        st.warning("Props exist but none match current filters. Try lowering Min Confidence or changing platforms.")

# ── Tab 4: Platform Comparison ─────────────────────────────────────────────────

with tab4:
    st.caption("Same player-prop across PrizePicks and Underdog side-by-side.")

    search     = st.text_input("Search player…", "")
    prop_all   = [p for p in filtered if p["pick_type"] == "prop"]
    grouped    = defaultdict(list)
    for p in prop_all:
        # Goblin/demon lines are deliberately shifted; comparing them against the
        # other platform's standard line isn't apples-to-apples.
        if p.get("odds_type", "standard") != "standard":
            continue
        grouped[(p["player_name"], p["stat_type"])].append(p)

    items = []
    for key, plist in grouped.items():
        if len({p["platform"] for p in plist}) < 2:
            continue
        # Same stat but far-apart lines = different market (standard vs alternate),
        # not a real cross-platform discrepancy.
        lines_vals = [p["line"] for p in plist]
        if max(lines_vals) - min(lines_vals) > 1.5:
            continue
        items.append((key, plist))
    if search:
        items = [i for i in items if search.lower() in i[0][0].lower()]

    if not items:
        st.info("No cross-platform overlaps at comparable lines today.")
    else:
        for (player, stat), picks_list in items[:40]:
            picks_list.sort(key=lambda x: x["edge"], reverse=True)
            top   = picks_list[0]
            lines_lbl = " vs ".join(
                f"{p['platform'][:2].upper()} {p['line']}" for p in picks_list)
            title = (f"**{player}** — {stat} · {lines_lbl} · "
                     f"best: {top['direction']} {top['line']} @ {top['platform']} `{top['edge']:+.1%}`")
            with st.expander(title, expanded=False):
                rows = [{
                    "Platform": p["platform"], "Line": p["line"],
                    "Pick": p["direction"], "Model %": f"{p['model_prob']:.1%}",
                    "Edge": f"{p['edge']:+.1%}", "EV/$100": f"${p['ev_per_100']:+.1f}",
                    "Conf": p["confidence_tier"], "Risk": p["risk_profile"],
                } for p in picks_list]
                st.dataframe(style_df(pd.DataFrame(rows)), use_container_width=True, hide_index=True)

# ── Tab 5: Bankroll Tracker ────────────────────────────────────────────────────

with tab5:
    metric_chips([("Bankroll", f"${bankroll:,.0f}"), ("1 Unit", f"${unit_size:.0f}")])

    st.subheader("Recommended Stakes")
    rows = [{
        "Selection":  p["selection"],
        "Type":       p["pick_type"].title(),
        "Conf":       p["confidence_tier"],
        "Units":      f"{p.get('units', 0):.1f}u",
        "Stake":      f"${p['stake_dollars']:.2f}",
        "Win":        f"${p['potential_win']:.2f}",
        "R/R":        f"{p['risk_reward_ratio']:.1f}x",
    } for p in hi[:20]]
    if rows:
        st.dataframe(style_df(pd.DataFrame(rows)), use_container_width=True, hide_index=True)
    else:
        st.info("No picks to stake today.")

    st.divider()
    c1, c2 = st.columns(2)
    with c2:
        st.subheader("Notes")
        st.caption(
            "Stakes are quarter-Kelly per leg sized as a standalone 3× bet — a real "
            "multi-leg slip needs every leg to hit, so treat prop stakes as aggressive "
            "upper bounds. 🐸 goblin edges are somewhat overstated (goblins lower the "
            "slip multiplier); 😈 demon edges understated. Graded results tracking runs "
            "on the local dashboard, which has persistent storage."
        )
    with c1:
        st.subheader("PrizePicks Slip Builder")
        st.caption("Select 2–6 prop picks to calculate slip EV.")
        prop_hi   = [p for p in hi if p["pick_type"] == "prop"]
        # One entry per selection string — the same player/stat/line can appear
        # on both platforms, and duplicates would double-count a leg's prob.
        slip_map: dict[str, float] = {}
        for p in prop_hi[:30]:
            slip_map.setdefault(p["selection"], p["model_prob"])
        selected  = st.multiselect("Select picks:", list(slip_map), max_selections=6)
        if len(selected) >= 2:
            probs  = [slip_map[sel] for sel in selected]
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

# ── Tab 6: Ask Why ─────────────────────────────────────────────────────────────

with tab6:
    st.caption(
        "Ask how the model reached a number — every step of the actual arithmetic, no AI involved. "
        "Try: *why does Breanna Stewart Rebs+Asts More 8.5 have 85%?*, then follow up with "
        "*what if she plays 25 minutes?*, *what about a line of 10.5?*, *what about Less?* — "
        "or ask *safest points pick tonight* / *compare A'ja Wilson across platforms*."
    )

    if "ask_history" not in st.session_state:
        st.session_state.ask_history = []

    for q_msg, a_msg in st.session_state.ask_history:
        with st.chat_message("user"):
            st.markdown(q_msg)
        with st.chat_message("assistant"):
            st.markdown(a_msg)

    question = st.chat_input("Why does … have …%?")
    if question:
        # Match against ALL picks (pre-filter) so sidebar filters never hide an answer
        answer = answer_question(question, all_picks, data["games"], views, st.session_state)
        st.session_state.ask_history.append((question, answer))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
