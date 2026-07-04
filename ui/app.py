"""
WNBA Bet — Streamlit Dashboard (local, uses SQLite)
Tabs: 🔥 Game Picks | 🏀 Game Predictions | 🎯 Player Props | 📊 Platform Comparison | 💰 Bankroll
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
from scipy.stats import norm
from zoneinfo import ZoneInfo
from datetime import datetime, timezone

from utils.db import get_conn, get_meta
from pipeline.schedule import get_today_games, load_saved_games
from pipeline.prizepicks import get_prizepicks_lines
from pipeline.underdog import get_underdog_lines
from pipeline.odds import get_all_odds
from models.game import predict_game
from picks.engine import build_picks, best_props_per_player, is_high_interest, MODEL_WEIGHT
from analysis.confidence import TIER_COLORS, TIER_RANK
from analysis.risk import RISK_COLORS
from analysis.ev import ev_slip
from analysis.explain import answer_question

st.set_page_config(page_title="WNBA Bet", page_icon="🏀", layout="wide", initial_sidebar_state="expanded")

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🏀 WNBA Bet")
    st.caption("AI-powered WNBA betting decisions")
    st.divider()

    bankroll  = st.number_input("My Bankroll ($)", min_value=10.0, value=500.0, step=10.0)
    unit_size = st.number_input("1 Unit = ($)",    min_value=1.0,  value=10.0,  step=1.0)

    st.divider()
    if st.button("🔄 Refresh All Data", use_container_width=True):
        # Each refresh costs Odds API credits (schedule + 3 markets); block
        # accidental double-clicks instead of silently re-spending them.
        last = st.session_state.get("last_refresh", 0.0)
        if time.time() - last < 60:
            st.warning("Refreshed less than a minute ago — using existing data.")
        else:
            st.session_state["last_refresh"] = time.time()
            with st.spinner("Refreshing…"):
                get_today_games()
                get_prizepicks_lines()
                get_underdog_lines()
                try:
                    get_all_odds()
                except Exception:
                    pass
            st.cache_data.clear()
            st.success("Data refreshed!")
            st.rerun()
    _credits = get_meta("odds_api_remaining")
    if _credits is not None:
        st.caption(f"Odds API credits remaining: **{_credits}**")

    st.divider()
    min_conf  = st.selectbox("Min Confidence", ["LOW", "MEDIUM", "HIGH", "STRONG"], index=1)
    platforms = st.multiselect("Prop Platforms", ["prizepicks", "underdog"], default=["prizepicks", "underdog"])
    show_game = st.toggle("Show game picks", value=True)
    show_prop = st.toggle("Show player props", value=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(bankroll, unit_size):
    # Read the saved schedule — get_today_games() burns an Odds API credit per
    # call, which the cache would re-spend every 5 minutes the app stays open.
    # Live fetches happen only via the Refresh button / start.sh.
    games = load_saved_games() or get_today_games()
    picks = build_picks(games, bankroll=bankroll, unit_size=unit_size)
    return games, picks

with st.spinner("Loading picks…"):
    games, all_picks = load_data(bankroll, unit_size)

# Log picks so analysis/tracking.py can grade them after the games finish
try:
    from analysis.tracking import save_picks
    save_picks(all_picks)
except Exception:
    pass

min_rank = TIER_RANK[min_conf]
filtered = [p for p in all_picks if TIER_RANK.get(p["confidence_tier"], 0) >= min_rank]
if platforms:
    filtered = [p for p in filtered if p["pick_type"] == "game" or p.get("platform") in platforms]
if not show_game:
    filtered = [p for p in filtered if p["pick_type"] != "game"]
if not show_prop:
    filtered = [p for p in filtered if p["pick_type"] != "prop"]

best   = best_props_per_player(filtered)
hi     = [p for p in best if is_high_interest(p)]

# ── Game-time lookup: team name → {date, time, opponent label} ────────────────
_team_game_map: dict[str, dict] = {}
for _g in games:
    try:
        _ct   = datetime.fromisoformat(_g["game_time"].replace("Z", "+00:00")).astimezone(ZoneInfo("America/Chicago"))
        _date = _ct.strftime("%b %d")
        _time = _ct.strftime("%-I:%M %p")
    except Exception:
        _date = _g.get("date", "—")
        _time = "—"
    _h, _a = _g["home_team"], _g["away_team"]
    # "sort" = ISO UTC tip-off, used to order tables chronologically — the
    # display strings ("Jul 5", "9:00 PM") don't sort correctly as text.
    _sort_key = _g.get("game_time", "") or "~"
    _team_game_map[_h] = {"date": _date, "time": _time, "opp": f"vs {_a}", "sort": _sort_key}
    _team_game_map[_a] = {"date": _date, "time": _time, "opp": f"@ {_h}", "sort": _sort_key}
_team_game_map_lc = {k.lower(): v for k, v in _team_game_map.items()}


def _game_info(team: str) -> dict:
    return _team_game_map.get(team) or _team_game_map_lc.get(str(team).lower(), {})


# ── Data freshness ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def data_freshness():
    conn = get_conn()
    out = {}
    for label, table in (("Lines", "prop_lines"), ("Odds", "game_odds")):
        try:
            out[label] = conn.execute(f"SELECT MAX(fetched_at) FROM {table}").fetchone()[0]
        except Exception:
            out[label] = None
    conn.close()
    return out


def _age_label(ts: str | None) -> tuple[str, float]:
    """Human age string + age in hours (inf when unknown)."""
    if not ts:
        return "never", float("inf")
    try:
        dt = datetime.fromisoformat(ts)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
    except Exception:
        return "unknown", float("inf")
    if hours < 1:
        return f"{int(hours * 60)} min ago", hours
    if hours < 48:
        return f"{hours:.1f} h ago", hours
    return f"{hours / 24:.0f} days ago", hours


def timestamp_bar():
    fresh = data_freshness()
    parts, worst = [], 0.0
    for label, ts in fresh.items():
        txt, hours = _age_label(ts)
        worst = max(worst, hours)
        parts.append(f"{label}: <strong style='color:#e8eaf0'>{txt}</strong>")
    stale = worst > 6
    border = "#dc2626" if stale else "#22c55e"
    warn = ("  &nbsp;·&nbsp; <strong style='color:#f87171'>stale — hit Refresh in the sidebar</strong>"
            if stale else "")
    st.markdown(
        f"<div style='background:#1a1d27;border-left:3px solid {border};padding:8px 14px;"
        f"border-radius:4px;font-size:0.85rem;color:#9ca3af;margin-bottom:8px'>"
        f"🕐 Data updated — {' &nbsp;·&nbsp; '.join(parts)}{warn}</div>",
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


# ── Table helpers ──────────────────────────────────────────────────────────────
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


def picks_to_df(picks, show_context=False):
    rows = []
    for p in picks:
        if p["pick_type"] == "game":
            gi  = _game_info(p.get("home_team", ""))
            row = {
                "_sort":     gi.get("sort", "~"),
                "Game":      f"{p.get('away_team','')} @ {p.get('home_team','')}",
                "Time":      gi.get("time", "—"),
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
            gi  = _game_info(p.get("player_team", ""))
            ot  = p.get("odds_type", "standard")
            sel = f"{p['player_name']} {p['stat_type']} {p['direction']} {p['line']}"
            if ot in ("goblin", "demon"):
                sel += {"goblin": " 🐸", "demon": " 😈"}[ot]
            row = {
                "_sort":     gi.get("sort", "~"),
                "Game":      gi.get("opp", "—"),
                "Time":      gi.get("time", "—"),
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


# ── Header (rendered once, above the tabs) ─────────────────────────────────────
timestamp_bar()

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
    st.caption("Game picks (ML/Spread/Totals) + high-interest player props, ranked by EV.")

    metric_chips([
        ("Total",      str(len(hi))),
        ("Game Picks", str(sum(1 for p in hi if p["pick_type"] == "game"))),
        ("Prop Picks", str(sum(1 for p in hi if p["pick_type"] == "prop"))),
        ("STRONG",     str(sum(1 for p in hi if p["confidence_tier"] == "STRONG"))),
    ])

    show_ctx = st.toggle("Show season/recent context (props)", value=False)
    df = picks_to_df(hi[:75], show_context=show_ctx)
    if df is not None:
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
    elif not games:
        st.info("No games found — hit Refresh in the sidebar to fetch today's schedule.")
    elif not all_picks:
        st.info("Games found, but no picks cleared the edge threshold today.")
    else:
        st.info("Picks exist but none match the current filters. Try lowering Min Confidence.")
    st.caption("**Model %** = model probability. **Edge** = vs implied odds. **Stake** = quarter-Kelly sizing.")

# ── Tab 2: Game Predictions ────────────────────────────────────────────────────
with tab2:
    game_picks = [p for p in best if p["pick_type"] == "game"]

    # Model's view for every game, even where there's no bet. Shown with an
    # explicit "pass" so an empty market reads as a decision, not missing data.
    @st.cache_data(ttl=300, show_spinner=False)
    def _model_views(cache_key):
        conn    = get_conn()
        odds_df = pd.read_sql("SELECT * FROM game_odds WHERE DATE(fetched_at) = DATE('now')", conn)
        conn.close()
        views = {}
        for _g in games:
            _h, _a = _g["home_team"], _g["away_team"]
            try:
                _pred = predict_game(_h, _a)
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
                "home_p": float(norm.cdf(diff / _pred["spread_std"])),
                "spread_std": _pred["spread_std"], "totals_std": _pred["totals_std"],
                "raw_diff": _pred["pred_diff"], "raw_total": _pred["pred_total"],
            }
        return views

    views = _model_views(str(data_freshness()) + f"|{len(games)}")

    if not games:
        st.info("No games found — hit Refresh in the sidebar to fetch today's schedule.")
    else:
        st.caption(
            "**pass** = the model's probability doesn't beat the market price by enough to overcome the vig "
            "(spread/totals at -110 need ~56%+). A confident pass is still a pass — the payout already reflects "
            "the market's matching confidence."
        )
        for g in games:
            home, away = g["home_team"], g["away_team"]
            g_picks = [p for p in game_picks if p["home_team"] == home]
            gi = _game_info(home)
            tip = f"  ·  {gi['date']} {gi['time']} CT" if gi.get("time") and gi["time"] != "—" else ""

            with st.expander(f"**{away}** @ **{home}**{tip}", expanded=len(g_picks) > 0):
                ml = [p for p in g_picks if p["market"] == "Moneyline"]
                sp = [p for p in g_picks if p["market"] == "Spread"]
                to = [p for p in g_picks if p["market"] == "Totals"]
                v  = views.get(home)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("**Moneyline**")
                    for p in ml:
                        st.markdown(f"- {p['selection']}  `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']} `{int(p['best_odds']):+d}`")
                    if not ml and v:
                        fav, fp = (home, v["home_p"]) if v["home_p"] >= 0.5 else (away, 1 - v["home_p"])
                        st.caption(f"Model: {fav} `{fp:.0%}` — market agrees → **pass**")
                with col2:
                    st.markdown("**Spread**")
                    for p in sp:
                        st.markdown(f"- {p['selection']}  `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                    if not sp and v:
                        fav_t, fav_m = (home, v["diff"]) if v["diff"] >= 0 else (away, -v["diff"])
                        if v["spread_line"] is not None:
                            p_cov = float(norm.cdf((v["diff"] + v["spread_line"]) / v["spread_std"]))
                            side_lbl, pc = ((f"{home} {v['spread_line']:+.1f}", p_cov) if p_cov >= 0.5
                                            else (f"{away} {-v['spread_line']:+.1f}", 1 - p_cov))
                            st.caption(f"Model: {fav_t} by `{fav_m:.1f}` · {side_lbl} covers `{pc:.0%}` → **pass**")
                        else:
                            st.caption(f"Model: {fav_t} by `{fav_m:.1f}` — no line posted")
                with col3:
                    st.markdown("**Totals**")
                    for p in to:
                        st.markdown(f"- {p['selection']}  `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                    if not to and v:
                        if v["total_line"] is not None:
                            p_over = float(1 - norm.cdf(v["total_line"], v["total"], v["totals_std"]))
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
        df = df.drop(columns=["Time", "EV/$100", "Stake", "Win"], errors="ignore")
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
        st.caption("**Season/Recent** = the player's per-game average for this stat (full season vs last 5).")
    else:
        st.info("No prop picks match current filters.")

# ── Tab 4: Platform Comparison ─────────────────────────────────────────────────
with tab4:
    st.caption("Same player-prop across PrizePicks and Underdog side-by-side.")

    search = st.text_input("Search player…", "")
    prop_all = [p for p in filtered if p["pick_type"] == "prop"]

    from collections import defaultdict
    grouped = defaultdict(list)
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
    @st.cache_data(ttl=300)
    def load_graded():
        conn = get_conn()
        try:
            df = pd.read_sql("SELECT * FROM picks WHERE result IN ('win','loss','push')", conn)
        except Exception:
            df = pd.DataFrame()
        conn.close()
        return df

    graded = load_graded()
    wl = graded[graded["result"].isin(["win", "loss"])] if not graded.empty else pd.DataFrame()

    chips = [("Bankroll", f"${bankroll:,.0f}"), ("1 Unit", f"${unit_size:.0f}")]
    if not wl.empty:
        wins = int((wl["result"] == "win").sum())
        chips += [("Record", f"{wins}–{len(wl) - wins}"),
                  ("Hit Rate", f"{wins / len(wl):.1%}")]
    metric_chips(chips)

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
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Results")
        st.caption("Graded picks from analysis/tracking.py (start.sh grades yesterday's picks each morning).")
        if wl.empty:
            st.info("No graded picks yet — results appear after games finish and grading runs "
                    "(`python analysis/tracking.py grade`).")
        else:
            wl = wl.copy()
            wl["hit"] = (wl["result"] == "win").astype(int)
            by_tier = (wl.groupby("confidence_tier")
                         .agg(Picks=("hit", "size"), **{"Hit Rate": ("hit", "mean"),
                                                        "Avg Model %": ("model_prob", "mean")})
                         .reindex(["STRONG", "HIGH", "MEDIUM", "LOW"]).dropna(how="all"))
            by_tier["Hit Rate"]    = by_tier["Hit Rate"].map(lambda x: f"{x:.1%}")
            by_tier["Avg Model %"] = by_tier["Avg Model %"].map(lambda x: f"{x:.1%}")
            by_tier["Picks"]       = by_tier["Picks"].astype(int)
            st.markdown("**By confidence tier** — hit rate should track Avg Model %:")
            st.dataframe(by_tier, use_container_width=True)

            by_mkt = (wl.groupby(wl["market"].fillna("Player Prop"))
                        .agg(Picks=("hit", "size"), **{"Hit Rate": ("hit", "mean")}))
            by_mkt["Hit Rate"] = by_mkt["Hit Rate"].map(lambda x: f"{x:.1%}")
            by_mkt["Picks"]    = by_mkt["Picks"].astype(int)
            st.markdown("**By market:**")
            st.dataframe(by_mkt, use_container_width=True)

    with col2:
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
        answer = answer_question(question, all_picks, games, views, st.session_state)
        st.session_state.ask_history.append((question, answer))
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            st.markdown(answer)
