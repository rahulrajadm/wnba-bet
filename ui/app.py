"""
WNBA Bet — Streamlit Dashboard (local, uses SQLite)
Tabs: 🔥 Game Picks | 🏀 Game Predictions | 🎯 Player Props | 📊 Platform Comparison | 💰 Bankroll
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import streamlit as st
import pandas as pd
from zoneinfo import ZoneInfo
from datetime import datetime

from pipeline.schedule import get_today_games
from pipeline.prizepicks import get_prizepicks_lines
from pipeline.underdog import get_underdog_lines
from pipeline.odds import get_all_odds
from picks.engine import build_picks, best_props_per_player, is_high_interest
from analysis.confidence import TIER_COLORS, TIER_RANK
from analysis.risk import RISK_COLORS
from analysis.ev import ev_slip

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

    st.divider()
    min_conf  = st.selectbox("Min Confidence", ["LOW", "MEDIUM", "HIGH", "STRONG"], index=1)
    platforms = st.multiselect("Prop Platforms", ["prizepicks", "underdog"], default=["prizepicks", "underdog"])
    show_game = st.toggle("Show game picks", value=True)
    show_prop = st.toggle("Show player props", value=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data(bankroll, unit_size):
    games = get_today_games()
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

# ── Helpers ────────────────────────────────────────────────────────────────────
def timestamp_bar():
    now = datetime.now(ZoneInfo("America/Chicago")).strftime("%b %d %Y, %I:%M %p")
    st.markdown(
        f"<div style='background:#1a1d27;border-left:3px solid #22c55e;padding:8px 14px;"
        f"border-radius:4px;font-size:0.85rem;color:#9ca3af;margin-bottom:8px'>"
        f"🕐 Data last updated: <strong style='color:#e8eaf0'>{now} CT</strong></div>",
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
                "Type":          p["market"],
                "Selection":     p["selection"],
                "Best Platform": p["best_platform"],
                "Odds":          f"{int(p['best_odds']):+d}" if p.get("best_odds") else "—",
                "Model %":       f"{p['model_prob']:.1%}",
                "Edge":          f"{p['edge']:+.1%}",
                "EV / $100":     f"${p['ev_per_100']:+.1f}",
                "Confidence":    p["confidence_tier"],
                "Risk":          p["risk_profile"],
                "Units":         f"{p.get('units', 0):.1f}u",
                "Stake ($)":     f"${p['stake_dollars']:.0f}",
                "Win ($)":       f"${p['potential_win']:.0f}",
            }
        else:
            row = {
                "Type":          "Prop",
                "Selection":     f"{p['player_name']} {p['stat_type']} {p['direction']} {p['line']}",
                "Best Platform": p["platform"],
                "Odds":          "—",
                "Model %":       f"{p['model_prob']:.1%}",
                "Edge":          f"{p['edge']:+.1%}",
                "EV / $100":     f"${p['ev_per_100']:+.1f}",
                "Confidence":    p["confidence_tier"],
                "Risk":          p["risk_profile"],
                "Units":         f"{p.get('units', 0):.1f}u",
                "Stake ($)":     f"${p['stake_dollars']:.0f}",
                "Win ($)":       f"${p['potential_win']:.0f}",
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
    timestamp_bar()
    st.header("🔥 Top Picks")
    st.caption("Game picks (ML/Spread/Totals) + high-interest player props, ranked by EV.")

    game_cnt = sum(1 for p in hi if p["pick_type"] == "game")
    prop_cnt = sum(1 for p in hi if p["pick_type"] == "prop")
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Total",       len(hi))
    c2.metric("Game Picks",  game_cnt)
    c3.metric("Prop Picks",  prop_cnt)
    c4.metric("STRONG",      sum(1 for p in hi if p["confidence_tier"] == "STRONG"))

    st.divider()
    show_ctx = st.toggle("Show season/recent context (props)", value=False)
    df = picks_to_df(hi[:75], show_context=show_ctx)
    if df is not None:
        st.dataframe(style_df(df), use_container_width=True, hide_index=True)
    else:
        st.info("No picks match current filters.")
    st.caption("**Model %** = model probability. **Edge** = vs implied odds. **Units** = quarter-Kelly stake.")

# ── Tab 2: Game Predictions ────────────────────────────────────────────────────
with tab2:
    timestamp_bar()
    st.header("🏀 Game Predictions")

    game_picks = [p for p in best if p["pick_type"] == "game"]
    if not game_picks and not games:
        st.info("No games found for today.")
    else:
        for g in games:
            home, away = g["home_team"], g["away_team"]
            g_picks = [p for p in game_picks if p["home_team"] == home]
            ml_picks = [p for p in g_picks if p["market"] == "Moneyline"]
            sp_picks = [p for p in g_picks if p["market"] == "Spread"]
            to_picks = [p for p in g_picks if p["market"] == "Totals"]

            with st.expander(f"**{away}** @ **{home}**", expanded=len(g_picks) > 0):
                if g_picks:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown("**Moneyline**")
                        for p in ml_picks:
                            st.markdown(f"- {p['selection']}  `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']} `{int(p['best_odds']):+d}`")
                    with col2:
                        st.markdown("**Spread**")
                        for p in sp_picks:
                            st.markdown(f"- {p['selection']}  `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                    with col3:
                        st.markdown("**Totals**")
                        for p in to_picks:
                            st.markdown(f"- {p['selection']}  `{p['model_prob']:.1%}` edge `{p['edge']:+.1%}` @ {p['best_platform']}")
                else:
                    st.caption("No +EV picks found for this game.")

# ── Tab 3: Player Props ────────────────────────────────────────────────────────
with tab3:
    timestamp_bar()
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
    timestamp_bar()
    st.header("📊 Platform Comparison")
    st.caption("Same player-prop across PrizePicks and Underdog side-by-side.")

    search = st.text_input("Search player…", "")
    prop_all = [p for p in filtered if p["pick_type"] == "prop"]

    from collections import defaultdict
    grouped = defaultdict(list)
    for p in prop_all:
        grouped[(p["player_name"], p["stat_type"])].append(p)

    items = [(k, v) for k, v in grouped.items() if len(v) > 1]
    if search:
        items = [i for i in items if search.lower() in i[0][0].lower()]

    if not items:
        st.info("No cross-platform comparisons available. Try refreshing data.")
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
    col1, col2 = st.columns(2)

    with col1:
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

    with col2:
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
