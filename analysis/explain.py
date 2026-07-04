"""
Deterministic pick explanations ("why does X have 86.7%?").

find_target() fuzzy-matches a free-text question to a prop pick or a game
market; render_*() turn the captured `explain` payloads (models/props.py,
picks/engine.py) into a markdown walkthrough of the actual arithmetic.
No LLM involved — every number shown is the one the pipeline computed.
"""
import re
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scipy.stats import norm

from models.props import STAT_MAP, LEAGUE_AVG_TEAM_PTS
from picks.engine import MIN_EDGE, MODEL_WEIGHT

# internal stat column → words a user might type for it
_STAT_WORDS = {
    "pts":     {"points", "pts", "point"},
    "reb":     {"rebounds", "rebs", "reb", "boards"},
    "ast":     {"assists", "asts", "ast", "dimes"},
    "stl":     {"steals", "stls", "steal"},
    "blk":     {"blocks", "blks", "block", "blocked"},
    "fg3m":    {"3-pointers", "3-pt", "3pt", "threes", "3s", "triples", "3-pointer"},
    "tov":     {"turnovers", "tov", "turnover"},
    "ftm":     {"free", "throws", "ft", "ftm"},
    "fantasy": {"fantasy"},
    "pra":     {"pra", "pts+rebs+asts"},
    "pts_reb": {"pts+rebs", "p+r"},
    "pts_ast": {"pts+asts", "p+a"},
    "reb_ast": {"rebs+asts", "r+a"},
    "blk_stl": {"blks+stls", "b+s"},
}
# combo stats also match on their component words
_STAT_WORDS["pra"]     |= {"points", "rebounds", "assists"}
_STAT_WORDS["pts_reb"] |= {"points", "rebounds"}
_STAT_WORDS["pts_ast"] |= {"points", "assists"}
_STAT_WORDS["reb_ast"] |= {"rebounds", "assists"}
_STAT_WORDS["blk_stl"] |= {"blocks", "steals"}

_COMBO_COLS = {"pra", "pts_reb", "pts_ast", "reb_ast", "blk_stl"}

_GENERIC_TEAM_WORDS = {"new", "las", "los", "state", "golden"}


def _numbers(text: str) -> list[float]:
    return [float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)]


def _score_prop(q: str, q_words: set, nums: list[float], p: dict) -> int:
    name  = p["player_name"].lower()
    last  = name.split()[-1]
    score = 0
    if name in q:
        score += 4
    elif last in q_words:
        score += 3
    else:
        return 0  # player name is required

    col = STAT_MAP.get(p["stat_type"])
    words = _STAT_WORDS.get(col, set())
    stat_hits = len(words & q_words)
    if stat_hits:
        # combos share component words with singles; reward matching more parts
        score += 2 if (col not in _COMBO_COLS or stat_hits >= 2) else 1

    if p["line"] in nums:
        score += 2
    d = p["direction"]
    if (d == "More" and ({"more", "over"} & q_words)) or (d == "Less" and ({"less", "under"} & q_words)):
        score += 1
    if p.get("platform", "") in q:
        score += 1
    return score


def _match_game(q_words: set, games: list[dict]):
    best, best_hits = None, 0
    for g in games:
        words = set()
        for team in (g["home_team"], g["away_team"]):
            words |= {w for w in team.lower().split() if len(w) >= 4 and w not in _GENERIC_TEAM_WORDS}
        # abbreviations like "gsv", "atl", "lva"
        for team in (g["home_team"], g["away_team"]):
            parts = team.lower().split()
            words.add("".join(w[0] for w in parts))
            words.add(parts[-1][:3])
        hits = len(words & q_words)
        if hits > best_hits:
            best, best_hits = g, hits
    return best


def find_target(query: str, picks: list[dict], games: list[dict], views: dict) -> dict | None:
    """Match a question to a prop pick or a game market.

    Returns {"kind": "prop", "pick": …} or
            {"kind": "game", "game": …, "market": …|None, "picks": […], "view": …|None}
    """
    q       = query.lower()
    q_words = set(re.findall(r"[\w+\-']+", q))
    nums    = _numbers(q)

    props  = [p for p in picks if p["pick_type"] == "prop"]
    scored = sorted(((_score_prop(q, q_words, nums, p), p) for p in props),
                    key=lambda t: (t[0], t[1]["edge"]), reverse=True)
    prop_score, prop_pick = scored[0] if scored else (0, None)

    if prop_score >= 3:
        return {"kind": "prop", "pick": prop_pick}

    game = _match_game(q_words, games)
    if game is not None:
        if {"over", "under", "total", "totals"} & q_words:
            market = "Totals"
        elif {"spread", "cover", "covers"} & q_words:
            market = "Spread"
        elif {"ml", "moneyline", "win", "wins"} & q_words:
            market = "Moneyline"
        else:
            market = None
        g_picks = [p for p in picks if p["pick_type"] == "game" and p["home_team"] == game["home_team"]]
        return {"kind": "game", "game": game, "market": market,
                "picks": g_picks, "view": views.get(game["home_team"])}

    if prop_pick is not None and prop_score > 0:
        return {"kind": "prop", "pick": prop_pick}
    return None


# ── Renderers ──────────────────────────────────────────────────────────────────

def render_prop(p: dict) -> str:
    e = p.get("explain") or {}
    if not e:
        return "That pick predates the explain feature — hit Refresh to rebuild picks."

    ot       = p.get("odds_type", "standard")
    ot_label = {"goblin": " · 🐸 goblin line", "demon": " · 😈 demon line"}.get(ot, "")
    lines = [f"**{p['player_name']} — {p['stat_type']} {p['direction']} {p['line']}** "
             f"({p['platform']}{ot_label}) → model **{p['model_prob']:.1%}**\n"]

    # 1. baseline
    if e["form_source"] == "per_min" and e.get("per_min") is not None:
        rec = f"{e['per_min_recent']:.3f}/min last 10" if e.get("per_min_recent") is not None else "no recent sample"
        lines.append(
            f"1. **Baseline** — production rate: {e['per_min_season']:.3f}/min this season, {rec} "
            f"→ 55/45 recent-weighted = {e['per_min']:.3f}/min × **{e['proj_min']:.1f} projected minutes** "
            f"(avg of last 5 played) = **{e['base_rate']}** ({e['n_games']} games in sample)")
    elif e["form_source"] == "blended":
        lines.append(
            f"1. **Baseline** — 55% × last-10 average ({p.get('recent_rate')}) + 45% × season average "
            f"({p.get('season_rate')}) = **{e['base_rate']}**")
    else:
        lines.append(f"1. **Baseline** — season average **{e['base_rate']}** over {e['n_games']} games "
                     f"(not enough recent data to weight form)")

    # 2. opponent defense
    if e.get("def_adj") is not None:
        opp = e.get("opponent") or "league-average opponent"
        lines.append(
            f"2. **Opponent** — {opp} allows {e['opp_pts']:.1f} ppg vs league average {LEAGUE_AVG_TEAM_PTS:.0f} "
            f"→ scoring stats scaled ×{e['def_adj']:.3f}")
    else:
        lines.append("2. **Opponent** — no scoring-defense adjustment (only applied to scoring stats)")

    # 3. pace
    if e.get("pace_factor") is not None:
        lines.append(f"3. **Pace** — expected game pace vs league average → ×{e['pace_factor']:.3f}")
    else:
        lines.append("3. **Pace** — no pace adjustment for this stat")

    lines.append(f"4. **Expected value: {e['expected']}** vs the line **{p['line']}**")

    # 5. distribution
    if e["dist"] == "normal":
        lines.append(
            f"5. **Probability** — normal distribution using {p['player_name'].split()[-1]}'s own game-to-game "
            f"std **{e['std']:.2f}**: P(more than {p['line']}) = {e['p_more']:.1%}, "
            f"P(less) = {1 - e['p_more']:.1%} → pick **{p['direction']}** at **{p['model_prob']:.1%}**")
    elif e["dist"] == "poisson":
        lines.append(
            f"5. **Probability** — Poisson(μ={e['expected']}) for a low-count stat: "
            f"P(more than {p['line']}) = {e['p_more']:.1%} → pick **{p['direction']}** at **{p['model_prob']:.1%}**")
    else:
        lines.append(
            f"5. **Probability** — no reliable per-player std, heuristic normal (σ = 35% of expected): "
            f"P(more) = {e['p_more']:.1%} → pick **{p['direction']}** at **{p['model_prob']:.1%}**")

    lines.append(
        f"6. **Edge** — {p['model_prob']:.1%} model vs **{e['breakeven']:.1%}** break-even "
        f"(a 2-pick 3× slip needs each leg at √(1/3) ≈ 57.7%) = **{p['edge']:+.1%}** "
        f"→ {p.get('confidence_tier', '—')} confidence")

    if ot == "goblin":
        lines.append("\n⚠️ Goblin (lowered) lines pay *less* than the standard 3× when in a slip, "
                     "so this edge is somewhat overstated.")
    elif ot == "demon":
        lines.append("\n⚠️ Demon (raised) lines pay *more* than the standard 3×, "
                     "so this edge is somewhat understated.")
    return "\n".join(lines)


def _game_numbers(picks: list[dict], view: dict | None) -> dict | None:
    """Prefer a pick's explain payload; fall back to the model-view dict."""
    for p in picks:
        if p.get("explain"):
            return p["explain"]
    if view is not None:
        return {
            "raw_diff": view.get("raw_diff"), "raw_total": view.get("raw_total"),
            "market_diff": -view["spread_line"] if view.get("spread_line") is not None else None,
            "market_total": view.get("total_line"),
            "anch_diff": view["diff"], "anch_total": view["total"],
            "spread_std": view["spread_std"], "totals_std": view["totals_std"],
            "home_win_prob": view["home_p"], "model_weight": MODEL_WEIGHT,
        }
    return None


def render_game(game: dict, market: str | None, picks: list[dict], view: dict | None) -> str:
    home, away = game["home_team"], game["away_team"]
    e = _game_numbers(picks, view)
    if e is None:
        return f"I can see **{away} @ {home}** on the schedule, but the model has no view yet — likely missing game logs or odds. Try Refresh."

    out = [f"**{away} @ {home}**\n"]
    markets = [market] if market else ["Moneyline", "Spread", "Totals"]

    fav, marg = (home, e["anch_diff"]) if e["anch_diff"] >= 0 else (away, -e["anch_diff"])
    raw_fav, raw_marg = (home, e["raw_diff"]) if e["raw_diff"] >= 0 else (away, -e["raw_diff"])

    for m in markets:
        pick = next((p for p in picks if p["market"] == m), None)
        has_line = e.get("market_diff") is not None if m in ("Moneyline", "Spread") else e.get("market_total") is not None
        if m in ("Moneyline", "Spread"):
            out.append(f"**{m}**")
            out.append(f"- XGBoost spread model (raw): {raw_fav} by {raw_marg:.1f}")
            if e.get("market_diff") is not None:
                mfav = home if e["market_diff"] >= 0 else away
                out.append(f"- Market line implies {mfav} by {abs(e['market_diff']):.1f} → anchored "
                           f"{e['model_weight']:.0%} model / {1 - e['model_weight']:.0%} market = **{fav} by {marg:.1f}**")
            else:
                out.append("- No market spread posted — raw model value used unanchored")
            if m == "Moneyline":
                out.append(f"- Win probability = Φ(margin {e['anch_diff']:+.1f} / calibration std {e['spread_std']:.1f}) "
                           f"= **{e['home_win_prob']:.1%} {home}** / {1 - e['home_win_prob']:.1%} {away}")
            elif e.get("market_diff") is not None:
                spread_line = -e["market_diff"]
                p_cover = float(norm.cdf((e["anch_diff"] + spread_line) / e["spread_std"]))
                side, pc = (f"{home} {spread_line:+.1f}", p_cover) if p_cover >= 0.5 else (f"{away} {-spread_line:+.1f}", 1 - p_cover)
                out.append(f"- Cover probability = Φ((margin {e['anch_diff']:+.1f} + line {spread_line:+.1f}) / "
                           f"std {e['spread_std']:.1f}) → **{side} covers {pc:.1%}**")
        else:
            out.append("**Totals**")
            out.append(f"- XGBoost totals model (raw): **{e['raw_total']:.1f}** combined points")
            if e.get("market_total") is not None:
                out.append(f"- Market line {e['market_total']:.1f} → anchored {e['model_weight']:.0%} model / "
                           f"{1 - e['model_weight']:.0%} market = **{e['anch_total']:.1f}**")
                p_over = float(1 - norm.cdf(e["market_total"], e["anch_total"], e["totals_std"]))
                ou, po = ("Over", p_over) if p_over >= 0.5 else ("Under", 1 - p_over)
                out.append(f"- P = Φ with calibration std {e['totals_std']:.1f} → **{ou} {e['market_total']:.1f} at {po:.1%}**")
            else:
                out.append("- No market total posted")

        if pick is not None:
            odds = f" `{int(pick['best_odds']):+d}`" if pick.get("best_odds") else ""
            out.append(f"- ✅ **Pick: {pick['selection']}**{odds} @ {pick['best_platform']} — "
                       f"model {pick['model_prob']:.1%} vs de-vigged implied {pick['implied_prob']:.1%} "
                       f"= edge **{pick['edge']:+.1%}** ({pick['confidence_tier']})")
        elif not has_line:
            out.append("- ❌ **No bet** — no odds fetched for this market yet (hit Refresh for current lines).")
        else:
            out.append(f"- ❌ **Pass** — edge vs the de-vigged book price is below the {MIN_EDGE:.0%} minimum. "
                       f"A confident probability isn't a bet if the payout already reflects it.")
        out.append("")
    return "\n".join(out)


NO_MATCH_MSG = (
    "I couldn't match that to a pick or game. Try including the **player's name** "
    "(e.g. *why does Breanna Stewart Rebs+Asts More 8.5 have 85%?*) or a **team + market** "
    "(e.g. *Chicago Sky @ Las Vegas Aces over 179.5*). Note: props are only explainable "
    "if they appear in today's pick list."
)
