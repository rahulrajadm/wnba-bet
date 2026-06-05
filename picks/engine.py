"""
WNBA picks engine.
Assembles game predictions (ML, spread, totals) + player props into
ranked +EV picks with confidence tiers, risk profiles, and Kelly stakes.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from utils.db import get_conn
from models.game import predict_game, prob_cover_spread, prob_over_total
from models.props import predict_props
from analysis.ev import ev_game, ev_prop, american_to_implied, remove_vig
from analysis.confidence import get_confidence_tier, TIER_RANK
from analysis.risk import get_risk_profile
from analysis.kelly import kelly_stake, stake_dollars, risk_reward

MIN_EDGE       = 0.04
MAX_PROB       = 0.88
MIN_PROB       = 0.12
DEFAULT_MULT   = 3.0   # 2-pick power play baseline for props

NO_LESS_AT_HALF = {"3-PT Made", "3-Pointers Made", "Blocked Shots", "Steals"}


def is_platform_realistic(pick: dict) -> bool:
    if pick.get("pick_type") == "prop":
        if pick["direction"] == "Less" and pick["line"] == 0.5 and pick["stat_type"] in NO_LESS_AT_HALF:
            return False
    return True


def is_high_interest(pick: dict) -> bool:
    if pick.get("pick_type") == "game":
        return True
    if pick["direction"] == "Less" and pick["line"] <= 0.5:
        return False
    return True


def build_game_picks(games: list[dict], bankroll: float, unit_size: float, odds_data: list[dict] | None = None, game_logs_df=None) -> list[dict]:
    """Generate ML, spread, and totals picks for today's games."""
    if odds_data is not None:
        odds_df = pd.DataFrame(odds_data)
    else:
        conn    = get_conn()
        odds_df = pd.read_sql("SELECT * FROM game_odds WHERE DATE(fetched_at) = DATE('now')", conn)
        conn.close()

    picks = []
    for game in games:
        home = game["home_team"]
        away = game["away_team"]

        pred = predict_game(home, away, game_logs_df=game_logs_df)
        if not pred:
            continue

        # Deduplicate odds: best line per market per team
        game_odds = odds_df[
            (odds_df["home_team"].str.lower() == home.lower()) |
            (odds_df["away_team"].str.lower() == home.lower())
        ]

        # ── Moneyline ──────────────────────────────────────────────────────────
        ml_odds = game_odds[game_odds["market"] == "moneyline"]
        if not ml_odds.empty:
            best_home = ml_odds.loc[ml_odds["home_odds"].apply(
                lambda x: x if x and x > 0 else (10000 / abs(x) if x else 0)).idxmax()]
            best_away = ml_odds.loc[ml_odds["away_odds"].apply(
                lambda x: x if x and x > 0 else (10000 / abs(x) if x else 0)).idxmax()]

            for side, model_p, best_row, odds_col in [
                (home, pred["home_win_prob"], best_home, "home_odds"),
                (away, pred["away_win_prob"], best_away, "away_odds"),
            ]:
                odds_val = best_row[odds_col]
                if not odds_val:
                    continue
                true_home_p, true_away_p = remove_vig(
                    best_home["home_odds"] or -110, best_away["away_odds"] or -110
                )
                implied_p = true_home_p if side == home else true_away_p
                edge      = model_p - implied_p
                if edge < MIN_EDGE or model_p > MAX_PROB or model_p < MIN_PROB:
                    continue
                ev        = ev_game(model_p, odds_val)
                conf      = get_confidence_tier(edge)
                risk      = get_risk_profile("moneyline", edge)
                mult      = (odds_val / 100 + 1) if odds_val > 0 else (100 / abs(odds_val) + 1)
                kpct      = kelly_stake(model_p, mult)
                stake     = stake_dollars(kpct, bankroll)
                rr        = risk_reward(stake, mult)

                picks.append({
                    "pick_type":         "game",
                    "market":            "Moneyline",
                    "selection":         f"{side} ML",
                    "home_team":         home,
                    "away_team":         away,
                    "best_platform":     best_row["platform"],
                    "best_odds":         odds_val,
                    "model_prob":        round(model_p, 4),
                    "implied_prob":      round(implied_p, 4),
                    "edge":              round(edge, 4),
                    "ev_per_100":        ev,
                    "confidence_tier":   conf,
                    "risk_profile":      risk,
                    "kelly_pct":         kpct,
                    "units":             round(stake / unit_size, 2) if unit_size > 0 else 0,
                    "stake_dollars":     stake,
                    "potential_win":     rr["potential_win"],
                    "risk_reward_ratio": rr["ratio"],
                    "stat_type":         "moneyline",
                    "direction":         "ML",
                    "line":              None,
                    "player_name":       "",
                    "platform":          best_row["platform"],
                })

        # ── Spread ─────────────────────────────────────────────────────────────
        sprd_odds = game_odds[game_odds["market"] == "spread"]
        if not sprd_odds.empty and pred.get("pred_diff") is not None:
            best_sprd = sprd_odds.sort_values("home_odds", ascending=False).iloc[0]
            spread_line = best_sprd.get("home_spread") or 0
            p_cover   = prob_cover_spread(pred["pred_diff"], spread_line, pred["spread_std"])
            p_not     = 1 - p_cover
            imp_p     = american_to_implied(best_sprd["home_odds"] or -110)

            for side_label, side_p, side_odds in [
                (f"{home} {spread_line:+.1f}", p_cover, best_sprd["home_odds"]),
                (f"{away} {-spread_line:+.1f}", p_not,  best_sprd["away_odds"]),
            ]:
                if not side_odds:
                    continue
                edge = side_p - american_to_implied(side_odds)
                if edge < MIN_EDGE or side_p > MAX_PROB or side_p < MIN_PROB:
                    continue
                ev   = ev_game(side_p, side_odds)
                conf = get_confidence_tier(edge)
                risk = get_risk_profile("spread", edge)
                mult = (side_odds / 100 + 1) if side_odds > 0 else (100 / abs(side_odds) + 1)
                kpct = kelly_stake(side_p, mult)
                stake = stake_dollars(kpct, bankroll)
                rr   = risk_reward(stake, mult)

                picks.append({
                    "pick_type":         "game",
                    "market":            "Spread",
                    "selection":         side_label,
                    "home_team":         home,
                    "away_team":         away,
                    "best_platform":     best_sprd["platform"],
                    "best_odds":         side_odds,
                    "model_prob":        round(side_p, 4),
                    "implied_prob":      round(american_to_implied(side_odds), 4),
                    "edge":              round(edge, 4),
                    "ev_per_100":        ev,
                    "confidence_tier":   conf,
                    "risk_profile":      risk,
                    "kelly_pct":         kpct,
                    "units":             round(stake / unit_size, 2) if unit_size > 0 else 0,
                    "stake_dollars":     stake,
                    "potential_win":     rr["potential_win"],
                    "risk_reward_ratio": rr["ratio"],
                    "stat_type":         "spread",
                    "direction":         "Cover",
                    "line":              spread_line,
                    "player_name":       "",
                    "platform":          best_sprd["platform"],
                })

        # ── Totals ─────────────────────────────────────────────────────────────
        tot_odds = game_odds[game_odds["market"] == "totals"]
        if not tot_odds.empty and pred.get("pred_total") is not None:
            best_tot    = tot_odds.sort_values("over_odds", ascending=False).iloc[0]
            total_line  = best_tot.get("total_line") or 165.0
            p_over      = prob_over_total(pred["pred_total"], total_line, pred["totals_std"])
            p_under     = 1 - p_over

            for label, side_p, side_odds in [
                (f"Over {total_line}", p_over,  best_tot["over_odds"]),
                (f"Under {total_line}", p_under, best_tot["under_odds"]),
            ]:
                if not side_odds:
                    continue
                edge = side_p - american_to_implied(side_odds)
                if edge < MIN_EDGE or side_p > MAX_PROB or side_p < MIN_PROB:
                    continue
                ev   = ev_game(side_p, side_odds)
                conf = get_confidence_tier(edge)
                risk = get_risk_profile("totals", edge)
                mult = (side_odds / 100 + 1) if side_odds > 0 else (100 / abs(side_odds) + 1)
                kpct = kelly_stake(side_p, mult)
                stake = stake_dollars(kpct, bankroll)
                rr   = risk_reward(stake, mult)

                picks.append({
                    "pick_type":         "game",
                    "market":            "Totals",
                    "selection":         label,
                    "home_team":         home,
                    "away_team":         away,
                    "best_platform":     best_tot["platform"],
                    "best_odds":         side_odds,
                    "model_prob":        round(side_p, 4),
                    "implied_prob":      round(american_to_implied(side_odds), 4),
                    "edge":              round(edge, 4),
                    "ev_per_100":        ev,
                    "confidence_tier":   conf,
                    "risk_profile":      risk,
                    "kelly_pct":         kpct,
                    "units":             round(stake / unit_size, 2) if unit_size > 0 else 0,
                    "stake_dollars":     stake,
                    "potential_win":     rr["potential_win"],
                    "risk_reward_ratio": rr["ratio"],
                    "stat_type":         "totals",
                    "direction":         "Over/Under",
                    "line":              total_line,
                    "player_name":       "",
                    "platform":          best_tot["platform"],
                })

    return picks


def build_prop_picks(bankroll: float, unit_size: float, lines_data=None, player_logs_df=None, team_logs_df=None) -> list[dict]:
    """Generate player prop picks."""
    raw   = predict_props(lines_data=lines_data, player_logs_df=player_logs_df, team_logs_df=team_logs_df)
    picks = []

    for pred in raw:
        edge       = pred["edge"]
        model_prob = pred["model_prob"]

        if edge < MIN_EDGE or model_prob > MAX_PROB or model_prob < MIN_PROB:
            continue

        fake_pick = {"pick_type": "prop", "direction": pred["direction"],
                     "line": pred["line"], "stat_type": pred["stat_type"]}
        if not is_platform_realistic(fake_pick):
            continue

        ev    = ev_prop(model_prob)
        conf  = get_confidence_tier(edge)
        risk  = get_risk_profile(pred["stat_type"], edge)
        kpct  = kelly_stake(model_prob, DEFAULT_MULT)
        stake = stake_dollars(kpct, bankroll)
        rr    = risk_reward(stake, DEFAULT_MULT)

        picks.append({
            "pick_type":         "prop",
            "market":            "Player Prop",
            "selection":         f"{pred['player_name']} {pred['stat_type']} {pred['direction']} {pred['line']}",
            "player_name":       pred["player_name"],
            "player_team":       pred.get("player_team", ""),
            "stat_type":         pred["stat_type"],
            "line":              pred["line"],
            "direction":         pred["direction"],
            "best_platform":     pred["platform"],
            "best_odds":         None,
            "model_prob":        model_prob,
            "implied_prob":      0.50,
            "edge":              edge,
            "ev_per_100":        ev,
            "confidence_tier":   conf,
            "risk_profile":      risk,
            "kelly_pct":         kpct,
            "units":             round(stake / unit_size, 2) if unit_size > 0 else 0,
            "stake_dollars":     stake,
            "potential_win":     rr["potential_win"],
            "risk_reward_ratio": rr["ratio"],
            "season_rate":       pred.get("season_rate"),
            "recent_rate":       pred.get("recent_rate"),
            "platform":          pred["platform"],
            "home_team":         "",
            "away_team":         "",
        })

    return picks


def build_picks(
    games: list[dict],
    bankroll: float = 500.0,
    unit_size: float = 10.0,
    lines_data=None,
    odds_data=None,
    game_logs_df=None,
    player_logs_df=None,
    team_logs_df=None,
) -> list[dict]:
    game_picks = build_game_picks(games, bankroll, unit_size, odds_data=odds_data, game_logs_df=game_logs_df)
    prop_picks = build_prop_picks(bankroll, unit_size, lines_data, player_logs_df=player_logs_df, team_logs_df=team_logs_df)
    all_picks  = game_picks + prop_picks
    all_picks.sort(
        key=lambda x: (TIER_RANK.get(x["confidence_tier"], 0), x["ev_per_100"]),
        reverse=True
    )
    return all_picks


def best_props_per_player(picks: list[dict]) -> list[dict]:
    seen, out = {}, []
    for p in picks:
        if p["pick_type"] != "prop":
            out.append(p)
            continue
        key = (p["player_name"], p["stat_type"])
        if key not in seen:
            seen[key] = True
            out.append(p)
    return out


if __name__ == "__main__":
    from pipeline.schedule import get_today_games
    games = get_today_games()
    picks = build_picks(games, bankroll=500, unit_size=10)
    best  = best_props_per_player(picks)

    game_picks = [p for p in best if p["pick_type"] == "game"]
    prop_picks = [p for p in best if p["pick_type"] == "prop"]

    print(f"\nGame picks: {len(game_picks)}  |  Prop picks: {len(prop_picks)}\n")
    print("── GAME PICKS ──────────────────────────────────────────────────────")
    print(f"{'Selection':<35} {'Market':>8} {'Model%':>7} {'Edge':>6} {'EV/100':>7} {'Conf':>7} {'Platform'}")
    print("-" * 90)
    for p in game_picks:
        print(f"{p['selection']:<35} {p['market']:>8} {p['model_prob']:>6.1%} "
              f"{p['edge']:>+6.1%} {p['ev_per_100']:>+7.1f} {p['confidence_tier']:>7}  {p['platform']}")

    print("\n── TOP PROP PICKS ──────────────────────────────────────────────────")
    hi_props = [p for p in prop_picks if is_high_interest(p) and 0.55 <= p["model_prob"] <= 0.85][:15]
    print(f"{'Player':<22} {'Stat':<22} {'Line':>5} {'Dir':>5} {'Model%':>7} {'Edge':>6}  {'Platform'}")
    print("-" * 90)
    for p in hi_props:
        print(f"{p['player_name']:<22} {p['stat_type']:<22} {p['line']:>5} "
              f"{p['direction']:>5} {p['model_prob']:>6.1%} {p['edge']:>+6.1%}  {p['platform']}")
