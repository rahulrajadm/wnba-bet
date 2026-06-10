"""
Pick logging and grading.

save_picks()  — write generated picks to the picks table (deduped per day).
grade_picks() — fetch final results from ESPN and grade saved picks.
report()      — hit rate by market/tier + probability calibration table.

Without this loop there is no way to tell whether a model change helped:
2 days of results is noise; judge over 100+ graded picks.

CLI:
    python analysis/tracking.py grade
    python analysis/tracking.py report
"""
import sys, os, json, sqlite3
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import datetime, date, timedelta
import pandas as pd
from utils.db import get_conn

EXTRA_COLS = {
    "game_date":    "TEXT",
    "market":       "TEXT",
    "home_team":    "TEXT",
    "away_team":    "TEXT",
    "player_name":  "TEXT",
    "stat_type":    "TEXT",
    "line":         "REAL",
    "direction":    "TEXT",
    "best_odds":    "REAL",
    "result":       "TEXT",   # win / loss / push
    "actual_value": "REAL",
    "graded_at":    "TEXT",
}


def _ensure_schema(conn):
    for col, typ in EXTRA_COLS.items():
        try:
            conn.execute(f"ALTER TABLE picks ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already exists
    conn.commit()


def _et_today() -> str:
    try:
        from zoneinfo import ZoneInfo
        return datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    except Exception:
        return date.today().isoformat()


def save_picks(picks: list[dict], game_date: str | None = None) -> int:
    """Insert today's picks, skipping ones already saved for this date."""
    if not picks:
        return 0
    game_date = game_date or _et_today()
    conn = get_conn()
    _ensure_schema(conn)

    existing = {
        (r[0], r[1], r[2]) for r in conn.execute(
            "SELECT selection, best_platform, IFNULL(line, -999) FROM picks WHERE game_date = ?",
            (game_date,),
        )
    }

    inserted = 0
    now = datetime.now().isoformat(timespec="seconds")
    for p in picks:
        key = (p.get("selection"), p.get("best_platform"), p.get("line") if p.get("line") is not None else -999)
        if key in existing:
            continue
        existing.add(key)
        conn.execute(
            """INSERT INTO picks (generated_at, pick_type, selection, best_platform,
                   model_prob, implied_prob, edge, ev_per_100, confidence_tier,
                   risk_profile, kelly_pct, units, details,
                   game_date, market, home_team, away_team, player_name,
                   stat_type, line, direction, best_odds)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (now, p.get("pick_type"), p.get("selection"), p.get("best_platform"),
             p.get("model_prob"), p.get("implied_prob"), p.get("edge"), p.get("ev_per_100"),
             p.get("confidence_tier"), p.get("risk_profile"), p.get("kelly_pct"), p.get("units"),
             json.dumps({"odds_type": p.get("odds_type", ""), "player_team": p.get("player_team", "")}),
             game_date, p.get("market"), p.get("home_team"), p.get("away_team"),
             p.get("player_name"), p.get("stat_type"), p.get("line"), p.get("direction"),
             p.get("best_odds")),
        )
        inserted += 1
    conn.commit()
    conn.close()
    return inserted


# ── Grading ────────────────────────────────────────────────────────────────────

def _team_key(name: str) -> str:
    return str(name).strip().split()[-1].lower() if name else ""


def _fetch_results(lookback: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Final team scores + player box scores for completed games, last N days."""
    from pipeline.cloud_data import _completed_game_ids, _box_score_team_rows, \
        _box_score_player_rows, _fetch_all
    gids = _completed_game_ids(lookback=lookback)
    team_df   = _fetch_all(gids, _box_score_team_rows)
    player_df = _fetch_all(gids, _box_score_player_rows)
    if not player_df.empty:
        player_df["pra"]     = player_df["pts"] + player_df["reb"] + player_df["ast"]
        player_df["pts_reb"] = player_df["pts"] + player_df["reb"]
        player_df["pts_ast"] = player_df["pts"] + player_df["ast"]
        player_df["reb_ast"] = player_df["reb"] + player_df["ast"]
        player_df["blk_stl"] = player_df["blk"] + player_df["stl"]
        player_df["fantasy"] = (player_df["pts"] + player_df["reb"] * 1.2 + player_df["ast"] * 1.5
                                + player_df["stl"] * 3 + player_df["blk"] * 3 - player_df["tov"])
    return team_df, player_df


def _close_date(d1: str, d2: str) -> bool:
    """ESPN dates are UTC, pick dates are ET — evening games differ by one day."""
    try:
        a = datetime.strptime(str(d1)[:10], "%Y-%m-%d").date()
        b = datetime.strptime(str(d2)[:10], "%Y-%m-%d").date()
        return abs((a - b).days) <= 1
    except Exception:
        return False


def _find_game(team_df: pd.DataFrame, home: str, away: str, gdate: str):
    """Return (home_pts, away_pts) for the matchup nearest gdate, or None."""
    hk, ak = _team_key(home), _team_key(away)
    keys = team_df["team_name"].map(_team_key)
    for gid, grp in team_df.groupby("game_id"):
        gkeys = set(grp["team_name"].map(_team_key))
        if hk in gkeys and ak in gkeys and _close_date(grp["game_date"].iloc[0], gdate):
            h = grp[grp["team_name"].map(_team_key) == hk]["pts"].iloc[0]
            a = grp[grp["team_name"].map(_team_key) == ak]["pts"].iloc[0]
            return float(h), float(a)
    return None


def _find_player_stat(player_df: pd.DataFrame, player: str, stat_col: str, gdate: str):
    rows = player_df[player_df["player_name"].str.lower() == str(player).lower()]
    if rows.empty:
        last = str(player).split()[-1].lower()
        cand = player_df[player_df["player_name"].str.lower().str.contains(last, na=False)]
        if cand.empty or cand["player_name"].str.lower().nunique() != 1:
            return None
        rows = cand
    rows = rows[rows["game_date"].map(lambda d: _close_date(d, gdate))]
    if rows.empty or stat_col not in rows.columns:
        return None
    return float(rows.iloc[0][stat_col])


def grade_picks(lookback: int = 10) -> dict:
    from models.props import STAT_MAP

    conn = get_conn()
    _ensure_schema(conn)
    ungraded = pd.read_sql(
        "SELECT * FROM picks WHERE (result IS NULL OR result = '') AND game_date <= ?",
        conn, params=(_et_today(),),
    )
    if ungraded.empty:
        conn.close()
        return {"graded": 0, "pending": 0}

    team_df, player_df = _fetch_results(lookback=lookback)
    graded = 0
    now = datetime.now().isoformat(timespec="seconds")

    for _, p in ungraded.iterrows():
        result, actual = None, None

        if p["pick_type"] == "game" and not team_df.empty:
            scores = _find_game(team_df, p["home_team"], p["away_team"], p["game_date"])
            if scores is None:
                continue
            h, a = scores
            sel = str(p["selection"])
            if p["market"] == "Moneyline":
                side_home = sel.lower().startswith(str(p["home_team"]).lower())
                won = h > a if side_home else a > h
                result, actual = ("win" if won else "loss"), (h - a)
            elif p["market"] == "Spread":
                side_home = sel.lower().startswith(str(p["home_team"]).lower())
                line   = float(p["line"] or 0)
                side_line = line if side_home else -line
                margin = (h - a) if side_home else (a - h)
                actual = h - a
                result = "push" if margin + side_line == 0 else ("win" if margin + side_line > 0 else "loss")
            elif p["market"] == "Totals":
                total, line = h + a, float(p["line"] or 0)
                actual = total
                if total == line:
                    result = "push"
                else:
                    over = total > line
                    result = "win" if (sel.startswith("Over") == over) else "loss"

        elif p["pick_type"] == "prop" and not player_df.empty:
            stat_col = STAT_MAP.get(p["stat_type"])
            if stat_col is None:
                continue
            actual = _find_player_stat(player_df, p["player_name"], stat_col, p["game_date"])
            if actual is None:
                continue
            line = float(p["line"] or 0)
            if actual == line:
                result = "push"
            else:
                more = actual > line
                result = "win" if ((p["direction"] == "More") == more) else "loss"

        if result is not None:
            conn.execute(
                "UPDATE picks SET result = ?, actual_value = ?, graded_at = ? WHERE id = ?",
                (result, actual, now, int(p["id"])),
            )
            graded += 1

    conn.commit()
    pending = pd.read_sql(
        "SELECT COUNT(*) AS n FROM picks WHERE (result IS NULL OR result = '')", conn
    )["n"].iloc[0]
    conn.close()
    return {"graded": graded, "pending": int(pending)}


def report() -> pd.DataFrame | None:
    conn = get_conn()
    _ensure_schema(conn)
    df = pd.read_sql("SELECT * FROM picks WHERE result IN ('win','loss')", conn)
    conn.close()
    if df.empty:
        print("No graded picks yet. Run: python analysis/tracking.py grade")
        return None

    df["hit"] = (df["result"] == "win").astype(int)

    print(f"\nGraded picks: {len(df)}  |  overall hit rate: {df['hit'].mean():.1%}\n")
    by_market = df.groupby(df["market"].fillna("Player Prop")).agg(
        picks=("hit", "size"), hit_rate=("hit", "mean"), avg_model_prob=("model_prob", "mean"))
    print("By market:")
    print(by_market.to_string(float_format=lambda x: f"{x:.3f}"))

    by_tier = df.groupby("confidence_tier").agg(picks=("hit", "size"), hit_rate=("hit", "mean"))
    print("\nBy confidence tier:")
    print(by_tier.to_string(float_format=lambda x: f"{x:.3f}"))

    # Calibration: does a 65% model prob actually hit 65%?
    df["prob_bucket"] = (df["model_prob"] * 10).round() / 10
    calib = df.groupby("prob_bucket").agg(picks=("hit", "size"), actual=("hit", "mean"))
    print("\nCalibration (model prob bucket vs actual hit rate):")
    print(calib.to_string(float_format=lambda x: f"{x:.3f}"))
    return df


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "grade":
        out = grade_picks()
        print(f"Graded {out['graded']} picks ({out['pending']} still pending).")
    else:
        report()
