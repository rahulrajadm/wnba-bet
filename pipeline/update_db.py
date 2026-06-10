"""
Ingest current-season game logs from ESPN into the local DB so the training
set includes this season (expansion teams, roster changes, current form).

Idempotent: rows for refetched game_ids are replaced, not duplicated.

Usage:
    python pipeline/update_db.py            # current calendar year
    python pipeline/update_db.py 2025 2026  # specific years
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from datetime import date
import pandas as pd
from utils.db import get_conn, init_db
from pipeline.cloud_data import (
    _get, _fetch_all, _box_score_team_rows, _box_score_player_rows,
)

TEAM_COLS = ["game_id", "season", "team_id", "team_name", "team_abbr", "game_date",
             "matchup", "wl", "pts", "fg_pct", "fg3_pct", "ft_pct", "reb", "ast",
             "stl", "blk", "tov", "plus_minus"]

PLAYER_COLS = ["season", "player_id", "player_name", "team_abbr", "game_id",
               "game_date", "matchup", "wl", "min", "pts", "reb", "ast", "stl",
               "blk", "tov", "fg3m", "fgm", "fga", "fg_pct", "ftm", "fta",
               "plus_minus"]


def _year_game_ids(year: int) -> list[str]:
    data = _get(
        "https://site.api.espn.com/apis/site/v2/sports/basketball/wnba/scoreboard",
        {"dates": f"{year}0101-{year}1231", "limit": 999},
    )
    return [
        e["id"] for e in data.get("events", [])
        if e.get("competitions", [{}])[0].get("status", {}).get("type", {}).get("completed")
        and e.get("season", {}).get("type", 2) != 1   # skip preseason/exhibitions
    ]


def _replace_rows(conn, table: str, df: pd.DataFrame, cols: list[str]):
    """Delete any existing rows for these game_ids, then insert fresh ones."""
    for c in cols:
        if c not in df.columns:
            df[c] = "" if c in ("matchup", "wl", "season", "player_id") else 0.0
    df = df[cols].copy()
    gids = df["game_id"].astype(str).unique().tolist()
    for i in range(0, len(gids), 500):
        chunk = gids[i:i + 500]
        conn.execute(
            f"DELETE FROM {table} WHERE game_id IN ({','.join('?' * len(chunk))})", chunk
        )
    df.to_sql(table, conn, if_exists="append", index=False)
    conn.commit()
    return len(df)


def ingest_year(year: int) -> None:
    print(f"Fetching {year} completed games from ESPN...")
    gids = _year_game_ids(year)
    print(f"  {len(gids)} completed games")
    if not gids:
        return

    team_df = _fetch_all(gids, _box_score_team_rows)

    # Drop preseason exhibitions vs national teams (ESPN lists them as e.g.
    # "JAPAN", "NIGERIA") — they aren't WNBA games and pollute rolling stats.
    if not team_df.empty:
        exhib_gids = set(team_df.loc[team_df["team_name"].str.isupper(), "game_id"])
        if exhib_gids:
            print(f"  Skipping {len(exhib_gids)} exhibition games (national teams)")
            team_df = team_df[~team_df["game_id"].isin(exhib_gids)]
            gids = [g for g in gids if g not in exhib_gids]

    if not team_df.empty:
        team_df["season"] = team_df["game_date"].astype(str).str[:4]
        conn = get_conn()
        n = _replace_rows(conn, "team_game_logs", team_df, TEAM_COLS)
        conn.close()
        print(f"  team_game_logs: {n} rows upserted")

    player_df = _fetch_all(gids, _box_score_player_rows)
    if not player_df.empty:
        player_df["season"] = player_df["game_date"].astype(str).str[:4]
        conn = get_conn()
        n = _replace_rows(conn, "player_game_logs", player_df, PLAYER_COLS)
        conn.close()
        print(f"  player_game_logs: {n} rows upserted")


if __name__ == "__main__":
    init_db()
    years = [int(y) for y in sys.argv[1:]] or [date.today().year]
    for y in years:
        ingest_year(y)
    print("\nDone. Retrain with: python models/train.py")
