"""
Pulls historical WNBA team game logs and player game logs from the NBA Stats API.
Run once to seed the database.
"""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog, playergamelogs, leaguedashteamstats
from utils.db import get_conn, init_db

SEASONS   = ["2022", "2023", "2024", "2025"]
LEAGUE_ID = "10"   # WNBA


def pull_team_game_logs():
    conn = get_conn()
    print("Pulling WNBA team game logs...")
    for season in SEASONS:
        print(f"  Season {season}...")
        try:
            log  = leaguegamelog.LeagueGameLog(league_id=LEAGUE_ID, season=season, season_type_all_star="Regular Season")
            df   = log.get_data_frames()[0]
            rows = []
            for _, r in df.iterrows():
                rows.append({
                    "game_id":    r["GAME_ID"],
                    "season":     season,
                    "team_id":    str(r["TEAM_ID"]),
                    "team_name":  r["TEAM_NAME"],
                    "team_abbr":  r["TEAM_ABBREVIATION"],
                    "game_date":  r["GAME_DATE"],
                    "matchup":    r["MATCHUP"],
                    "wl":         r["WL"],
                    "pts":        r["PTS"],
                    "fg_pct":     r["FG_PCT"],
                    "fg3_pct":    r["FG3_PCT"],
                    "ft_pct":     r["FT_PCT"],
                    "reb":        r["REB"],
                    "ast":        r["AST"],
                    "stl":        r["STL"],
                    "blk":        r["BLK"],
                    "tov":        r["TOV"],
                    "plus_minus": r["PLUS_MINUS"],
                })
            pd.DataFrame(rows).to_sql("team_game_logs", conn, if_exists="append", index=False)
            print(f"    {len(rows)} team-game rows")
            time.sleep(0.6)
        except Exception as e:
            print(f"    Warning: {e}")
    conn.commit()
    conn.close()


def pull_team_stats():
    conn = get_conn()
    print("Pulling WNBA team season stats (base + advanced)...")
    for season in SEASONS:
        print(f"  Season {season}...")
        try:
            # Base stats (pts, reb, ast, etc.)
            base = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable=LEAGUE_ID,
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
            ).get_data_frames()[0]
            base["season"] = season
            time.sleep(0.6)

            # Advanced stats (off_rating, def_rating, net_rating, pace)
            adv = leaguedashteamstats.LeagueDashTeamStats(
                league_id_nullable=LEAGUE_ID,
                season=season,
                season_type_all_star="Regular Season",
                per_mode_detailed="PerGame",
                measure_type_detailed_defense="Advanced",
            ).get_data_frames()[0]
            adv["season"] = season
            time.sleep(0.6)

            # Merge base + advanced on TEAM_ID
            df = base.merge(adv[["TEAM_ID", "OFF_RATING", "DEF_RATING", "NET_RATING", "PACE"]],
                            on="TEAM_ID", how="left")

            df.rename(columns={
                "GP": "gp", "W": "w", "L": "l",
                "PTS": "pts_pg", "REB": "reb_pg", "AST": "ast_pg",
                "STL": "stl_pg", "BLK": "blk_pg", "TOV": "tov_pg",
                "FG_PCT": "fg_pct", "FG3_PCT": "fg3_pct", "FT_PCT": "ft_pct",
                "TEAM_ID": "team_id", "TEAM_NAME": "team_name",
                "OFF_RATING": "off_rating", "DEF_RATING": "def_rating",
                "NET_RATING": "net_rating", "PACE": "pace",
            }, inplace=True)

            keep = ["season", "team_id", "team_name", "gp", "w", "l",
                    "pts_pg", "reb_pg", "ast_pg", "stl_pg", "blk_pg",
                    "tov_pg", "fg_pct", "fg3_pct", "ft_pct",
                    "off_rating", "def_rating", "net_rating", "pace"]
            df[[c for c in keep if c in df.columns]].to_sql(
                "team_stats", conn, if_exists="append", index=False
            )
            print(f"    {len(df)} teams (base + advanced)")
        except Exception as e:
            print(f"    Warning: {e}")
    conn.commit()
    conn.close()


def pull_player_game_logs():
    conn = get_conn()
    print("Pulling WNBA player game logs...")
    for season in SEASONS:
        print(f"  Season {season}...")
        try:
            logs = playergamelogs.PlayerGameLogs(
                league_id_nullable=LEAGUE_ID,
                season_nullable=season,
                season_type_nullable="Regular Season",
            )
            df = logs.get_data_frames()[0]
            df["season"] = season
            df.rename(columns={
                "PLAYER_ID": "player_id", "PLAYER_NAME": "player_name",
                "TEAM_ABBREVIATION": "team_abbr", "GAME_ID": "game_id",
                "GAME_DATE": "game_date", "MATCHUP": "matchup", "WL": "wl",
                "MIN": "min", "PTS": "pts", "REB": "reb", "AST": "ast",
                "STL": "stl", "BLK": "blk", "TOV": "tov", "FG3M": "fg3m",
                "FGM": "fgm", "FGA": "fga", "FG_PCT": "fg_pct",
                "FTM": "ftm", "FTA": "fta", "PLUS_MINUS": "plus_minus",
            }, inplace=True)
            keep = ["season", "player_id", "player_name", "team_abbr", "game_id",
                    "game_date", "matchup", "wl", "min", "pts", "reb", "ast",
                    "stl", "blk", "tov", "fg3m", "fgm", "fga", "fg_pct",
                    "ftm", "fta", "plus_minus"]
            df[[c for c in keep if c in df.columns]].to_sql(
                "player_game_logs", conn, if_exists="append", index=False
            )
            print(f"    {len(df)} player-game rows")
            time.sleep(0.6)
        except Exception as e:
            print(f"    Warning: {e}")
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    pull_team_game_logs()
    pull_team_stats()
    pull_player_game_logs()
    print("\nHistorical data pull complete.")
