import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "../data/wnba_bet.db")


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return sqlite3.connect(DB_PATH)


def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.executescript("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            date TEXT,
            home_team TEXT,
            away_team TEXT,
            home_team_id TEXT,
            away_team_id TEXT,
            game_time TEXT,
            season TEXT
        );

        CREATE TABLE IF NOT EXISTS team_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            season TEXT,
            team_id TEXT,
            team_name TEXT,
            team_abbr TEXT,
            game_date TEXT,
            matchup TEXT,
            wl TEXT,
            pts INTEGER,
            fg_pct REAL,
            fg3_pct REAL,
            ft_pct REAL,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            plus_minus REAL
        );

        CREATE TABLE IF NOT EXISTS team_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT,
            team_id TEXT,
            team_name TEXT,
            gp INTEGER,
            w INTEGER,
            l INTEGER,
            pts_pg REAL,
            opp_pts_pg REAL,
            reb_pg REAL,
            ast_pg REAL,
            stl_pg REAL,
            blk_pg REAL,
            tov_pg REAL,
            fg_pct REAL,
            fg3_pct REAL,
            ft_pct REAL,
            off_rating REAL,
            def_rating REAL,
            net_rating REAL,
            pace REAL
        );

        CREATE TABLE IF NOT EXISTS player_game_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            season TEXT,
            player_id TEXT,
            player_name TEXT,
            team_abbr TEXT,
            game_id TEXT,
            game_date TEXT,
            matchup TEXT,
            wl TEXT,
            min REAL,
            pts INTEGER,
            reb INTEGER,
            ast INTEGER,
            stl INTEGER,
            blk INTEGER,
            tov INTEGER,
            fg3m INTEGER,
            fgm INTEGER,
            fga INTEGER,
            fg_pct REAL,
            ftm INTEGER,
            fta INTEGER,
            plus_minus REAL
        );

        CREATE TABLE IF NOT EXISTS game_odds (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            platform TEXT,
            game_id TEXT,
            home_team TEXT,
            away_team TEXT,
            market TEXT,
            home_odds REAL,
            away_odds REAL,
            home_spread REAL,
            away_spread REAL,
            over_odds REAL,
            under_odds REAL,
            total_line REAL
        );

        CREATE TABLE IF NOT EXISTS prop_lines (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fetched_at TEXT,
            platform TEXT,
            game_id TEXT,
            player_name TEXT,
            player_team TEXT,
            stat_type TEXT,
            line REAL,
            more_odds REAL,
            less_odds REAL
        );

        CREATE TABLE IF NOT EXISTS picks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            generated_at TEXT,
            pick_type TEXT,
            selection TEXT,
            best_platform TEXT,
            model_prob REAL,
            implied_prob REAL,
            edge REAL,
            ev_per_100 REAL,
            confidence_tier TEXT,
            risk_profile TEXT,
            kelly_pct REAL,
            units REAL,
            details TEXT
        );
    """)

    conn.commit()
    conn.close()
    print("WNBA database initialized.")


if __name__ == "__main__":
    init_db()
