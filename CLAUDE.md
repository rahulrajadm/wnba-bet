# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ML-driven WNBA betting tool: XGBoost game models (moneyline/spread/totals) + a distribution-based player-props model, compared against live odds and pick'em lines to surface +EV picks with confidence tiers and Kelly stakes. Deployed at https://wnba-bet.streamlit.app/.

## Commands

```bash
# One-time local setup
pip install -r requirements.txt
cp .env.example .env              # add ODDS_API_KEY
python pipeline/historical.py     # pull 4 seasons into data/wnba_bet.db (~5 min)
python models/train.py            # train + save models to data/models/

# Daily local run (refreshes SQLite, grades yesterday's picks, opens local dashboard)
./start.sh

# Run the local dashboard directly
streamlit run ui/app.py

# There is no test suite. Most modules have __main__ smoke blocks — run them
# directly to verify a change, e.g.:
python models/props.py            # prop predictions from local DB
python picks/engine.py            # full pick build from local DB
python pipeline/prizepicks.py     # live line fetch + stat-type counts
```

The props pipeline prints filter diagnostics to stderr (`[props] lines=... no_stat=... passed=...` and `unknown_stat_types`); check these after any change to stat mapping or filters.

## Two entry points, two data paths

- **`ui/app_cloud.py`** — the deployed app (Streamlit Community Cloud). No SQLite: all data is fetched in-memory on a passcode-gated refresh (`REFRESH_CODE` in `st.secrets`) and held in an `st.cache_resource` store that survives page reloads but is wiped on redeploy. Game logs come from the unofficial ESPN API (`pipeline/cloud_data.py`) because stats.nba.com blocks cloud IPs.
- **`ui/app.py`** — local dashboard backed by SQLite (`data/wnba_bet.db`), populated by the individual `pipeline/*.py` scripts that `start.sh` runs.

Because of this split, nearly every model/metrics function takes an optional DataFrame (`game_logs_df`, `player_logs_df`, `team_logs_df`, `lines_data`, `odds_data`) and falls back to SQLite when it's `None`. **Preserve this convention when changing signatures — a change that only handles one path silently breaks the other app.**

Deployment = push to `main`; Streamlit Cloud auto-redeploys. After redeploy the data store is empty until someone enters the passcode and hits Refresh. `app_cloud.py` deletes `__pycache__` dirs at startup because stale `.pyc` files survive Streamlit Cloud deployments. It also purges stale project modules from `sys.modules` when the loaded `analysis/explain.py` predates its `SCHEMA_VERSION` — Streamlit Cloud can hot-swap source into a running process. **Bump `SCHEMA_VERSION` (and `_REQUIRED_SCHEMA` in `app_cloud.py`) whenever the UI starts depending on new symbols from shared modules**, or the next redeploy crashes with an ImportError.

## Data sources and costs

- **The Odds API** (`ODDS_API_KEY` in `.env` locally, `st.secrets` on cloud): the only metered source. One refresh = one call (3 credits, ~500/month free tier). Schedule is derived from the same response — don't add separate schedule calls.
- **PrizePicks, Underdog, ESPN**: free/unofficial, no keys. ESPN box-score fetches are parallelized and failure counts surface in the UI.
- **PrizePicks is behind DataDome bot protection** (403 on plain requests). The two apps handle it differently on purpose:
  - **Local** (`ui/app.py` → `get_prizepicks_lines()`): mints a DataDome cookie in a *headed* browser once, caches it to `data/.pp_cookie.json`, and reuses it via `curl_cffi`. The browser only launches when the cookie is stale; normal refreshes are a fast HTTP call. Needs `requirements-local.txt` (`curl_cffi`, `playwright` + `playwright install chromium`) — a desktop session, not a headless server. Headless browsers are detected and get a challenge cookie that still 403s.
  - **Cloud** (`ui/app_cloud.py` → `fetch_wnba_lines()`): left as plain requests, which 403s on Streamlit Cloud's datacenter IP (DataDome hard-blocks datacenter IPs even with a valid cookie). The cloud app already treats that as "no PrizePicks" and runs on **Underdog props only**. Don't add the browser workaround to the cloud path or `requirements.txt`.
  - Underdog stamps both its two-way "balanced" lines and one-way "alternate" lines with `odds_type="standard"`, so the `models/props.py` dedupe prefers the balanced line (`allowed_direction is None`) or the model would surface arbitrary alternates instead of the standard line.

## Prediction pipeline (the big picture)

```
odds + schedule (Odds API)      prop lines (PrizePicks/Underdog)     game logs (ESPN or SQLite)
        │                                  │                                  │
        └──────────────► picks/engine.py: build_picks() ◄────────────────────┘
                          ├─ models/game.py   (game markets)
                          └─ models/props.py  (player props)
                          then EV/confidence/risk/Kelly from analysis/
```

Key conventions that span files — read these before touching probabilities:

1. **The spread regressor is the single source of truth for game probabilities.** Win prob = `norm.cdf(pred_diff / spread_std)`; the trained ML classifier exists but is deliberately not used (it produced contradictory numbers vs. cover probability). `spread_std`/`totals_std` come from holdout residuals saved in `data/models/game_calibration.pkl` at train time — never hardcode tighter values.
2. **Market anchoring**: predictions are shrunk toward the market line (`MODEL_WEIGHT = 0.6` in `picks/engine.py`) before probabilities are computed. The same anchoring is duplicated in `app_cloud.py`'s `_model_views` so the "pass" display matches what the pick filter used — keep them in sync.
3. **Edges are measured against real break-evens, not 0.50**: game picks against de-vigged book odds (`analysis/ev.py:remove_vig`), props against the pick'em slip break-even (~0.577 for a 2-pick 3x, `breakeven_prob`).
4. **Raw model margins are compressed** (regression to the mean), so vs. the market the model systematically leans underdog/under. Anchoring softens but doesn't remove this — treat "every pick is a dog" output as expected model behavior, not a bug.
5. Retraining (`models/train.py`) requires the local SQLite DB and uses a chronological 80/20 split; it refits on all data after computing holdout calibration. Trained `.pkl` files are committed in `data/models/`.

## Stat-name and team-name sync (easy to break)

- Platform stat names differ: PrizePicks uses `"Pts+Rebs"`, Underdog uses `"Points + Rebounds"` (spaces). Any new stat name must be added in **three places together**: `STAT_MAP` (models/props.py), `LESS_MIN_LINE`/`MORE_MIN_LINE` (picks/engine.py), and the variance sets in `analysis/risk.py`. Unmapped names are silently skipped (visible in the `[props]` stderr counters).
- PrizePicks team abbreviations are mapped in `_TEAM_ABBR` (pipeline/prizepicks.py) and the map is imported by the UI for date lookups. PrizePicks has changed abbreviations before (LVA vs LAS, NYL vs NY) — an unmapped abbreviation causes missing dates in the UI *and* a wrong opponent-defense adjustment in props. Multi-team labels like `PHX/SEA` belong to multi-player combo props, which the model intentionally skips.
- Team/player matching elsewhere is fuzzy (exact match, then last-word contains). Player matching in props deliberately refuses ambiguous last-name matches to avoid blending two players' logs.
- `odds_type` on prop lines: PrizePicks sends `standard`/`goblin`/`demon` (goblin/demon are More-only); Underdog lines are always `standard`. Everything downstream (filters, dedupe keys, UI labels) assumes this field exists.

## Known modeling caveats (deliberate, deferred)

- `breakeven_prob` assumes the standard 3x multiplier for all odds types; PrizePicks pays less when goblins are in a slip, so goblin edges are overstated and demon edges understated. A per-odds-type multiplier table is planned but not implemented.
- Prop Kelly stakes size each leg as a standalone 3x bet (`kelly_stake(model_prob, 3.0)`); a real slip needs all legs, so prop stakes are aggressive. Props' "EV / $100" is per-leg edge ×100, not slip EV.
