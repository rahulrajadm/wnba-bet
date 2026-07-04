---
name: analyze-app
description: Full review of the WNBA-bet app — launch and drive both dashboards in a browser, check UI consistency between the local and cloud apps, verify the prediction math end to end, and audit security. Use when the user asks to analyze, review, or audit the app.
---

Analyze the entire WNBA-bet Streamlit app: launch the web application, check for UI
consistencies, analyze code to ensure accurate analysis and predictions, and ensure
security is maintained. Read CLAUDE.md first — it documents the two-entry-point split
and the conventions referenced below.

## 1. Launch and drive (not just start)

- **Local app**: `streamlit run ui/app.py --server.port 8599 --server.headless true`
  in the background, then drive it with the Playwright browser tools. If prop lines or
  odds are stale (the freshness banner shows it), re-fetch the free sources first:
  `python pipeline/prizepicks.py && python pipeline/underdog.py`. Do NOT press
  "Refresh All Data" or call `pipeline/odds.py` / `get_today_games()` without asking —
  those spend metered Odds API credits (sidebar shows the remaining balance).
- **Cloud app** (what production runs): `printf 'REFRESH_CODE = "0000"\n' >
  .streamlit/secrets.toml`, run `ui/app_cloud.py` on port 8600, refresh through the
  browser with passcode 0000 (this consumes 3 Odds API credits — mention it), and
  **delete `.streamlit/secrets.toml` when done**.
- Click through **all six tabs** in each app with real data, at desktop (1440px) and
  phone (390px) widths. In the Ask Why tab, exercise a prop question, a follow-up
  what-if, a game question, and a ranking question.
- Streamlit servers cache Python modules: after editing shared code, restart the
  server and clear `__pycache__` before re-testing.

## 2. UI consistency

- `ui/app.py` and `ui/app_cloud.py` duplicate display code on purpose. Diff them
  tab-by-tab: same columns and header names, chips, goblin/demon markers, edge
  coloring, empty states, and Ask Why behavior. Divergence is a finding unless it's a
  documented platform difference (date filter, line types, injuries, passcode refresh
  are cloud-only; graded results and API-credit display are local-only).
- No table may clip columns off-screen at 1440px; metric rows must wrap on mobile.
- The freshness banner must reflect real fetch times, and stale data must be flagged.
- The Game Predictions "model view / pass" numbers must match what the pick filter
  used (same anchoring as `picks/engine.py` — `MODEL_WEIGHT`).

## 3. Prediction accuracy

- Trace displayed numbers to the pipeline: Ask Why explain payloads must equal what
  `build_picks()` computed; spot-check one prop by hand (baseline → opponent adj →
  pace → distribution → break-even) and one game market (raw XGBoost → 60/40 market
  anchoring → `norm.cdf` with calibration std).
- Calibration stds come from `data/models/game_calibration.pkl` — flag any hardcoded
  tighter values. Game edges must be measured against de-vigged odds
  (`analysis/ev.py:remove_vig`); prop edges against the pick'em break-even (~0.577).
- Run the smoke blocks (`python models/props.py`, `python picks/engine.py`) and read
  the `[props]` stderr counters — a jump in `no_stat`/`direction`/`no_rate` or new
  `unknown_stat_types` entries means a mapping broke.
- Check the three-way stat-name sync (`STAT_MAP`, `LESS_MIN_LINE`/`MORE_MIN_LINE`,
  `analysis/risk.py`) and the PrizePicks `_TEAM_ABBR` map against live line data.
- Both data paths must behave identically: SQLite fallback vs in-memory DataFrames
  (a change that only handles one path silently breaks the other app).
- Known deferred caveats (goblin/demon multiplier, per-leg Kelly sizing) are NOT
  findings — but verify their UI disclaimers are still shown.

## 4. Security

- Secrets: `.env`, `.streamlit/secrets.toml`, and `data/wnba_bet.db` stay gitignored
  and uncommitted; `git log` must be free of keys; the Odds API key must never appear
  in the UI, error states, or tracebacks (check `st.exception` paths).
- The deployed app is public: everything a visitor can do without the passcode must
  be spend-free (the only metered action is the passcode-gated refresh) and the
  passcode must be compared against `st.secrets`, never a literal.
- Free-text inputs (Ask Why, search boxes) must not reach SQL, eval, or shell — and
  user-typed content must not be interpolated into unescaped HTML
  (`unsafe_allow_html=True` blocks take only app-generated values).
- SQL must be parameterized (`?` placeholders) anywhere user- or platform-derived
  strings meet the database.

## Report format

Lead with a verdict, then findings ranked by severity (crash/money-losing → wrong
numbers → UX → nits), each with file:line and a concrete failure scenario. Fixes only
if asked — this command is a review, not a repair. End by restoring state: stop test
servers you started, delete the temp secrets file, and leave the repo tree clean.
