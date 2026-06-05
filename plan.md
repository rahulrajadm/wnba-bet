# WNBA Bet: AI-Powered WNBA Betting Decision Tool

A local + cloud tool that predicts WNBA game outcomes (moneyline, spread, totals) and player performance, compares model probabilities against live odds and pick'em lines, and surfaces +EV picks with confidence scores, risk profiles, and Kelly-sized stakes.

**Primary focus: game predictions (ML, spread, totals)**
**Secondary focus: player props (Points, Rebounds, Assists, PRA, Fantasy Score, etc.)**

---

## Texas Legal Platforms

Same as mlb-bet — all Texas-legal:

| Platform | Type | WNBA coverage |
|---|---|---|
| **PrizePicks** | Pick'em DFS | ✅ 2,000+ props daily |
| **Underdog Fantasy** | Pick'em DFS | ✅ Active |
| **Fliff** | Sweepstakes | ✅ ML, spread, totals |
| **Polymarket** | Prediction market | ✅ Game outcomes |
| **DraftKings Pick6** | Pick'em DFS | ✅ Active |
| **Chalkboard** | Pick'em DFS | ✅ Active |
| **Sleeper** | Pick'em / Prediction | ✅ Active |

---

## Output Per Pick

### Game predictions (ML, spread, totals)
| Field | Description |
|---|---|
| **Selection** | "Connecticut Sun ML", "Sun -4.5", "Over 163.5" |
| **Best platform** | Which platform has the best line/odds for this pick |
| **Model probability** | Model's estimated win/cover/hit probability |
| **Implied probability** | What the odds imply |
| **Edge** | Model prob − implied prob |
| **EV per $100** | Expected value |
| **Confidence tier** | STRONG / HIGH / MEDIUM / LOW |
| **Risk profile** | LOW / MEDIUM / HIGH |
| **Units** | Quarter-Kelly stake in units |

### Player props (pick'em)
Same as mlb-bet: model prediction vs line, edge vs 50% implied, confidence + risk + units.

---

## Game Prediction Model

Unlike mlb-bet (which was primarily player props), game predictions are the core here.

### Features
| Category | Features |
|---|---|
| Offense | Points per game, FG%, 3P%, FT%, pace, offensive rating |
| Defense | Points allowed, defensive rating, opponent FG% |
| Recent form | Last 5 and last 10 game averages (rolling) |
| Rest | Days since last game, back-to-back flag |
| Home/away | Home court win% splits |
| Head-to-head | Season H2H record and scoring margin |
| Roster | Key player availability (injury flag) |

### Models
- **Moneyline**: XGBoost binary classifier → P(home win)
- **Spread**: XGBoost regressor → predicted point differential → P(cover)
- **Totals**: XGBoost regressor → predicted total points → P(over)

### EV calculation (game picks)
- Odds API provides American odds → implied probability (with vig removed)
- Edge = model prob − true implied prob
- EV = model_prob × net_win − (1 − model_prob) × 100

---

## Player Props Model

Reuses the mlb-bet architecture adapted for basketball:
- Season per-game averages (current season + prior season weighted)
- Recent form blend (last 10 games at 55%, season at 45%)
- Opponent defensive rating adjustment (strong defense → fewer points)
- Pace adjustment (fast-paced games inflate counting stats)
- Poisson distribution → P(stat > line)

### Prop types covered
Points, Rebounds, Assists, Steals, Blocks, 3-PT Made, PRA (Pts+Rebs+Asts), Pts+Rebs, Pts+Asts, Rebs+Asts, Fantasy Score

---

## Data Sources

| Source | Data | Access |
|---|---|---|
| NBA Stats API (`stats.nba.com`) | WNBA game logs, team stats, player stats, schedules | Free, no key |
| The Odds API | WNBA moneyline, spread, totals across all books | Free tier (shared key with mlb-bet) |
| PrizePicks API | Live WNBA player prop lines | Free unofficial endpoint |
| Underdog API | Live WNBA player prop lines | Free unofficial endpoint |
| Basketball Reference | Historical season stats | Via `basketball_reference_web_scraper` or requests |

---

## Stack

| Layer | Tool |
|---|---|
| Historical stats | NBA Stats API + Basketball Reference |
| Live odds | The Odds API (`basketball_wnba`) |
| Live props | PrizePicks · Underdog |
| ML models | XGBoost + scikit-learn |
| Storage (local) | SQLite |
| Dashboard | Streamlit |

---

## Project Structure

```
wnba-bet/
├── plan.md
├── .env.example              # ODDS_API_KEY (shared with mlb-bet)
├── requirements.txt
├── data/
│   └── wnba_bet.db           # SQLite: game logs, team stats, player stats, odds, picks
├── pipeline/
│   ├── historical.py         # Pull historical WNBA game logs + team/player stats
│   ├── schedule.py           # Today's WNBA schedule from NBA Stats API
│   ├── odds.py               # Odds API: h2h, spreads, totals for basketball_wnba
│   ├── prizepicks.py         # PrizePicks WNBA prop lines (league_id=3)
│   ├── underdog.py           # Underdog WNBA prop lines
│   └── injuries.py           # NBA Stats API injury/availability report
├── models/
│   ├── train.py              # Feature engineering + train all models
│   ├── game.py               # Moneyline, spread, totals game prediction models
│   └── props.py              # Player prop prediction engine (Poisson)
├── analysis/
│   ├── ev.py                 # EV calculation (traditional odds + pick'em)
│   ├── confidence.py         # Confidence tier assignment
│   ├── risk.py               # Risk profile assignment
│   └── kelly.py              # Fractional Kelly sizing
├── picks/
│   └── engine.py             # Assembles game + prop picks, ranks by EV
├── ui/
│   ├── app.py                # Local Streamlit dashboard (SQLite)
│   └── app_cloud.py          # Cloud Streamlit dashboard (in-memory)
└── utils/
    └── db.py                 # SQLite schema + helpers
```

---

## Build Order

### Phase 1 — Data Pipeline
- [ ] `utils/db.py` — SQLite schema: games, team_stats, player_stats, odds, prop_lines, picks
- [ ] `pipeline/historical.py` — pull 2–3 seasons of WNBA game logs + team/player stats
- [ ] `pipeline/schedule.py` — today's WNBA schedule + matchups from NBA Stats API
- [ ] `pipeline/odds.py` — live ML, spread, totals from Odds API
- [ ] `pipeline/prizepicks.py` — live WNBA prop lines (league_id=3)
- [ ] `pipeline/underdog.py` — live WNBA prop lines

### Phase 2 — Game Models (primary focus)
- [ ] `models/train.py` — feature engineering: team rolling averages, pace, H2H, rest
- [ ] `models/game.py` — train moneyline, spread, totals XGBoost models; serialize to disk

### Phase 3 — Props Model
- [ ] `models/props.py` — per-player season + recent form rates; Poisson probability vs line
- [ ] Opponent defensive rating adjustment
- [ ] Pace adjustment (fast pace = more possessions = more stats)

### Phase 4 — Analysis Engine
- [ ] `analysis/ev.py` — EV for traditional odds (game picks) + pick'em multipliers (props)
- [ ] `analysis/confidence.py` — confidence tiers
- [ ] `analysis/risk.py` — risk profiles
- [ ] `analysis/kelly.py` — quarter-Kelly sizing
- [ ] `picks/engine.py` — assemble + rank all picks; tag best platform

### Phase 5 — UI
- [ ] `ui/app.py` — local Streamlit: Game Predictions tab (new), Player Props, High Interest, Bankroll
- [ ] `ui/app_cloud.py` — cloud version with passcode refresh + timestamp

### Phase 6 — Deploy
- [ ] GitHub repo
- [ ] Streamlit Cloud deployment

---

## Key Differences from MLB Bet

| | MLB Bet | WNBA Bet |
|---|---|---|
| Primary focus | Player props | **Game predictions** |
| Odds source | Not used on cloud | **Core data source** |
| Stat model | Poisson (counting stats) | Poisson (props) + XGBoost (game scores) |
| Park factors | Yes | No (indoor arenas, no weather) |
| Arsenal factor | Yes (pitchers) | No equivalent |
| Platoon splits | Yes | No equivalent |
| Pace adjustment | No | **Yes — pace drives counting stats** |
| Opponent defense | Via K-rate | **Via defensive rating** |

---

## Setup (after build)

```bash
cd wnba-bet
pip install -r requirements.txt
cp .env.example .env
python pipeline/historical.py    # one-time: pull historical data
python models/train.py           # one-time: train models
streamlit run ui/app.py          # daily: run dashboard
```

---

## Built by

Rahul Raja Durai Murugan
BS Biomedical Engineering, UT Austin · MS Engineering Data Science & AI, University of Houston (incoming)
