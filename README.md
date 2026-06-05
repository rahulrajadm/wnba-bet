# WNBA Bet: AI-Powered WNBA Betting Decision Tool

An ML-driven tool that predicts WNBA game outcomes (moneyline, spread, totals) and player performance, comparing model probabilities against live odds and pick'em lines to surface +EV picks with confidence scores, risk profiles, and Kelly-sized stakes.

**Primary focus: game predictions (ML, spread, totals)**
**Secondary focus: player props (Points, Rebounds, Assists, PRA, Fantasy Score, etc.)**

---

## Live Demo

🚀 **[bet-wnba.streamlit.app](https://bet-wnba.streamlit.app)**

---

## What it does

1. **Fetches live data** — WNBA schedule, PrizePicks + Underdog props, Odds API game lines
2. **Predicts game outcomes** using XGBoost models trained on 4 seasons of WNBA data:
   - Moneyline (67% accuracy)
   - Spread / point differential (MAE 9.4 pts)
   - Totals (MAE 14.6 pts)
3. **Predicts player props** using per-player stat models with:
   - Season averages + recent 10-game form (55/45 blend)
   - Opponent defensive rating adjustment
   - Game pace adjustment
4. **Computes edge** — model probability vs bookmaker implied probability
5. **Ranks picks** by confidence tier (STRONG / HIGH / MEDIUM / LOW) and risk profile
6. **Sizes stakes** using fractional Kelly criterion (0.25×) in units

---

## Model Factors

### Game Predictions
| Factor | Description |
|---|---|
| Rolling team averages | pts, FG%, rebounds, assists, turnovers over last 5 + 10 games |
| Win % | Last 10 games |
| Rest days | Actual days since last game (not hardcoded) |
| Home court | Explicit home advantage feature |
| Rest advantage | Days rest differential between teams |

### Player Props
| Factor | Description |
|---|---|
| Season average | Per-game rates for current + prior seasons |
| Recent form | Last 10 games weighted 55% vs season 45% |
| Opponent defense | Actual points allowed per game by the opponent |
| Pace adjustment | Expected game pace vs league average (82 pts/team) |

---

## Stack

| Layer | Tool |
|---|---|
| Historical stats | NBA Stats API (WNBA, `league_id=10`) |
| Live odds | The Odds API (`basketball_wnba`) |
| Live props | PrizePicks · Underdog Fantasy |
| Game models | XGBoost (classifier + regressors) |
| Props model | Poisson / Normal distribution |
| Dashboard | Streamlit |
| Storage (local) | SQLite |

---

## Dashboard

| Tab | Content |
|---|---|
| 🔥 Top Picks | Game picks + high-interest props ranked by EV |
| 🏀 Game Predictions | Per-game ML / spread / totals breakdown |
| 🎯 Player Props | Props with stat filter + season/recent context |
| 📊 Platform Comparison | PrizePicks vs Underdog side-by-side |
| 💰 Bankroll Tracker | Kelly stakes in units + slip builder |

---

## Local Setup

```bash
git clone https://github.com/rahulrajadm/wnba-bet.git
cd wnba-bet
pip install -r requirements.txt
cp .env.example .env   # add your free Odds API key
python pipeline/historical.py   # one-time: pull 4 seasons of data (~5 min)
python models/train.py          # one-time: train game models
./start.sh                      # daily: refresh data + open dashboard
```

Or with the alias (add to `.zshrc`):
```bash
alias wnba-bet="/path/to/wnba-bet/start.sh"
```

---

## Project Structure

```
wnba-bet/
├── pipeline/         # Data ingestion (schedule, odds, PrizePicks, Underdog, team metrics)
├── models/           # Game prediction models + player prop engine
├── analysis/         # EV, confidence, risk, Kelly
├── picks/            # Pick assembly and ranking engine
├── ui/
│   ├── app.py        # Local Streamlit dashboard (SQLite)
│   └── app_cloud.py  # Cloud dashboard (in-memory, passcode refresh)
├── utils/            # SQLite schema + helpers
├── data/models/      # Pre-trained model files (committed)
└── start.sh          # Daily launcher
```

---

## Built by

**Rahul Raja Durai Murugan**

BS Biomedical Engineering, UT Austin · MS Engineering Data Science & AI, University of Houston (incoming)

[LinkedIn](https://linkedin.com/in/rahulrajadm) · [GitHub](https://github.com/rahulrajadm) · rahulrdm13@gmail.com

---

## Disclaimer

For informational and educational purposes. Designed for legal pick'em DFS platforms and prediction markets. Gamble responsibly.

## License

MIT © 2026 Rahul Raja Durai Murugan
