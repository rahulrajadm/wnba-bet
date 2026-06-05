#!/bin/bash
# WNBA Bet — daily launcher
cd "$(dirname "$0")"

echo "🏀 WNBA Bet starting..."
echo ""

echo "📅 Fetching today's schedule..."
python pipeline/schedule.py

echo "📊 Fetching PrizePicks lines..."
python pipeline/prizepicks.py

echo "📊 Fetching Underdog lines..."
python pipeline/underdog.py

echo "📈 Fetching odds (ML, spread, totals)..."
python pipeline/odds.py

echo ""
echo "✅ Data ready. Opening dashboard..."
echo ""

streamlit run ui/app.py
