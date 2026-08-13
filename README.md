# Mean-Reversion-Trader

Mean reversion backtester: buy when price drops below its moving average, sell when it rises above.

## Web App (Vercel — no backend needed)

The web app runs **entirely in the browser** with bundled price data. No API, no yfinance calls, no cold starts.

### Deploy

1. Push to GitHub
2. [vercel.com](https://vercel.com) → Import repo → Deploy
3. Done — `vercel.json` serves the `/frontend` folder

### Refresh price data (optional, run locally)

```bash
pip install yfinance pandas
python scripts/fetch_data.py
git add frontend/data && git commit -m "Update price data" && git push
```

## CLI Optimizer (Python)

```bash
pip install -r requirements.txt
python Trader.py
```

Grid-searches parameters across 25 stocks using live yfinance data.

## Render API (optional)

`app.py` is a Flask API if you want server-side backtests, but the Vercel web app no longer depends on it.
