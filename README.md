# Mean-Reversion-Trader

Buying low and selling high.

This project implements a basic mean reversion trading strategy: buy when a stock drops below a statistically significant threshold relative to its moving average, and sell when it rises above it.

The goal is to exploit the natural tendency of stock prices to revert toward their historical mean over time. It uses historical price data, moving averages, and Z-scores to identify entry and exit points.

## Web App

Try different stocks and parameters in the browser with interactive charts and full performance stats.

### Setup

```bash
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

For local production-style testing:

```bash
gunicorn app:app --bind 0.0.0.0:5000 --workers 1 --threads 2 --timeout 120
```

### Deploy to the web (Render)

1. Push this repo to GitHub.
2. Sign up at [render.com](https://render.com) and click **New → Blueprint**.
3. Connect the GitHub repo — Render will read `render.yaml` automatically.
4. Click **Apply** to deploy. You'll get a public URL like `https://mean-reversion-trader.onrender.com`.

**Manual setup** (if you skip the blueprint):

- **New → Web Service** → connect repo
- **Build command:** `pip install -r requirements.txt`
- **Start command:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

### Deploy to Railway

1. Push to GitHub and sign up at [railway.app](https://railway.app).
2. **New Project → Deploy from GitHub repo** and select this repo.
3. Railway detects the `Procfile` and deploys automatically.

### Features

- Pick any stock ticker (25 presets included, or enter your own)
- Adjust moving average window, Z-score threshold, and time period
- Interactive charts comparing strategy vs buy-and-hold vs S&P 500 (SPY)
- Full stats: returns, Sharpe, Calmar, max drawdown, win rate, trades, and more

## CLI Optimizer

The original command-line optimizer is still available:

```bash
python Trader.py
```

This grid-searches parameter combinations across 25 stocks and prints optimization results.
