# Mean-Reversion-Trader

Buying low and selling high.

Mean reversion backtester: buy when price drops below its moving average, sell when it rises above.

## Architecture

| Part | Host | What it does |
|------|------|--------------|
| **Frontend** | Vercel | Static HTML/JS — fast, reliable UI |
| **API** | Render | Python backtests + chart images |

## Deploy frontend (Vercel)

1. Push repo to GitHub
2. [vercel.com](https://vercel.com) → **Import Project** → select repo
3. Vercel reads `vercel.json` automatically (serves `/frontend`, proxies `/api` to Render)
4. Deploy — you get a URL like `https://mean-reversion-trader.vercel.app`

No env vars needed unless your Render URL changes (update `vercel.json` rewrite).

## Deploy API (Render)

1. [render.com](https://render.com) → **Web Service** from same repo
2. **Build:** `pip install -r requirements.txt`
3. **Start:** `gunicorn app:app --bind 0.0.0.0:$PORT --workers 1 --threads 2 --timeout 120`

Render runs the Python API only. The old Flask HTML UI is replaced by the Vercel frontend.

## Local development

**API:**
```bash
pip install -r requirements.txt
python app.py
```

**Frontend:** open `frontend/index.html` via a local server, and set `frontend/config.js`:
```javascript
window.API_URL = "http://127.0.0.1:5000";
```

## CLI Optimizer

```bash
python Trader.py
```

Grid-searches parameters across 25 stocks.
