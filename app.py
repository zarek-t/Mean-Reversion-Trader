import os
import threading

from flask import Flask, jsonify, request
from flask_cors import CORS

from backtest import WEB_TICKER_SYMBOLS, run_full_backtest, warmup_web_cache

app = Flask(__name__)

allowed_origins = os.environ.get("CORS_ORIGINS", "*")
if allowed_origins == "*":
    CORS(app, resources={r"/api/*": {"origins": "*"}})
else:
    CORS(app, resources={r"/api/*": {"origins": [o.strip() for o in allowed_origins.split(",")]}})


@app.route("/")
def index():
    return jsonify(
        {
            "service": "Mean Reversion Backtester API",
            "frontend": "Deploy the /frontend folder to Vercel",
            "health": "/health",
            "backtest": "POST /api/backtest",
        }
    )


@app.route("/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/tickers")
def api_tickers():
    from backtest import WEB_TICKERS

    return jsonify([{"symbol": s, "name": n} for s, n in WEB_TICKERS])


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.get_json(silent=True) or {}

    ticker = (data.get("ticker") or "").strip().upper()
    if not ticker:
        return jsonify({"error": "Please select a stock."}), 400

    if ticker not in WEB_TICKER_SYMBOLS:
        return jsonify({"error": "Please select one of the available stocks."}), 400

    try:
        window = int(data.get("window", 8))
        z_score = float(data.get("z_score", 0.5))
        period = data.get("period", "5y")
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid parameter values."}), 400

    if window < 2 or window > 52:
        return jsonify({"error": "Window must be between 2 and 52 weeks."}), 400

    if z_score <= 0 or z_score > 5:
        return jsonify({"error": "Z-score must be between 0.1 and 5."}), 400

    allowed_periods = {"1y", "2y", "3y", "5y", "10y", "max"}
    if period not in allowed_periods:
        return jsonify({"error": "Invalid time period."}), 400

    try:
        result = run_full_backtest(ticker, window, z_score, period=period)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        return jsonify({"error": "Backtest failed. Please try again."}), 500


def _start_cache_warmup():
    thread = threading.Thread(target=warmup_web_cache, daemon=True)
    thread.start()


_start_cache_warmup()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
