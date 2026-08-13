from flask import Flask, jsonify, render_template, request

from backtest import DEFAULT_TICKERS, run_full_backtest

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html", tickers=DEFAULT_TICKERS)


@app.route("/api/backtest", methods=["POST"])
def api_backtest():
    data = request.get_json(silent=True) or {}

    ticker = (data.get("ticker") or "").strip()
    if not ticker:
        return jsonify({"error": "Please enter a stock ticker."}), 400

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


if __name__ == "__main__":
    import os

    port = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "1") == "1"
    app.run(debug=debug, host="0.0.0.0", port=port)
