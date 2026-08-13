"""One-time script to refresh bundled price data for the static web app."""

import json
from pathlib import Path

import yfinance as yf

TICKERS = ["AAPL", "MSFT", "GOOGL", "JPM", "XOM", "COP", "JNJ", "TSLA", "SPY"]
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "frontend" / "data"


def fetch_ticker(ticker: str) -> dict:
    data = yf.download(ticker, period="5y", interval="1wk", auto_adjust=True, progress=False)
    if data.empty:
        raise RuntimeError(f"No data for {ticker}")

    if hasattr(data.columns, "levels"):
        data.columns = data.columns.get_level_values(0)

    col = "Close" if "Close" in data.columns else data.columns[0]
    close = data[col].astype(float).dropna()

    return {
        "ticker": ticker,
        "dates": [d.strftime("%Y-%m-%d") for d in close.index],
        "close": [round(v, 2) for v in close.tolist()],
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for ticker in TICKERS:
        payload = fetch_ticker(ticker)
        path = OUTPUT_DIR / f"{ticker}.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        print(f"Wrote {path} ({len(payload['close'])} weeks)")


if __name__ == "__main__":
    main()
