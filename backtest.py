import base64
import io
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "TSLA",
    "JPM", "BAC", "WFC", "GS", "MS",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "JNJ", "PFE", "UNH", "ABBV", "MRK",
    "PG", "KO", "PEP", "WMT", "HD",
]

WEB_TICKERS = [
    ("AAPL", "Apple"),
    ("MSFT", "Microsoft"),
    ("GOOGL", "Alphabet"),
    ("JPM", "JPMorgan"),
    ("XOM", "Exxon Mobil"),
    ("COP", "ConocoPhillips"),
    ("JNJ", "Johnson & Johnson"),
    ("TSLA", "Tesla"),
]

WEB_TICKER_SYMBOLS = {symbol for symbol, _ in WEB_TICKERS}

START_CASH = 10000
TRANSACTION_COST = 0.001
DEFAULT_PERIOD = "5y"

_data_cache = {}


def download_and_prepare_data(ticker, period="5y", interval="1wk"):
    """Download and prepare weekly close prices for a ticker."""
    cache_key = (ticker.upper(), period, interval)
    if cache_key in _data_cache:
        return _data_cache[cache_key].copy()

    try:
        data = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )

        if data.empty:
            return None

        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
        close = data[price_col].astype(float).dropna()

        if len(close) < 50:
            return None

        _data_cache[cache_key] = close.copy()
        return close
    except Exception:
        return None


def warmup_web_cache(period=DEFAULT_PERIOD):
    """Pre-download web app tickers so backtests stay fast."""
    for symbol, _ in WEB_TICKERS:
        download_and_prepare_data(symbol, period=period)
    download_and_prepare_data("SPY", period=period)


def calculate_signals(close, window, z_buy, z_sell):
    """Calculate trading signals for given parameters."""
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()

    valid_idx = ma.index.intersection(std.index).intersection(close.index)
    close = close.loc[valid_idx]
    ma = ma.loc[valid_idx]
    std = std.loc[valid_idx]

    z = (close - ma) / std

    holding = False
    signals = []

    for _, row in pd.DataFrame({"Z": z}).iterrows():
        if not holding and row["Z"] < z_buy:
            signals.append("BUY")
            holding = True
        elif holding and row["Z"] > z_sell:
            signals.append("SELL")
            holding = False
        else:
            signals.append("HOLD")

    return close, ma, std, z, signals


def backtest_strategy(close, signals, start_cash, transaction_cost=0):
    """Backtest the strategy with transaction costs."""
    cash = start_cash
    shares = 0
    portfolio_values = []
    buy_dates = []
    sell_dates = []
    trades = 0

    for i, (date, price) in enumerate(close.items()):
        signal = signals[i]
        if signal == "BUY" and cash >= price:
            available_cash = cash * (1 - transaction_cost)
            shares = available_cash // price
            cash -= shares * price * (1 + transaction_cost)
            buy_dates.append(date)
            trades += 1
        elif signal == "SELL" and shares > 0:
            cash += shares * price * (1 - transaction_cost)
            shares = 0
            sell_dates.append(date)
            trades += 1
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)

    return portfolio_values, buy_dates, sell_dates, trades


def calculate_enhanced_metrics(portfolio_values, close, start_cash, trades=0):
    """Calculate comprehensive performance metrics."""
    final_value = portfolio_values[-1]
    total_return = (final_value / start_cash - 1) * 100

    bh_shares = start_cash // close.iloc[0]
    bh_cash = start_cash - bh_shares * close.iloc[0]
    bh_value = bh_cash + bh_shares * close.iloc[-1]
    bh_return = (bh_value / start_cash - 1) * 100

    returns = pd.Series(portfolio_values).pct_change().dropna()

    volatility = returns.std() * np.sqrt(52) * 100
    sharpe = (
        (returns.mean() * 52) / (returns.std() * np.sqrt(52))
        if returns.std() > 0
        else 0
    )

    peak = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - peak) / peak
    max_drawdown = drawdown.min() * 100

    calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0

    positive_periods = (returns > 0).sum()
    total_periods = len(returns)
    win_rate = (positive_periods / total_periods) * 100 if total_periods > 0 else 0

    avg_trade_return = total_return / trades if trades > 0 else 0

    return {
        "final_value": final_value,
        "total_return": total_return,
        "bh_return": bh_return,
        "excess_return": total_return - bh_return,
        "volatility": volatility,
        "sharpe": sharpe,
        "calmar": calmar,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "trades": trades,
        "avg_trade_return": avg_trade_return,
    }


def _buy_and_hold_series(close, start_cash):
    shares = start_cash // close.iloc[0]
    cash = start_cash - shares * close.iloc[0]
    return [cash + shares * price for price in close]


def _return_pct(series, start_cash):
    if not series:
        return 0.0
    return (series[-1] / start_cash - 1) * 100


def _serialize_date(date):
    return date.strftime("%Y-%m-%d")


def _fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=120, bbox_inches="tight", facecolor="#0f1419")
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


def generate_chart_images(
    ticker,
    window,
    z_score,
    close_clean,
    ma,
    portfolio_values,
    bh_portfolio_values,
    spy_portfolio_values,
    buy_dates,
    sell_dates,
    start_cash,
):
    """Render backtest charts server-side as PNG images."""
    dates = close_clean.index

    fig1, ax1 = plt.subplots(figsize=(10, 4))
    fig1.patch.set_facecolor("#0f1419")
    ax1.set_facecolor("#1a2332")
    ax1.plot(dates, close_clean, label="Price", color="#e8edf4", linewidth=1.5)
    ax1.plot(dates, ma, label=f"{window}-Week MA", color="#8b9cb3", linestyle="--")
    if buy_dates:
        ax1.scatter(
            buy_dates,
            close_clean.loc[buy_dates],
            marker="^",
            color="#22c55e",
            s=70,
            label="Buy",
            zorder=5,
        )
    if sell_dates:
        ax1.scatter(
            sell_dates,
            close_clean.loc[sell_dates],
            marker="v",
            color="#ef4444",
            s=70,
            label="Sell",
            zorder=5,
        )
    ax1.set_title(
        f"{ticker} Price & Signals (Z=±{z_score})",
        color="#e8edf4",
        fontsize=12,
    )
    ax1.tick_params(colors="#8b9cb3")
    ax1.grid(True, alpha=0.2, color="#2d3a4f")
    ax1.legend(facecolor="#1a2332", edgecolor="#2d3a4f", labelcolor="#e8edf4")
    for spine in ax1.spines.values():
        spine.set_color("#2d3a4f")
    price_chart = _fig_to_base64(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    fig2.patch.set_facecolor("#0f1419")
    ax2.set_facecolor("#1a2332")
    ax2.plot(dates, portfolio_values, label="Strategy", color="#3b82f6", linewidth=2)
    ax2.plot(dates, bh_portfolio_values, label="Buy & Hold", color="#f97316", linewidth=2)
    if spy_portfolio_values:
        ax2.plot(dates, spy_portfolio_values, label="S&P 500 (SPY)", color="#a855f7", linewidth=2)
    ax2.axhline(y=start_cash, color="#8b9cb3", linestyle=":", alpha=0.7, label="Start $10k")
    ax2.set_title("Portfolio Comparison", color="#e8edf4", fontsize=12)
    ax2.set_ylabel("Value ($)", color="#8b9cb3")
    ax2.tick_params(colors="#8b9cb3")
    ax2.grid(True, alpha=0.2, color="#2d3a4f")
    ax2.legend(facecolor="#1a2332", edgecolor="#2d3a4f", labelcolor="#e8edf4")
    for spine in ax2.spines.values():
        spine.set_color("#2d3a4f")
    portfolio_chart = _fig_to_base64(fig2)

    return {"price": price_chart, "portfolio": portfolio_chart}


def run_full_backtest(
    ticker,
    window,
    z_score,
    period=DEFAULT_PERIOD,
    start_cash=START_CASH,
    transaction_cost=TRANSACTION_COST,
):
    """Run a complete backtest and return JSON-serializable results."""
    ticker = ticker.upper().strip()
    close = download_and_prepare_data(ticker, period=period)
    if close is None:
        raise ValueError(
            f"Could not download enough data for '{ticker}'. "
            "Check the ticker symbol and try again."
        )

    z_buy = -z_score
    z_sell = z_score

    close_clean, ma, std, z, signals = calculate_signals(
        close, window, z_buy, z_sell
    )
    portfolio_values, buy_dates, sell_dates, trades = backtest_strategy(
        close_clean, signals, start_cash, transaction_cost
    )
    metrics = calculate_enhanced_metrics(
        portfolio_values, close_clean, start_cash, trades
    )

    bh_portfolio_values = _buy_and_hold_series(close_clean, start_cash)

    spy_data = download_and_prepare_data("SPY", period=period)
    spy_portfolio_values = []
    if spy_data is not None:
        spy_aligned = spy_data.reindex(close_clean.index, method="ffill")
        if not spy_aligned.isna().all():
            spy_portfolio_values = _buy_and_hold_series(spy_aligned, start_cash)

    spy_return = _return_pct(spy_portfolio_values, start_cash)
    metrics["spy_return"] = spy_return
    metrics["excess_vs_spy"] = metrics["total_return"] - spy_return

    charts = generate_chart_images(
        ticker,
        window,
        z_score,
        close_clean,
        ma,
        portfolio_values,
        bh_portfolio_values,
        spy_portfolio_values,
        buy_dates,
        sell_dates,
        start_cash,
    )

    return {
        "ticker": ticker,
        "window": window,
        "z_score": z_score,
        "period": period,
        "start_cash": start_cash,
        "metrics": {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()},
        "charts": charts,
    }
