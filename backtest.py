import warnings

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

START_CASH = 10000
TRANSACTION_COST = 0.001
DEFAULT_PERIOD = "5y"


def download_and_prepare_data(ticker, period="5y", interval="1wk"):
    """Download and prepare weekly close prices for a ticker."""
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

        return close
    except Exception:
        return None


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

    dates = [_serialize_date(d) for d in close_clean.index]

    return {
        "ticker": ticker,
        "window": window,
        "z_score": z_score,
        "period": period,
        "start_cash": start_cash,
        "metrics": {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items()},
        "chart": {
            "dates": dates,
            "close": [round(v, 2) for v in close_clean.tolist()],
            "ma": [round(v, 2) if not np.isnan(v) else None for v in ma.tolist()],
            "portfolio": [round(v, 2) for v in portfolio_values],
            "buy_hold": [round(v, 2) for v in bh_portfolio_values],
            "spy": [round(v, 2) for v in spy_portfolio_values] if spy_portfolio_values else [],
            "buy_dates": [_serialize_date(d) for d in buy_dates],
            "sell_dates": [_serialize_date(d) for d in sell_dates],
        },
    }
