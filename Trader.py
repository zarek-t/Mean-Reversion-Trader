import matplotlib.pyplot as plt
import numpy as np
from itertools import product

from backtest import (
    DEFAULT_PERIOD,
    DEFAULT_TICKERS,
    START_CASH,
    TRANSACTION_COST,
    backtest_strategy,
    calculate_enhanced_metrics,
    calculate_signals,
    download_and_prepare_data,
)

TICKERS = DEFAULT_TICKERS
PERIOD = DEFAULT_PERIOD

WINDOWS = [3, 4, 6, 8, 10, 12, 16, 20, 24]
Z_SCORES = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]


def optimize_parameters():
    """Parameter optimization for 5-year period"""
    all_results = []

    print("Mean Reversion Strategy Optimizer")
    print("=" * 60)
    print(f"Testing {len(TICKERS)} stocks over 5-year period")
    print(
        f"Parameter combinations: {len(WINDOWS)} windows × {len(Z_SCORES)} Z-scores = "
        f"{len(WINDOWS) * len(Z_SCORES)}"
    )
    print("-" * 60)

    stock_data = {}
    for ticker in TICKERS:
        data = download_and_prepare_data(ticker, period=PERIOD)
        if data is not None:
            stock_data[ticker] = data

    if len(stock_data) < 3:
        print("Insufficient data. Check your internet connection and try again.")
        return []

    print(f"Successfully downloaded data for {len(stock_data)} stocks")

    for window, z_score in product(WINDOWS, Z_SCORES):
        z_buy = -z_score
        z_sell = z_score

        ticker_results = []

        for ticker, close in stock_data.items():
            try:
                close_clean, ma, std, z, signals = calculate_signals(
                    close, window, z_buy, z_sell
                )
                portfolio_values, buy_dates, sell_dates, trades = backtest_strategy(
                    close_clean, signals, START_CASH, TRANSACTION_COST
                )
                metrics = calculate_enhanced_metrics(
                    portfolio_values, close_clean, START_CASH, trades
                )

                ticker_results.append(
                    {
                        "ticker": ticker,
                        "window": window,
                        "z_score": z_score,
                        "z_buy": z_buy,
                        "z_sell": z_sell,
                        **metrics,
                    }
                )

            except Exception:
                continue

        if len(ticker_results) >= 3:
            all_results.append(
                {
                    "window": window,
                    "z_score": z_score,
                    "avg_return": np.mean([r["total_return"] for r in ticker_results]),
                    "avg_excess_return": np.mean(
                        [r["excess_return"] for r in ticker_results]
                    ),
                    "avg_sharpe": np.mean([r["sharpe"] for r in ticker_results]),
                    "avg_calmar": np.mean([r["calmar"] for r in ticker_results]),
                    "avg_drawdown": np.mean(
                        [r["max_drawdown"] for r in ticker_results]
                    ),
                    "avg_win_rate": np.mean([r["win_rate"] for r in ticker_results]),
                    "avg_trades": np.mean([r["trades"] for r in ticker_results]),
                    "ticker_results": ticker_results,
                }
            )

    return all_results


def display_results(results):
    """Display optimization results"""
    if not results:
        print("No results to display.")
        return

    results.sort(key=lambda x: x["avg_excess_return"], reverse=True)

    print("\n" + "=" * 100)
    print("OPTIMIZATION RESULTS (Sorted by Average Excess Return)")
    print("=" * 100)

    print(
        f"{'Window':<8} {'Z-Score':<8} {'Avg Return':<12} {'Excess Return':<15} "
        f"{'Sharpe':<8} {'Calmar':<8} {'Win Rate':<10} {'Trades':<8}"
    )
    print("-" * 100)

    for r in results[:15]:
        print(
            f"{r['window']:<8} {r['z_score']:<8} {r['avg_return']:<12.2f} "
            f"{r['avg_excess_return']:<15.2f} {r['avg_sharpe']:<8.2f} "
            f"{r['avg_calmar']:<8.2f} {r['avg_win_rate']:<10.1f} {r['avg_trades']:<8.1f}"
        )

    print("\n" + "=" * 100)
    print("DETAILED RESULTS FOR TOP 3 PARAMETER COMBINATIONS")
    print("=" * 100)

    for i, result in enumerate(results[:3]):
        print(f"\n{i + 1}. Window={result['window']}, Z-Score={result['z_score']}")
        print(f"   Average Return: {result['avg_return']:.2f}%")
        print(f"   Average Excess Return: {result['avg_excess_return']:.2f}%")
        print(f"   Average Sharpe: {result['avg_sharpe']:.2f}")
        print(f"   Average Calmar: {result['avg_calmar']:.2f}")
        print(f"   Average Max Drawdown: {result['avg_drawdown']:.2f}%")
        print(f"   Average Win Rate: {result['avg_win_rate']:.1f}%")
        print(f"   Average Trades: {result['avg_trades']:.1f}")

        print("   Individual Stock Results:")
        for tr in result["ticker_results"]:
            print(
                f"     {tr['ticker']}: {tr['total_return']:.2f}% "
                f"(vs BH: {tr['bh_return']:.2f}%, Sharpe: {tr['sharpe']:.2f})"
            )


def run_single_backtest(ticker, window, z_score):
    """Run a single backtest with given parameters"""
    print(f"\nRunning backtest for {ticker}")
    print(f"Parameters: Window={window}, Z-Score={z_score}")

    close = download_and_prepare_data(ticker, period=PERIOD)
    if close is None:
        print(f"Could not download data for {ticker}")
        return

    z_buy = -z_score
    z_sell = z_score

    close_clean, ma, std, z, signals = calculate_signals(close, window, z_buy, z_sell)
    portfolio_values, buy_dates, sell_dates, trades = backtest_strategy(
        close_clean, signals, START_CASH, TRANSACTION_COST
    )
    metrics = calculate_enhanced_metrics(
        portfolio_values, close_clean, START_CASH, trades
    )

    bh_shares = START_CASH // close_clean.iloc[0]
    bh_cash = START_CASH - bh_shares * close_clean.iloc[0]
    bh_portfolio_values = [bh_cash + bh_shares * price for price in close_clean]

    spy_data = download_and_prepare_data("SPY", period=PERIOD)
    spy_portfolio_values = []
    if spy_data is not None:
        spy_aligned = spy_data.reindex(close_clean.index, method="ffill")
        spy_shares = START_CASH // spy_aligned.iloc[0]
        spy_cash = START_CASH - spy_shares * spy_aligned.iloc[0]
        spy_portfolio_values = [spy_cash + spy_shares * price for price in spy_aligned]

    print(f"Final strategy value: ${metrics['final_value']:.2f}")
    print(f"Strategy return: {metrics['total_return']:.2f}%")
    print(f"Buy-and-hold return: {metrics['bh_return']:.2f}%")
    print(f"Excess return: {metrics['excess_return']:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"Calmar ratio: {metrics['calmar']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    print(f"Win rate: {metrics['win_rate']:.1f}%")
    print(f"Total trades: {trades}")
    print(f"Average trade return: {metrics['avg_trade_return']:.2f}%")

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    axes[0].plot(close_clean.index, close_clean, label="Price", linewidth=1.5)
    axes[0].plot(
        ma.index,
        ma,
        label=f"{window}-Week MA",
        linestyle="--",
        alpha=0.8,
    )
    if buy_dates:
        axes[0].scatter(
            buy_dates,
            close_clean.loc[buy_dates],
            marker="^",
            color="green",
            label="Buy",
            s=100,
        )
    if sell_dates:
        axes[0].scatter(
            sell_dates,
            close_clean.loc[sell_dates],
            marker="v",
            color="red",
            label="Sell",
            s=100,
        )
    axes[0].set_title(
        f"{ticker} Mean Reversion Strategy (Window={window}, Z-Score={z_score})"
    )
    axes[0].set_ylabel("Price")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(
        close_clean.index,
        portfolio_values,
        label="Strategy Portfolio",
        linewidth=1.5,
        color="blue",
    )
    axes[1].plot(
        close_clean.index,
        bh_portfolio_values,
        label="Buy & Hold",
        linewidth=1.5,
        color="orange",
    )
    if spy_portfolio_values:
        axes[1].plot(
            close_clean.index,
            spy_portfolio_values,
            label="SPY",
            linewidth=1.5,
            color="red",
        )
    axes[1].axhline(
        y=START_CASH,
        color="black",
        linestyle="--",
        alpha=0.5,
        label="Initial Capital",
    )
    axes[1].set_ylabel("Portfolio Value ($)")
    axes[1].set_xlabel("Date")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def analyze_parameter_sensitivity(results):
    """Analyze how sensitive the strategy is to different parameters"""
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)

    window_performance = {}
    zscore_performance = {}

    for r in results:
        window_performance.setdefault(r["window"], []).append(r["avg_excess_return"])
        zscore_performance.setdefault(r["z_score"], []).append(r["avg_excess_return"])

    print("\nWindow Sensitivity (Average Excess Return):")
    for window in sorted(window_performance.keys()):
        avg_return = np.mean(window_performance[window])
        print(f"  Window {window:2d}: {avg_return:6.2f}%")

    print("\nZ-Score Sensitivity (Average Excess Return):")
    for zscore in sorted(zscore_performance.keys()):
        avg_return = np.mean(zscore_performance[zscore])
        print(f"  Z-Score {zscore:3.1f}: {avg_return:6.2f}%")


if __name__ == "__main__":
    print("Mean Reversion Strategy Optimizer")
    print("=" * 60)

    results = optimize_parameters()

    if results:
        display_results(results)
        analyze_parameter_sensitivity(results)

        best_params = results[0]
        best_stock_result = max(
            best_params["ticker_results"], key=lambda x: x["total_return"]
        )
        best_ticker = best_stock_result["ticker"]

        print("\nRunning backtest with best parameters:")
        print(f"Window={best_params['window']}, Z-Score={best_params['z_score']}")
        print(
            f"Best performing stock: {best_ticker} "
            f"(Return: {best_stock_result['total_return']:.2f}%)"
        )
        run_single_backtest(best_ticker, best_params["window"], best_params["z_score"])
    else:
        print("No results obtained. Check your internet connection and try again.")
