import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Enhanced Parameters
TICKERS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',  # Tech
    'JPM', 'BAC', 'WFC', 'GS', 'MS',           # Finance
    'XOM', 'CVX', 'COP', 'SLB', 'EOG',         # Energy
    'JNJ', 'PFE', 'UNH', 'ABBV', 'MRK',        # Healthcare
    'PG', 'KO', 'PEP', 'WMT', 'HD'             # Consumer
]

START_CASH = 10000
TRANSACTION_COST = 0.001  # 0.1% per trade

# Enhanced parameter ranges
WINDOWS = [3, 4, 6, 8, 10, 12, 16, 20, 24]
Z_SCORES = [0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 1.8, 2.0, 2.5]

# Time period to test
PERIOD = '5y'

def download_and_prepare_data(ticker, period='5y', interval='1wk'):
    """Download and prepare data for a given ticker"""
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)
        
        if data.empty:
            return None
            
        # Fix multi-index columns (flatten)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Select the Close price column
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        close = data[price_col].astype(float).dropna()
        
        # Ensure minimum data points
        if len(close) < 50:
            return None
            
        return close
    except Exception as e:
        return None

def calculate_signals(close, window, z_buy, z_sell):
    """Calculate trading signals for given parameters"""
    # Calculate moving average and std
    ma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    
    # Drop rows with NaN
    valid_idx = ma.index.intersection(std.index).intersection(close.index)
    close = close.loc[valid_idx]
    ma = ma.loc[valid_idx]
    std = std.loc[valid_idx]
    
    # Calculate Z-score
    z = (close - ma) / std
    
    # Generate signals
    holding = False
    signals = []
    
    for date, row in pd.DataFrame({'Z': z}).iterrows():
        if not holding and row['Z'] < z_buy:
            signals.append('BUY')
            holding = True
        elif holding and row['Z'] > z_sell:
            signals.append('SELL')
            holding = False
        else:
            signals.append('HOLD')
    
    return close, ma, std, z, signals

def backtest_strategy(close, signals, start_cash, transaction_cost=0):
    """Backtest the strategy with transaction costs"""
    cash = start_cash
    shares = 0
    portfolio_values = []
    buy_dates = []
    sell_dates = []
    trades = 0
    
    for i, (date, price) in enumerate(close.items()):
        signal = signals[i]
        if signal == 'BUY' and cash >= price:
            # Calculate shares with transaction cost
            available_cash = cash * (1 - transaction_cost)
            shares = available_cash // price
            cash -= shares * price * (1 + transaction_cost)
            buy_dates.append(date)
            trades += 1
        elif signal == 'SELL' and shares > 0:
            # Sell with transaction cost
            cash += shares * price * (1 - transaction_cost)
            shares = 0
            sell_dates.append(date)
            trades += 1
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)
    
    return portfolio_values, buy_dates, sell_dates, trades

def calculate_enhanced_metrics(portfolio_values, close, start_cash, trades=0):
    """Calculate comprehensive performance metrics"""
    final_value = portfolio_values[-1]
    total_return = (final_value / start_cash - 1) * 100
    
    # Buy and hold comparison
    bh_shares = start_cash // close.iloc[0]
    bh_cash = start_cash - bh_shares * close.iloc[0]
    bh_value = bh_cash + bh_shares * close.iloc[-1]
    bh_return = (bh_value / start_cash - 1) * 100
    
    # Calculate returns series
    returns = pd.Series(portfolio_values).pct_change().dropna()
    
    # Risk metrics
    volatility = returns.std() * np.sqrt(52) * 100  # Annualized
    sharpe = (returns.mean() * 52) / (returns.std() * np.sqrt(52)) if returns.std() > 0 else 0
    
    # Maximum drawdown
    peak = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    # Calmar ratio (return / max drawdown)
    calmar = total_return / abs(max_drawdown) if max_drawdown != 0 else 0
    
    # Win rate (simplified - positive vs negative periods)
    positive_periods = (returns > 0).sum()
    total_periods = len(returns)
    win_rate = (positive_periods / total_periods) * 100 if total_periods > 0 else 0
    
    # Average trade return (if we have trade data)
    avg_trade_return = total_return / trades if trades > 0 else 0
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'bh_return': bh_return,
        'excess_return': total_return - bh_return,
        'volatility': volatility,
        'sharpe': sharpe,
        'calmar': calmar,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'trades': trades,
        'avg_trade_return': avg_trade_return
    }

def optimize_parameters():
    """Parameter optimization for 5-year period"""
    all_results = []
    
    print("Mean Reversion Strategy Optimizer")
    print("="*60)
    print(f"Testing {len(TICKERS)} stocks over 5-year period")
    print(f"Parameter combinations: {len(WINDOWS)} windows × {len(Z_SCORES)} Z-scores = {len(WINDOWS) * len(Z_SCORES)}")
    print("-" * 60)
    
    # Download data for all stocks
    stock_data = {}
    for ticker in TICKERS:
        data = download_and_prepare_data(ticker, period=PERIOD)
        if data is not None:
            stock_data[ticker] = data
    
    if len(stock_data) < 3:  # Need at least 3 stocks for meaningful results
        print("Insufficient data. Check your internet connection and try again.")
        return []
    
    print(f"Successfully downloaded data for {len(stock_data)} stocks")
    
    # Test all parameter combinations
    for window, z_score in product(WINDOWS, Z_SCORES):
        z_buy = -z_score
        z_sell = z_score
        
        ticker_results = []
        
        for ticker, close in stock_data.items():
            try:
                # Calculate signals and backtest
                close_clean, ma, std, z, signals = calculate_signals(close, window, z_buy, z_sell)
                portfolio_values, buy_dates, sell_dates, trades = backtest_strategy(
                    close_clean, signals, START_CASH, TRANSACTION_COST
                )
                metrics = calculate_enhanced_metrics(portfolio_values, close_clean, START_CASH, trades)
                
                ticker_results.append({
                    'ticker': ticker,
                    'window': window,
                    'z_score': z_score,
                    'z_buy': z_buy,
                    'z_sell': z_sell,
                    **metrics
                })
                
            except Exception as e:
                continue
        
        if len(ticker_results) >= 3:  # Only include if we have results for at least 3 stocks
            # Calculate average metrics across all stocks for this parameter combination
            avg_return = np.mean([r['total_return'] for r in ticker_results])
            avg_excess_return = np.mean([r['excess_return'] for r in ticker_results])
            avg_sharpe = np.mean([r['sharpe'] for r in ticker_results])
            avg_calmar = np.mean([r['calmar'] for r in ticker_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in ticker_results])
            avg_win_rate = np.mean([r['win_rate'] for r in ticker_results])
            avg_trades = np.mean([r['trades'] for r in ticker_results])
            
            all_results.append({
                'window': window,
                'z_score': z_score,
                'avg_return': avg_return,
                'avg_excess_return': avg_excess_return,
                'avg_sharpe': avg_sharpe,
                'avg_calmar': avg_calmar,
                'avg_drawdown': avg_drawdown,
                'avg_win_rate': avg_win_rate,
                'avg_trades': avg_trades,
                'ticker_results': ticker_results
            })
    
    return all_results

def display_results(results):
    """Display optimization results"""
    if not results:
        print("No results to display.")
        return
    
    # Sort by average excess return
    results.sort(key=lambda x: x['avg_excess_return'], reverse=True)
    
    print("\n" + "="*100)
    print("OPTIMIZATION RESULTS (Sorted by Average Excess Return)")
    print("="*100)
    
    print(f"{'Window':<8} {'Z-Score':<8} {'Avg Return':<12} {'Excess Return':<15} {'Sharpe':<8} {'Calmar':<8} {'Win Rate':<10} {'Trades':<8}")
    print("-" * 100)
    
    for r in results[:15]:  # Top 15 results
        print(f"{r['window']:<8} {r['z_score']:<8} {r['avg_return']:<12.2f} {r['avg_excess_return']:<15.2f} {r['avg_sharpe']:<8.2f} {r['avg_calmar']:<8.2f} {r['avg_win_rate']:<10.1f} {r['avg_trades']:<8.1f}")
    
    # Show detailed results for top 3
    print("\n" + "="*100)
    print("DETAILED RESULTS FOR TOP 3 PARAMETER COMBINATIONS")
    print("="*100)
    
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Window={result['window']}, Z-Score={result['z_score']}")
        print(f"   Average Return: {result['avg_return']:.2f}%")
        print(f"   Average Excess Return: {result['avg_excess_return']:.2f}%")
        print(f"   Average Sharpe: {result['avg_sharpe']:.2f}")
        print(f"   Average Calmar: {result['avg_calmar']:.2f}")
        print(f"   Average Max Drawdown: {result['avg_drawdown']:.2f}%")
        print(f"   Average Win Rate: {result['avg_win_rate']:.1f}%")
        print(f"   Average Trades: {result['avg_trades']:.1f}")
        
        print("   Individual Stock Results:")
        for tr in result['ticker_results']:
            print(f"     {tr['ticker']}: {tr['total_return']:.2f}% (vs BH: {tr['bh_return']:.2f}%, Sharpe: {tr['sharpe']:.2f})")

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
    metrics = calculate_enhanced_metrics(portfolio_values, close_clean, START_CASH, trades)
    
    # Calculate buy-and-hold portfolio values
    bh_shares = START_CASH // close_clean.iloc[0]
    bh_cash = START_CASH - bh_shares * close_clean.iloc[0]
    bh_portfolio_values = [bh_cash + bh_shares * price for price in close_clean]
    
    # Download SPY data for comparison
    spy_data = download_and_prepare_data('SPY', period=PERIOD)
    spy_portfolio_values = []
    if spy_data is not None:
        # Align SPY data with the same date range as the ticker
        spy_aligned = spy_data.reindex(close_clean.index, method='ffill')
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
    
    # Enhanced plot with only 2 subplots
    df = pd.DataFrame({
        'Close': close_clean,
        'MA': ma,
        'Portfolio': portfolio_values,
        'BuyHold': bh_portfolio_values
    })
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Price and MA plot
    axes[0].plot(df.index, df['Close'], label='Price', linewidth=1.5)
    axes[0].plot(df.index, df['MA'], label=f'{window}-Week MA', linestyle='--', alpha=0.8)
    axes[0].scatter(df.loc[buy_dates].index, df.loc[buy_dates]['Close'], marker='^', color='green', label='Buy', s=100)
    axes[0].scatter(df.loc[sell_dates].index, df.loc[sell_dates]['Close'], marker='v', color='red', label='Sell', s=100)
    axes[0].set_title(f'{ticker} Mean Reversion Strategy (Window={window}, Z-Score={z_score})')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Portfolio value plot with buy-and-hold and SPY comparison
    axes[1].plot(df.index, df['Portfolio'], label='Strategy Portfolio', linewidth=1.5, color='blue')
    axes[1].plot(df.index, df['BuyHold'], label='Buy & Hold', linewidth=1.5, color='orange', linestyle='-')
    if spy_portfolio_values:
        axes[1].plot(df.index, spy_portfolio_values, label='SPY', linewidth=1.5, color='red', linestyle='-')
    axes[1].axhline(y=START_CASH, color='black', linestyle='--', alpha=0.5, label='Initial Capital')
    axes[1].set_ylabel('Portfolio Value ($)')
    axes[1].set_xlabel('Date')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def analyze_parameter_sensitivity(results):
    """Analyze how sensitive the strategy is to different parameters"""
    print("\n" + "="*80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Group by parameter
    window_performance = {}
    zscore_performance = {}
    
    for r in results:
        # Window analysis
        if r['window'] not in window_performance:
            window_performance[r['window']] = []
        window_performance[r['window']].append(r['avg_excess_return'])
        
        # Z-score analysis
        if r['z_score'] not in zscore_performance:
            zscore_performance[r['z_score']] = []
        zscore_performance[r['z_score']].append(r['avg_excess_return'])
    
    # Display window sensitivity
    print("\nWindow Sensitivity (Average Excess Return):")
    for window in sorted(window_performance.keys()):
        avg_return = np.mean(window_performance[window])
        print(f"  Window {window:2d}: {avg_return:6.2f}%")
    
    # Display Z-score sensitivity
    print("\nZ-Score Sensitivity (Average Excess Return):")
    for zscore in sorted(zscore_performance.keys()):
        avg_return = np.mean(zscore_performance[zscore])
        print(f"  Z-Score {zscore:3.1f}: {avg_return:6.2f}%")

# Main execution
if __name__ == "__main__":
    print("Mean Reversion Strategy Optimizer")
    print("="*60)
    
    # Run optimization
    results = optimize_parameters()
    
    if results:
        display_results(results)
        analyze_parameter_sensitivity(results)
        
        # Run single backtest with best parameters
        best_params = results[0]
        
        # Find the best-performing stock from the best parameter combination
        best_stock_result = max(best_params['ticker_results'], key=lambda x: x['total_return'])
        best_ticker = best_stock_result['ticker']
        
        print(f"\nRunning backtest with best parameters:")
        print(f"Window={best_params['window']}, Z-Score={best_params['z_score']}")
        print(f"Best performing stock: {best_ticker} (Return: {best_stock_result['total_return']:.2f}%)")
        run_single_backtest(best_ticker, best_params['window'], best_params['z_score'])
    else:
        print("No results obtained. Check your internet connection and try again.")
