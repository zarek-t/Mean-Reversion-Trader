import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from itertools import product

# Parameters
TICKERS = ['AAPL', 'MSFT', 'TSLA', 'JPM', 'XOM']  # Diverse stocks: Tech, Finance, Energy
START_CASH = 10000

# Parameter ranges to test
WINDOWS = [4, 6, 8, 12, 16, 20]
Z_SCORES = [0.5, 1.0, 1.5, 2.0, 2.5]

def download_and_prepare_data(ticker, period='5y', interval='1wk'):
    """Download and prepare data for a given ticker"""
    try:
        data = yf.download(ticker, period=period, interval=interval, auto_adjust=True)
        
        # Fix multi-index columns (flatten)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        # Select the Close price column
        price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
        close = data[price_col].astype(float).dropna()
        
        return close
    except Exception as e:
        print(f"Error downloading data for {ticker}: {e}")
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

def backtest_strategy(close, signals, start_cash):
    """Backtest the strategy and return results"""
    cash = start_cash
    shares = 0
    portfolio_values = []
    buy_dates = []
    sell_dates = []
    
    for i, (date, price) in enumerate(close.items()):
        signal = signals[i]
        if signal == 'BUY' and cash >= price:
            shares = cash // price
            cash -= shares * price
            buy_dates.append(date)
        elif signal == 'SELL' and shares > 0:
            cash += shares * price
            shares = 0
            sell_dates.append(date)
        portfolio_value = cash + shares * price
        portfolio_values.append(portfolio_value)
    
    return portfolio_values, buy_dates, sell_dates

def calculate_metrics(portfolio_values, close, start_cash):
    """Calculate performance metrics"""
    final_value = portfolio_values[-1]
    total_return = (final_value / start_cash - 1) * 100
    
    # Buy and hold comparison
    bh_shares = start_cash // close.iloc[0]
    bh_cash = start_cash - bh_shares * close.iloc[0]
    bh_value = bh_cash + bh_shares * close.iloc[-1]
    bh_return = (bh_value / start_cash - 1) * 100
    
    # Calculate Sharpe ratio (simplified)
    returns = pd.Series(portfolio_values).pct_change().dropna()
    sharpe = returns.mean() / returns.std() if returns.std() > 0 else 0
    
    # Calculate max drawdown
    peak = pd.Series(portfolio_values).expanding().max()
    drawdown = (pd.Series(portfolio_values) - peak) / peak
    max_drawdown = drawdown.min() * 100
    
    return {
        'final_value': final_value,
        'total_return': total_return,
        'bh_return': bh_return,
        'excess_return': total_return - bh_return,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }

def optimize_parameters():
    """Find optimal parameters across all stocks"""
    results = []
    
    print("Testing parameter combinations...")
    print(f"Stocks: {TICKERS}")
    print(f"Windows: {WINDOWS}")
    print(f"Z-scores: {Z_SCORES}")
    print("-" * 80)
    
    # Download data for all stocks
    stock_data = {}
    for ticker in TICKERS:
        data = download_and_prepare_data(ticker)
        if data is not None:
            stock_data[ticker] = data
            print(f"Downloaded {ticker}: {len(data)} data points")
        else:
            print(f"Failed to download {ticker}")
    
    if not stock_data:
        print("No data downloaded. Exiting.")
        return
    
    # Test all parameter combinations
    for window, z_score in product(WINDOWS, Z_SCORES):
        z_buy = -z_score
        z_sell = z_score
        
        ticker_results = []
        
        for ticker, close in stock_data.items():
            try:
                # Calculate signals and backtest
                close_clean, ma, std, z, signals = calculate_signals(close, window, z_buy, z_sell)
                portfolio_values, buy_dates, sell_dates = backtest_strategy(close_clean, signals, START_CASH)
                metrics = calculate_metrics(portfolio_values, close_clean, START_CASH)
                
                ticker_results.append({
                    'ticker': ticker,
                    'window': window,
                    'z_score': z_score,
                    'z_buy': z_buy,
                    'z_sell': z_sell,
                    **metrics
                })
                
            except Exception as e:
                print(f"Error processing {ticker} with window={window}, z_score={z_score}: {e}")
        
        if ticker_results:
            # Calculate average metrics across all stocks for this parameter combination
            avg_return = np.mean([r['total_return'] for r in ticker_results])
            avg_excess_return = np.mean([r['excess_return'] for r in ticker_results])
            avg_sharpe = np.mean([r['sharpe'] for r in ticker_results])
            avg_drawdown = np.mean([r['max_drawdown'] for r in ticker_results])
            
            results.append({
                'window': window,
                'z_score': z_score,
                'avg_return': avg_return,
                'avg_excess_return': avg_excess_return,
                'avg_sharpe': avg_sharpe,
                'avg_drawdown': avg_drawdown,
                'ticker_results': ticker_results
            })
    
    return results

def display_results(results):
    """Display optimization results"""
    if not results:
        print("No results to display.")
        return
    
    # Sort by average excess return
    results.sort(key=lambda x: x['avg_excess_return'], reverse=True)
    
    print("\n" + "="*80)
    print("OPTIMIZATION RESULTS (Sorted by Average Excess Return)")
    print("="*80)
    
    print(f"{'Window':<8} {'Z-Score':<8} {'Avg Return':<12} {'Excess Return':<15} {'Sharpe':<8} {'Max DD':<8}")
    print("-" * 80)
    
    for r in results[:10]:  # Top 10 results
        print(f"{r['window']:<8} {r['z_score']:<8} {r['avg_return']:<12.2f} {r['avg_excess_return']:<15.2f} {r['avg_sharpe']:<8.2f} {r['avg_drawdown']:<8.2f}")
    
    # Show detailed results for top 3
    print("\n" + "="*80)
    print("DETAILED RESULTS FOR TOP 3 PARAMETER COMBINATIONS")
    print("="*80)
    
    for i, result in enumerate(results[:3]):
        print(f"\n{i+1}. Window={result['window']}, Z-Score={result['z_score']}")
        print(f"   Average Return: {result['avg_return']:.2f}%")
        print(f"   Average Excess Return: {result['avg_excess_return']:.2f}%")
        print(f"   Average Sharpe: {result['avg_sharpe']:.2f}")
        print(f"   Average Max Drawdown: {result['avg_drawdown']:.2f}%")
        
        print("   Individual Stock Results:")
        for tr in result['ticker_results']:
            print(f"     {tr['ticker']}: {tr['total_return']:.2f}% (vs BH: {tr['bh_return']:.2f}%)")

def run_single_backtest(ticker, window, z_score):
    """Run a single backtest with given parameters"""
    print(f"\nRunning single backtest for {ticker} with window={window}, z_score={z_score}")
    
    close = download_and_prepare_data(ticker)
    if close is None:
        return
    
    z_buy = -z_score
    z_sell = z_score
    
    close_clean, ma, std, z, signals = calculate_signals(close, window, z_buy, z_sell)
    portfolio_values, buy_dates, sell_dates = backtest_strategy(close_clean, signals, START_CASH)
    metrics = calculate_metrics(portfolio_values, close_clean, START_CASH)
    
    print(f"Final strategy value: ${metrics['final_value']:.2f}")
    print(f"Strategy return: {metrics['total_return']:.2f}%")
    print(f"Buy-and-hold return: {metrics['bh_return']:.2f}%")
    print(f"Excess return: {metrics['excess_return']:.2f}%")
    print(f"Sharpe ratio: {metrics['sharpe']:.2f}")
    print(f"Max drawdown: {metrics['max_drawdown']:.2f}%")
    
    # Plot
    df = pd.DataFrame({
        'Close': close_clean,
        'MA': ma,
        'Z': z
    })
    
    plt.figure(figsize=(12, 8))
    
    # Price and MA plot
    plt.subplot(2, 1, 1)
    plt.plot(df.index, df['Close'], label='Price')
    plt.plot(df.index, df['MA'], label=f'{window}-Week MA', linestyle='--')
    plt.scatter(df.loc[buy_dates].index, df.loc[buy_dates]['Close'], marker='^', color='green', label='Buy', s=100)
    plt.scatter(df.loc[sell_dates].index, df.loc[sell_dates]['Close'], marker='v', color='red', label='Sell', s=100)
    plt.title(f'{ticker} Mean Reversion Strategy (Window={window}, Z-Score={z_score})')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    
    # Z-score plot
    plt.subplot(2, 1, 2)
    plt.plot(df.index, df['Z'], label='Z-Score')
    plt.axhline(y=z_buy, color='green', linestyle='--', label=f'Buy threshold ({z_buy})')
    plt.axhline(y=z_sell, color='red', linestyle='--', label=f'Sell threshold ({z_sell})')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.ylabel('Z-Score')
    plt.xlabel('Date')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    print("Mean Reversion Strategy Optimizer")
    print("="*50)
    
    # Run optimization
    results = optimize_parameters()
    
    if results:
        display_results(results)
        
        # Run single backtest with best parameters
        best_params = results[0]
        print(f"\nRunning single backtest with best parameters: Window={best_params['window']}, Z-Score={best_params['z_score']}")
        run_single_backtest('AAPL', best_params['window'], best_params['z_score'])
    else:
        print("No results obtained. Check your internet connection and try again.")
