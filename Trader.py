import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
TICKER = 'AAPL'
START_CASH = 10000
WINDOW = 4
Z_BUY = -1
Z_SELL = 1

# Download 1 year of weekly data
data = yf.download(TICKER, period='1y', interval='1wk', auto_adjust=True)
if data is None or data.empty:
    print(f"Failed to download data for {TICKER}.")
    exit(1)

# Make sure 'Close' exists and is float
if 'Close' not in data.columns:
    print("Close price data is missing.")
    exit(1)

data['Close'] = data['Close'].astype(float)

# Calculate rolling stats
data['MA'] = data['Close'].rolling(window=WINDOW).mean()
data['STD'] = data['Close'].rolling(window=WINDOW).std()

# Drop rows with NaNs in any calculated columns
data = data.dropna()  # Now safe: only drops rows where MA, STD, or Close are NaN

# Calculate Z-score
data['Z'] = (data['Close'] - data['MA']) / data['STD']

# Generate signals
holding = False
signals = []
for idx, row in data.iterrows():
    if not holding and row['Z'] < Z_BUY:
        signals.append('BUY')
        holding = True
    elif holding and row['Z'] > Z_SELL:
        signals.append('SELL')
        holding = False
    else:
        signals.append('HOLD')
data['Signal'] = signals

# Backtest
cash = START_CASH
shares = 0
portfolio_values = []
buy_dates = []
sell_dates = []

for idx, row in data.iterrows():
    price = row['Close']
    signal = row['Signal']
    if signal == 'BUY' and cash >= price:
        shares = cash // price
        cash -= shares * price
        buy_dates.append(idx)
    elif signal == 'SELL' and shares > 0:
        cash += shares * price
        shares = 0
        sell_dates.append(idx)
    portfolio_value = cash + shares * price
    portfolio_values.append(portfolio_value)

data['Portfolio'] = portfolio_values

# Buy-and-hold comparison
bh_shares = START_CASH // data['Close'].iloc[0]
bh_cash = START_CASH - bh_shares * data['Close'].iloc[0]
bh_value = bh_cash + bh_shares * data['Close'].iloc[-1]

# Print results
final_value = data['Portfolio'].iloc[-1]
print(f"Final strategy value: ${final_value:.2f}")
print(f"Buy-and-hold value:   ${bh_value:.2f}")
print(f"Strategy return:      {100 * (final_value / START_CASH - 1):.2f}%")
print(f"Buy-and-hold return:  {100 * (bh_value / START_CASH - 1):.2f}%")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['Close'], label='Price')
plt.plot(data.index, data['MA'], label=f'{WINDOW}-Week MA', linestyle='--')
plt.scatter(data.loc[buy_dates].index, data.loc[buy_dates]['Close'], marker='^', color='green', label='Buy', s=100)
plt.scatter(data.loc[sell_dates].index, data.loc[sell_dates]['Close'], marker='v', color='red', label='Sell', s=100)
plt.title(f'{TICKER} Mean Reversion Strategy Backtest')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
