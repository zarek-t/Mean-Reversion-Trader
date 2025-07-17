import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Parameters
TICKER = 'PLTR'
START_CASH = 10000
WINDOW = 4
Z_BUY = -1
Z_SELL = 1

# Download data
data = yf.download(TICKER, period='1y', interval='1wk', auto_adjust=True)

# Fix multi-index columns (flatten)
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Check columns
print("Columns:", data.columns)

# Select the Close price column (usually 'Adj Close' if auto_adjust=True)
price_col = 'Adj Close' if 'Adj Close' in data.columns else 'Close'

# Extract Close price series
close = data[price_col].astype(float).dropna()

# Calculate moving average and std
ma = close.rolling(window=WINDOW).mean()
std = close.rolling(window=WINDOW).std()

# Drop rows with NaN
valid_idx = ma.index.intersection(std.index).intersection(close.index)
close = close.loc[valid_idx]
ma = ma.loc[valid_idx]
std = std.loc[valid_idx]

# Calculate Z-score
z = (close - ma) / std

# Create DataFrame to hold all
df = pd.DataFrame({
    'Close': close,
    'MA': ma,
    'STD': std,
    'Z': z,
})

# Generate signals
holding = False
signals = []

for date, row in df.iterrows():
    if not holding and row['Z'] < Z_BUY:
        signals.append('BUY')
        holding = True
    elif holding and row['Z'] > Z_SELL:
        signals.append('SELL')
        holding = False
    else:
        signals.append('HOLD')

df['Signal'] = signals

# Backtest
cash = START_CASH
shares = 0
portfolio_values = []
buy_dates = []
sell_dates = []

for date, row in df.iterrows():
    price = row['Close']
    signal = row['Signal']
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

df['Portfolio'] = portfolio_values

# Buy and hold
bh_shares = START_CASH // df['Close'].iloc[0]
bh_cash = START_CASH - bh_shares * df['Close'].iloc[0]
bh_value = bh_cash + bh_shares * df['Close'].iloc[-1]

# Results
print(f"Final strategy value: ${df['Portfolio'].iloc[-1]:.2f}")
print(f"Buy-and-hold value:   ${bh_value:.2f}")
print(f"Strategy return:      {(df['Portfolio'].iloc[-1] / START_CASH - 1)*100:.2f}%")
print(f"Buy-and-hold return:  {(bh_value / START_CASH - 1)*100:.2f}%")

# Plot
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='Price')
plt.plot(df.index, df['MA'], label=f'{WINDOW}-Week MA', linestyle='--')
plt.scatter(df.loc[buy_dates].index, df.loc[buy_dates]['Close'], marker='^', color='green', label='Buy', s=100)
plt.scatter(df.loc[sell_dates].index, df.loc[sell_dates]['Close'], marker='v', color='red', label='Sell', s=100)
plt.title(f'{TICKER} Mean Reversion Strategy Backtest')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
