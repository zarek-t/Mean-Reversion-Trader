const START_CASH = 10000;
const TRANSACTION_COST = 0.001;

const TICKERS = [
  { symbol: "AAPL", name: "Apple" },
  { symbol: "MSFT", name: "Microsoft" },
  { symbol: "GOOGL", name: "Alphabet" },
  { symbol: "JPM", name: "JPMorgan" },
  { symbol: "XOM", name: "Exxon Mobil" },
  { symbol: "COP", name: "ConocoPhillips" },
  { symbol: "JNJ", name: "Johnson & Johnson" },
  { symbol: "TSLA", name: "Tesla" },
];

const dataCache = {};

function rollingMean(values, window) {
  return values.map((_, i) => {
    if (i < window - 1) return null;
    const slice = values.slice(i - window + 1, i + 1);
    return slice.reduce((a, b) => a + b, 0) / window;
  });
}

function rollingStd(values, window) {
  return values.map((_, i) => {
    if (i < window - 1) return null;
    const slice = values.slice(i - window + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / window;
    const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / window;
    return Math.sqrt(variance);
  });
}

function calculateSignals(close, window, zBuy, zSell) {
  const ma = rollingMean(close, window);
  const std = rollingStd(close, window);
  const signals = [];
  let holding = false;

  for (let i = 0; i < close.length; i++) {
    if (ma[i] == null || std[i] == null || std[i] === 0) {
      signals.push("HOLD");
      continue;
    }
    const z = (close[i] - ma[i]) / std[i];
    if (!holding && z < zBuy) {
      signals.push("BUY");
      holding = true;
    } else if (holding && z > zSell) {
      signals.push("SELL");
      holding = false;
    } else {
      signals.push("HOLD");
    }
  }

  return { ma, signals };
}

function backtestStrategy(close, signals, startCash = START_CASH) {
  let cash = startCash;
  let shares = 0;
  const portfolio = [];
  const buyDates = [];
  const sellDates = [];
  let trades = 0;

  for (let i = 0; i < close.length; i++) {
    const price = close[i];
    const signal = signals[i];

    if (signal === "BUY" && cash >= price) {
      const available = cash * (1 - TRANSACTION_COST);
      shares = Math.floor(available / price);
      cash -= shares * price * (1 + TRANSACTION_COST);
      buyDates.push(i);
      trades += 1;
    } else if (signal === "SELL" && shares > 0) {
      cash += shares * price * (1 - TRANSACTION_COST);
      shares = 0;
      sellDates.push(i);
      trades += 1;
    }

    portfolio.push(cash + shares * price);
  }

  return { portfolio, buyDates, sellDates, trades };
}

function buyAndHoldSeries(close, startCash = START_CASH) {
  const shares = Math.floor(startCash / close[0]);
  const cash = startCash - shares * close[0];
  return close.map((price) => cash + shares * price);
}

function calculateMetrics(portfolio, close, startCash, trades) {
  const finalValue = portfolio[portfolio.length - 1];
  const totalReturn = (finalValue / startCash - 1) * 100;

  const bhSeries = buyAndHoldSeries(close, startCash);
  const bhReturn = (bhSeries[bhSeries.length - 1] / startCash - 1) * 100;

  const returns = portfolio.slice(1).map((v, i) => v / portfolio[i] - 1);
  const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
  const std = Math.sqrt(returns.reduce((a, b) => a + (b - mean) ** 2, 0) / returns.length);
  const sharpe = std > 0 ? (mean * 52) / (std * Math.sqrt(52)) : 0;
  const volatility = std * Math.sqrt(52) * 100;

  let peak = portfolio[0];
  let maxDrawdown = 0;
  for (const value of portfolio) {
    peak = Math.max(peak, value);
    maxDrawdown = Math.min(maxDrawdown, (value - peak) / peak);
  }

  const winRate = (returns.filter((r) => r > 0).length / returns.length) * 100;
  const calmar = maxDrawdown !== 0 ? totalReturn / Math.abs(maxDrawdown * 100) : 0;

  return {
    final_value: finalValue,
    total_return: totalReturn,
    bh_return: bhReturn,
    excess_return: totalReturn - bhReturn,
    sharpe,
    calmar,
    max_drawdown: maxDrawdown * 100,
    volatility,
    win_rate: winRate,
    trades,
    avg_trade_return: trades > 0 ? totalReturn / trades : 0,
  };
}

async function loadTickerData(symbol) {
  if (dataCache[symbol]) return dataCache[symbol];
  const response = await fetch(`data/${symbol}.json`);
  if (!response.ok) throw new Error(`Missing data for ${symbol}`);
  const payload = await response.json();
  dataCache[symbol] = payload;
  return payload;
}

async function runBacktest(ticker, window, zScore) {
  const stock = await loadTickerData(ticker);
  const spy = await loadTickerData("SPY");

  const dates = stock.dates;
  const close = stock.close;
  const zBuy = -zScore;
  const zSell = zScore;

  const { ma, signals } = calculateSignals(close, window, zBuy, zSell);
  const { portfolio, buyDates, sellDates, trades } = backtestStrategy(close, signals);
  const metrics = calculateMetrics(portfolio, close, START_CASH, trades);
  const buyHold = buyAndHoldSeries(close);

  const spyClose = spy.close;
  const spyPortfolio = buyAndHoldSeries(spyClose);
  const spyReturn = (spyPortfolio[spyPortfolio.length - 1] / START_CASH - 1) * 100;
  metrics.spy_return = spyReturn;
  metrics.excess_vs_spy = metrics.total_return - spyReturn;

  return {
    ticker,
    window,
    z_score: zScore,
    dates,
    close,
    ma,
    portfolio,
    buyHold,
    spyPortfolio,
    buyDates,
    sellDates,
    metrics,
  };
}

window.Backtest = { TICKERS, runBacktest, START_CASH };
