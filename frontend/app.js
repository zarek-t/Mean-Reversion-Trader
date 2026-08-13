const form = document.getElementById("backtest-form");
const loading = document.getElementById("loading");
const results = document.getElementById("results");
const errorBanner = document.getElementById("error-banner");
const runBtn = document.getElementById("run-btn");
const tickerSelect = document.getElementById("ticker");

function showError(message) {
  errorBanner.textContent = message;
  errorBanner.classList.add("visible");
}

function hideError() {
  errorBanner.classList.remove("visible");
}

function formatMoney(value) {
  return "$" + Number(value).toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
}

function formatPct(value) {
  const sign = value >= 0 ? "+" : "";
  return sign + Number(value).toFixed(2) + "%";
}

function pctClass(value) {
  if (value > 0) return "positive";
  if (value < 0) return "negative";
  return "";
}

function renderStats(metrics) {
  const stats = [
    ["Final Value", formatMoney(metrics.final_value)],
    ["Strategy Return", formatPct(metrics.total_return), pctClass(metrics.total_return)],
    ["Buy & Hold", formatPct(metrics.bh_return), pctClass(metrics.bh_return)],
    ["S&P 500", formatPct(metrics.spy_return), pctClass(metrics.spy_return)],
    ["vs Buy & Hold", formatPct(metrics.excess_return), pctClass(metrics.excess_return)],
    ["vs S&P 500", formatPct(metrics.excess_vs_spy), pctClass(metrics.excess_vs_spy)],
    ["Sharpe", metrics.sharpe.toFixed(2)],
    ["Max Drawdown", formatPct(metrics.max_drawdown), "negative"],
    ["Win Rate", metrics.win_rate.toFixed(1) + "%"],
    ["Trades", String(metrics.trades)],
  ];

  document.getElementById("stats-grid").innerHTML = stats.map(([label, value, cls]) => `
    <div class="stat-card">
      <div class="label">${label}</div>
      <div class="value ${cls || ""}">${value}</div>
    </div>
  `).join("");
}

function initTickers() {
  tickerSelect.innerHTML = Backtest.TICKERS.map((t) =>
    `<option value="${t.symbol}">${t.symbol} — ${t.name}</option>`
  ).join("");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  results.classList.remove("visible");
  loading.classList.add("visible");
  runBtn.disabled = true;

  const ticker = tickerSelect.value;
  const windowSize = parseInt(document.getElementById("window").value, 10);
  const zScore = parseFloat(document.getElementById("z_score").value);

  document.getElementById("loading-message").textContent = `Running ${ticker}…`;

  try {
    const result = await Backtest.runBacktest(ticker, windowSize, zScore);

    document.getElementById("results-title").textContent =
      `${result.ticker} Mean Reversion Results`;
    document.getElementById("results-params").textContent =
      `5-year · ${result.window}-week MA · Z-score ±${result.z_score}`;

    Charts.renderCharts(result);
    renderStats(result.metrics);
    results.classList.add("visible");
  } catch (err) {
    showError(err.message || "Backtest failed.");
  } finally {
    loading.classList.remove("visible");
    runBtn.disabled = false;
  }
});

initTickers();

// Preload default ticker data in background for instant first run
Backtest.loadTickerData("AAPL");
Backtest.loadTickerData("SPY");
