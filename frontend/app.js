const API_URL = window.API_URL.replace(/\/$/, "");

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

function renderStats(metrics, startCash) {
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

async function loadTickers() {
  const response = await fetch(`${API_URL}/api/tickers`);
  const tickers = await response.json();
  tickerSelect.innerHTML = tickers.map((t) =>
    `<option value="${t.symbol}">${t.symbol} — ${t.name}</option>`
  ).join("");
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  hideError();
  results.classList.remove("visible");
  loading.classList.add("visible");
  runBtn.disabled = true;

  const payload = {
    ticker: tickerSelect.value,
    period: "5y",
    window: parseInt(document.getElementById("window").value, 10),
    z_score: parseFloat(document.getElementById("z_score").value),
  };

  document.getElementById("loading-message").textContent =
    `Running backtest for ${payload.ticker}…`;

  try {
    const response = await fetch(`${API_URL}/api/backtest`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.error || "Backtest failed.");
    }

    document.getElementById("results-title").textContent =
      `${data.ticker} Mean Reversion Results`;
    document.getElementById("results-params").textContent =
      `5-year · ${data.window}-week MA · Z-score ±${data.z_score}`;

    document.getElementById("price-chart").src =
      `data:image/png;base64,${data.charts.price}`;
    document.getElementById("portfolio-chart").src =
      `data:image/png;base64,${data.charts.portfolio}`;

    renderStats(data.metrics, data.start_cash);
    results.classList.add("visible");
  } catch (err) {
    showError(err.message || "Could not reach the API. Try again in a moment.");
  } finally {
    loading.classList.remove("visible");
    runBtn.disabled = false;
  }
});

loadTickers().catch(() => {
  tickerSelect.innerHTML = `
    <option value="AAPL">AAPL — Apple</option>
    <option value="MSFT">MSFT — Microsoft</option>
    <option value="GOOGL">GOOGL — Alphabet</option>
    <option value="JPM">JPM — JPMorgan</option>
    <option value="XOM">XOM — Exxon Mobil</option>
    <option value="COP">COP — ConocoPhillips</option>
    <option value="JNJ">JNJ — Johnson & Johnson</option>
    <option value="TSLA">TSLA — Tesla</option>
  `;
});
