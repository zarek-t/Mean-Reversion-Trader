const CHART_DEFAULTS = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      labels: { color: "#e8edf4", boxWidth: 12 },
    },
  },
  scales: {
    x: {
      ticks: { color: "#8b9cb3", maxTicksLimit: 8 },
      grid: { color: "rgba(45, 58, 79, 0.4)" },
    },
    y: {
      ticks: { color: "#8b9cb3" },
      grid: { color: "rgba(45, 58, 79, 0.4)" },
    },
  },
};

let priceChart = null;
let portfolioChart = null;

function destroyCharts() {
  if (priceChart) priceChart.destroy();
  if (portfolioChart) portfolioChart.destroy();
}

function renderCharts(result) {
  destroyCharts();

  const buyData = result.close.map((price, i) =>
    result.buyDates.includes(i) ? price : null
  );
  const sellData = result.close.map((price, i) =>
    result.sellDates.includes(i) ? price : null
  );

  priceChart = new Chart(document.getElementById("price-chart"), {
    type: "line",
    data: {
      labels: result.dates,
      datasets: [
        {
          label: "Price",
          data: result.close,
          borderColor: "#e8edf4",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          pointRadius: 0,
        },
        {
          label: `${result.window}-Week MA`,
          data: result.ma,
          borderColor: "#8b9cb3",
          backgroundColor: "transparent",
          borderWidth: 1.5,
          borderDash: [4, 4],
          pointRadius: 0,
        },
        {
          label: "Buy",
          data: buyData,
          borderColor: "#22c55e",
          backgroundColor: "#22c55e",
          pointStyle: "triangle",
          pointRadius: 7,
          showLine: false,
        },
        {
          label: "Sell",
          data: sellData,
          borderColor: "#ef4444",
          backgroundColor: "#ef4444",
          pointStyle: "triangle",
          rotation: 180,
          pointRadius: 7,
          showLine: false,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        title: {
          display: true,
          text: `${result.ticker} Price & Signals`,
          color: "#e8edf4",
        },
      },
    },
  });

  portfolioChart = new Chart(document.getElementById("portfolio-chart"), {
    type: "line",
    data: {
      labels: result.dates,
      datasets: [
        {
          label: "Strategy",
          data: result.portfolio,
          borderColor: "#3b82f6",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
        },
        {
          label: "Buy & Hold",
          data: result.buyHold,
          borderColor: "#f97316",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
        },
        {
          label: "S&P 500",
          data: result.spyPortfolio,
          borderColor: "#a855f7",
          backgroundColor: "transparent",
          borderWidth: 2,
          pointRadius: 0,
        },
      ],
    },
    options: {
      ...CHART_DEFAULTS,
      plugins: {
        ...CHART_DEFAULTS.plugins,
        title: {
          display: true,
          text: "Portfolio Comparison",
          color: "#e8edf4",
        },
      },
    },
  });
}

window.Charts = { renderCharts };
