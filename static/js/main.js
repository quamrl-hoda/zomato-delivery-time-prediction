// ══════════════════════════════════════════════════════
//  ZomatoML · main.js
// ══════════════════════════════════════════════════════

// ── Feature lists ─────────────────────────────────────
const FIELDS_NUM = ["age", "ratings", "pickup_time_minutes", "distance",
                    "vehicle_condition", "multiple_deliveries"];
const FIELDS_CAT = [
  "weather", "type_of_order", "type_of_vehicle", "festival",
  "city_type", "is_weekend", "order_time_of_day",
  "traffic", "distance_type"
];

// ── State ──────────────────────────────────────────────
let predHistory = [];
let chartInstance = null;
let lastInputData = null;

// ══════════════════════════════════════════════════════
//  PAGE ROUTING
// ══════════════════════════════════════════════════════
function showPage(name) {
  document.querySelectorAll(".page").forEach(p => p.classList.remove("active"));
  document.querySelectorAll(".nav-btn").forEach(b => b.classList.remove("active"));
  document.getElementById("page-" + name).classList.add("active");
  const navBtn = document.getElementById("nav-" + name);
  if (navBtn) navBtn.classList.add("active");
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ══════════════════════════════════════════════════════
//  FORM HELPERS
// ══════════════════════════════════════════════════════
function getFormData() {
  const data = {}, missing = [];
  FIELDS_NUM.forEach(f => {
    const v = document.getElementById(f)?.value.trim();
    if (!v) missing.push(f); else data[f] = parseFloat(v);
  });
  FIELDS_CAT.forEach(f => {
    const v = document.getElementById(f)?.value;
    if (!v) missing.push(f); else data[f] = v;
  });
  return { data, missing };
}

function showErr(msg) {
  const el = document.getElementById("form-err");
  el.textContent = msg; el.style.display = "block";
  setTimeout(() => el.style.display = "none", 5000);
}

function resetForm() {
  FIELDS_NUM.forEach(f => { const el = document.getElementById(f); if (el) el.value = ""; });
  FIELDS_CAT.forEach(f => { const el = document.getElementById(f); if (el) el.value = ""; });
  // back to idle
  document.getElementById("result-idle").style.display  = "flex";
  document.getElementById("result-active").style.display = "none";
  document.getElementById("result-card").style.alignItems = "center";
  document.getElementById("result-card").style.justifyContent = "center";
  document.getElementById("r-bar").style.width = "0%";
}

function resetAll() {
  resetForm();
  predHistory = [];
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  renderHistory();
}

// ══════════════════════════════════════════════════════
//  PREDICTION
// ══════════════════════════════════════════════════════
async function singlePredict() {
  const btn = document.getElementById("pred-btn");
  const { data, missing } = getFormData();

  if (missing.length) {
    showErr("Please fill in: " + missing.join(", "));
    return;
  }

  btn.classList.add("loading");

  try {
    const res  = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const json = await res.json();

    if (json.prediction !== undefined) {
      const val = parseFloat(json.prediction).toFixed(2);
      lastInputData = data;
      showResult(val, data);
      addToHistory(data, val);
    } else {
      showErr(json.error || "Unknown error from server.");
    }
  } catch (e) {
    showErr("Request failed: " + e.message);
  } finally {
    btn.classList.remove("loading");
  }
}

function showResult(val, data) {
  document.getElementById("result-idle").style.display   = "none";
  document.getElementById("result-active").style.display = "block";
  document.getElementById("result-card").style.alignItems    = "flex-start";
  document.getElementById("result-card").style.justifyContent = "flex-start";

  document.getElementById("r-val").textContent = val;

  // Animate bar (cap 120 min)
  document.getElementById("r-bar").style.width = "0%";
  setTimeout(() => {
    document.getElementById("r-bar").style.width =
      Math.min(100, (parseFloat(val) / 120) * 100) + "%";
  }, 80);

  // Meta chips
  const meta = document.getElementById("res-meta");
  meta.innerHTML = [
    `<span class="meta-chip"><strong>Traffic:</strong> ${data.traffic}</span>`,
    `<span class="meta-chip"><strong>Distance:</strong> ${data.distance} km</span>`,
    `<span class="meta-chip"><strong>Vehicle:</strong> ${data.type_of_vehicle}</span>`,
    `<span class="meta-chip"><strong>City:</strong> ${data.city_type}</span>`,
    `<span class="meta-chip"><strong>Weather:</strong> ${data.weather}</span>`,
    `<span class="meta-chip"><strong>Weekend:</strong> ${data.is_weekend}</span>`,
  ].join("");
}

function predictAgain() {
  document.getElementById("result-idle").style.display   = "flex";
  document.getElementById("result-active").style.display = "none";
  document.getElementById("result-card").style.alignItems    = "center";
  document.getElementById("result-card").style.justifyContent = "center";
  document.getElementById("r-bar").style.width = "0%";
  // scroll to top of form
  document.getElementById("page-predict").scrollTo({ top: 0 });
}

// ══════════════════════════════════════════════════════
//  HISTORY
// ══════════════════════════════════════════════════════
function nowTime() {
  return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function addToHistory(data, prediction) {
  predHistory.unshift({
    id:         predHistory.length + 1,
    time:       nowTime(),
    traffic:    data.traffic,
    distance:   data.distance,
    vehicle:    data.type_of_vehicle,
    weather:    data.weather,
    city:       data.city_type,
    prediction: parseFloat(prediction).toFixed(2),
  });
  renderHistory();
  // update nav badge
  document.getElementById("hist-count").textContent =
    `${predHistory.length} prediction${predHistory.length > 1 ? "s" : ""} this session`;
}

function renderHistory() {
  const empty   = document.getElementById("hist-empty");
  const content = document.getElementById("hist-content");

  if (predHistory.length === 0) {
    empty.style.display   = "flex";
    content.style.display = "none";
    document.getElementById("hist-count").textContent = "0 predictions this session";
    return;
  }

  empty.style.display   = "none";
  content.style.display = "block";

  // Stats
  const vals = predHistory.map(e => parseFloat(e.prediction));
  document.getElementById("stat-avg").textContent   = (vals.reduce((a,b)=>a+b,0)/vals.length).toFixed(1);
  document.getElementById("stat-min").textContent   = Math.min(...vals).toFixed(1);
  document.getElementById("stat-max").textContent   = Math.max(...vals).toFixed(1);
  document.getElementById("stat-total").textContent = predHistory.length;

  // Table
  document.getElementById("hist-body").innerHTML = predHistory.map((e, i) => `
    <tr class="${i === 0 ? "new-row" : ""}">
      <td style="color:var(--muted)">${e.id}</td>
      <td>${e.time}</td>
      <td class="traffic-${e.traffic}">${e.traffic}</td>
      <td>${e.distance} km</td>
      <td>${e.vehicle}</td>
      <td>${e.weather}</td>
      <td>${e.city}</td>
      <td class="pred-val">${e.prediction} min</td>
    </tr>
  `).join("");

  // Chart
  renderChart([...predHistory].reverse());
}

function renderChart(chrono) {
  const canvas = document.getElementById("trend-chart");
  const labels = chrono.map(e => `#${e.id}`);
  const values = chrono.map(e => parseFloat(e.prediction));

  if (chartInstance) {
    chartInstance.data.labels = labels;
    chartInstance.data.datasets[0].data = values;
    chartInstance.update("active");
    return;
  }

  const load = () => buildChart(canvas, labels, values);
  if (!window.Chart) {
    const s = document.createElement("script");
    s.src = "https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js";
    s.onload = load;
    document.head.appendChild(s);
  } else { load(); }
}

function buildChart(canvas, labels, values) {
  chartInstance = new Chart(canvas, {
    type: "line",
    data: {
      labels,
      datasets: [{
        data: values,
        borderColor: "#f97316",
        backgroundColor: "rgba(249,115,22,.08)",
        borderWidth: 2,
        pointBackgroundColor: "#f97316",
        pointRadius: 4, pointHoverRadius: 6,
        fill: true, tension: 0.4,
      }]
    },
    options: {
      responsive: true,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: "#12141b", borderColor: "#1e2130", borderWidth: 1,
          titleColor: "#9aa3b8", bodyColor: "#f97316",
          bodyFont: { family: "'IBM Plex Mono'" },
          callbacks: { label: ctx => ` ${ctx.parsed.y.toFixed(2)} min` }
        }
      },
      scales: {
        x: { grid: { color: "rgba(30,33,48,.8)" },
             ticks: { color: "#4a5268", font: { family: "'IBM Plex Mono'", size: 10 } } },
        y: { grid: { color: "rgba(30,33,48,.8)" },
             ticks: { color: "#4a5268", font: { family: "'IBM Plex Mono'", size: 10 },
                      callback: v => v + " min" } }
      }
    }
  });
}

function clearHistory() {
  predHistory = [];
  if (chartInstance) { chartInstance.destroy(); chartInstance = null; }
  renderHistory();
}