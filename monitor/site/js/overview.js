// Overview view: KPI row, leaderboard heatmap, cumulative-CRPS lines, gap log.

import { HORIZONS, METHOD_ORDER, METHOD_LABEL, METHOD_VAR, CONVENTIONAL, REPRESENTATIVE_MODEL, modelLabel } from "./config.js";
import { lineChart, bindHover, fmtCrps, fmtDate } from "./charts.js";

let state = { manifest: null, leaderboard: null, gaps: null, horizon: 5, showTable: false };

export function initOverview(root, { manifest, leaderboard, gaps }) {
  state = { ...state, manifest, leaderboard, gaps };
  render(root);
}

function render(root) {
  root.innerHTML = "";
  root.appendChild(kpiRow());
  root.appendChild(heatmapCard());
  root.appendChild(cumulativeCard());
  root.appendChild(gapCard());
}

// ---- KPI row --------------------------------------------------------------

function stat(label, value, sub) {
  const d = document.createElement("div");
  d.className = "stat";
  d.innerHTML = `<div class="label">${label}</div><div class="value">${value}</div><div class="sub">${sub || ""}</div>`;
  return d;
}

function kpiRow() {
  const { manifest, leaderboard } = state;
  const row = document.createElement("div");
  row.className = "kpi-row";
  const totalResolutions = leaderboard.cells.reduce((a, c) => a + c.n, 0);
  row.appendChild(stat("Methods tracked", String(manifest.methods.length), `${manifest.models.length}-model matrix on LLM rungs`));
  row.appendChild(stat("Origins committed", String(manifest.origin_count), `latest ${fmtDate(manifest.latest_origin)}`));
  row.appendChild(stat("Resolved scores", totalResolutions.toLocaleString(), "across all cells & horizons"));
  row.appendChild(stat("Logged gaps", String(manifest.gap_count ?? 0), "never backfilled"));
  return row;
}

// ---- leaderboard heatmap --------------------------------------------------

function cellByKey(method, model, horizon) {
  return state.leaderboard.cells.find(
    (c) => c.method === method && c.model === model && c.horizon === horizon,
  );
}

function rowsForMethod(method) {
  // Conventional methods have a single (model=null) row; LLM methods have one
  // row per model in the matrix.
  if (CONVENTIONAL.has(method)) return [{ method, model: null }];
  return state.manifest.models.map((model) => ({ method, model }));
}

function tintFor(t) {
  // Single-hue sequential: mix a mid-dark blue toward the surface. Low CRPS
  // recedes (near surface); high CRPS reads as saturated. Text stays in ink.
  const pct = Math.round(12 + t * 72);
  return `color-mix(in srgb, var(--seq-550) ${pct}%, var(--surface-1))`;
}

function heatmapCard() {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML =
    "<h2>Leaderboard — mean CRPS by method x model x horizon</h2>" +
    '<p class="note">Lower is better. Cell shade is per-horizon (columns are not comparable across horizons — CRPS grows with horizon). Hover for sample size, coverage, and freshness.</p>';

  // per-horizon min/max for shading
  const bounds = {};
  for (const h of HORIZONS) {
    const vals = state.leaderboard.cells.filter((c) => c.horizon === h).map((c) => c.mean_crps);
    bounds[h] = { min: Math.min(...vals), max: Math.max(...vals) };
  }

  const scroll = document.createElement("div");
  scroll.className = "scroll-x";
  const table = document.createElement("table");
  table.className = "heatmap";
  const thead = document.createElement("thead");
  thead.innerHTML =
    "<tr><th class='rowlab'>Method / model</th>" +
    HORIZONS.map((h) => `<th>h = ${h}</th>`).join("") +
    "</tr>";
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  for (const method of METHOD_ORDER) {
    for (const { model } of rowsForMethod(method)) {
      const tr = document.createElement("tr");
      const lab = document.createElement("th");
      lab.className = "rowlab";
      lab.innerHTML =
        `<span class="swatch" style="background:var(${METHOD_VAR[method]})"></span>` +
        `${METHOD_LABEL[method]}${model ? ` <span class="mm">· ${modelLabel(model)}</span>` : ""}`;
      tr.appendChild(lab);

      for (const h of HORIZONS) {
        const c = cellByKey(method, model, h);
        const td = document.createElement("td");
        if (!c) {
          td.className = "empty";
          td.textContent = "—";
        } else {
          const b = bounds[h];
          const t = (c.mean_crps - b.min) / (b.max - b.min || 1);
          td.style.background = tintFor(t);
          td.textContent = fmtCrps(c.mean_crps);
          const cov = c.coverage_90 === null || c.coverage_90 === undefined ? "n/a" : `${(c.coverage_90 * 100).toFixed(0)}%`;
          bindHover(
            td,
            () =>
              `<div class="tt-title">${METHOD_LABEL[method]}${model ? ` · ${model}` : ""} · h=${h}</div>` +
              `<div class="tt-row"><span>mean CRPS</span><span>${fmtCrps(c.mean_crps)}</span></div>` +
              `<div class="tt-row tt-muted"><span>n resolved</span><span>${c.n}</span></div>` +
              `<div class="tt-row tt-muted"><span>90% coverage</span><span>${cov}</span></div>` +
              `<div class="tt-row tt-muted"><span>updated</span><span>${fmtDate(c.last_updated)}</span></div>`,
          );
        }
        tr.appendChild(td);
      }
      tbody.appendChild(tr);
    }
  }
  table.appendChild(tbody);
  scroll.appendChild(table);
  card.appendChild(scroll);
  return card;
}

// ---- cumulative CRPS over time --------------------------------------------

function cumulativeSeries(horizon) {
  const series = [];
  for (const method of METHOD_ORDER) {
    const model = CONVENTIONAL.has(method) ? null : REPRESENTATIVE_MODEL;
    const entry = state.leaderboard.cumulative.find(
      (c) => c.method === method && c.model === model && c.horizon === horizon,
    );
    if (!entry) continue;
    series.push({
      key: `${method}`,
      label: METHOD_LABEL[method],
      colorVar: METHOD_VAR[method],
      points: entry.series.map((p) => ({ x: p.origin_date, y: p.cumulative_mean_crps })),
    });
  }
  return series;
}

function cumulativeCard() {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML =
    "<h2>Cumulative mean CRPS over time</h2>" +
    `<p class="note">Running mean CRPS as origins resolve, one line per method (LLM rungs shown on <code>${REPRESENTATIVE_MODEL}</code>). A line settling lower is the better forecaster; watch the gaps compound.</p>`;

  const controls = document.createElement("div");
  controls.className = "controls";
  const seg = document.createElement("div");
  seg.className = "seg";
  seg.setAttribute("role", "group");
  seg.setAttribute("aria-label", "Horizon");
  HORIZONS.forEach((h) => {
    const b = document.createElement("button");
    b.textContent = `h = ${h}`;
    b.setAttribute("aria-pressed", String(h === state.horizon));
    b.addEventListener("click", () => {
      state.horizon = h;
      drawChart();
    });
    seg.appendChild(b);
  });
  const label = document.createElement("label");
  label.textContent = "Horizon (business days):";
  controls.appendChild(label);
  controls.appendChild(seg);

  const tableToggle = document.createElement("button");
  tableToggle.className = "seg";
  tableToggle.style.marginLeft = "auto";
  tableToggle.textContent = "Show data table";
  tableToggle.addEventListener("click", () => {
    state.showTable = !state.showTable;
    tableToggle.textContent = state.showTable ? "Hide data table" : "Show data table";
    drawChart();
  });
  controls.appendChild(tableToggle);
  card.appendChild(controls);

  const chartHost = document.createElement("figure");
  chartHost.className = "chart";
  card.appendChild(chartHost);

  const legend = document.createElement("div");
  legend.className = "legend";
  card.appendChild(legend);

  const tableHost = document.createElement("div");
  tableHost.className = "scroll-x";
  card.appendChild(tableHost);

  function drawChart() {
    seg.querySelectorAll("button").forEach((b, i) => b.setAttribute("aria-pressed", String(HORIZONS[i] === state.horizon)));
    const series = cumulativeSeries(state.horizon);
    lineChart(chartHost, { series, yLabel: `cumulative mean CRPS · h=${state.horizon}` });
    legend.innerHTML = series
      .map((s) => `<span class="item"><span class="key" style="background:var(${s.colorVar})"></span>${s.label}</span>`)
      .join("");
    tableHost.innerHTML = "";
    if (state.showTable) tableHost.appendChild(cumulativeTable(series));
  }
  drawChart();
  return card;
}

function cumulativeTable(series) {
  const t = document.createElement("table");
  t.className = "data";
  t.innerHTML =
    "<thead><tr><th>Method</th><th class='num'>n</th><th class='num'>final cumulative mean CRPS</th></tr></thead>";
  const tb = document.createElement("tbody");
  for (const s of series) {
    const last = s.points[s.points.length - 1];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td><span class="swatch" style="background:var(${s.colorVar})"></span>${s.label}</td><td class="num">${last ? s.points.length : 0}</td><td class="num">${last ? last.y.toFixed(4) : "—"}</td>`;
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  return t;
}

// ---- gap log --------------------------------------------------------------

function gapCard() {
  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML =
    "<h2>Gap log</h2>" +
    '<p class="note">Days (or method scopes) that failed to submit after bounded same-evening retries. Gaps are documented facts, never backfilled.</p>';
  const scroll = document.createElement("div");
  scroll.className = "scroll-x";
  const t = document.createElement("table");
  t.className = "data";
  t.innerHTML =
    "<thead><tr><th>Date</th><th>Scope</th><th>Reason</th><th class='num'>Retries</th><th>Logged at</th></tr></thead>";
  const tb = document.createElement("tbody");
  const gaps = state.gaps.gaps.slice().sort((a, b) => (a.date < b.date ? 1 : -1));
  if (gaps.length === 0) {
    tb.innerHTML = "<tr><td colspan='5' class='tt-muted'>No gaps logged.</td></tr>";
  }
  for (const g of gaps) {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${fmtDate(g.date)}</td><td><span class="pill pending">${g.scope}</span></td><td>${g.reason}</td>` +
      `<td class="num">${g.retries_attempted ?? "—"}</td><td class="tt-muted">${g.logged_at.replace("T", " ").replace("Z", " UTC")}</td>`;
    tb.appendChild(tr);
  }
  t.appendChild(tb);
  scroll.appendChild(t);
  card.appendChild(scroll);
  return card;
}
