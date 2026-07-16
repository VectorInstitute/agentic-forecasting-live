// Minimal hand-built SVG charts (no external library — dependency-light).
// Follows the dataviz skill: thin marks (2px lines), recessive grid/axes,
// direct labels + legend for multi-series, a hover/crosshair tooltip layer,
// and a companion table view offered alongside each chart by the caller.

const SVGNS = "http://www.w3.org/2000/svg";

// ---- formatting -----------------------------------------------------------

export function fmtPct(logReturn, digits = 2) {
  // Cumulative log returns are ~= simple returns at these magnitudes; show %.
  const pct = logReturn * 100;
  const sign = pct >= 0 ? "+" : "";
  return `${sign}${pct.toFixed(digits)}%`;
}

export function fmtCrps(v) {
  return v.toFixed(4);
}

export function fmtDate(iso) {
  const d = new Date(`${iso}T00:00:00Z`);
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric", timeZone: "UTC" });
}

// ---- svg helpers ----------------------------------------------------------

function el(name, attrs = {}, parent = null) {
  const node = document.createElementNS(SVGNS, name);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (parent) parent.appendChild(node);
  return node;
}

function makeSvg(container, vbW, vbH) {
  container.innerHTML = "";
  const svg = el("svg", {
    viewBox: `0 0 ${vbW} ${vbH}`,
    width: "100%",
    role: "img",
    preserveAspectRatio: "xMidYMid meet",
  });
  svg.style.display = "block";
  container.appendChild(svg);
  return svg;
}

// ---- tooltip (single shared instance) -------------------------------------

let tipEl = null;
function tooltip() {
  if (!tipEl) {
    tipEl = document.createElement("div");
    tipEl.className = "tooltip";
    document.body.appendChild(tipEl);
  }
  return tipEl;
}
function showTip(html, x, y) {
  const t = tooltip();
  t.innerHTML = html;
  t.classList.add("on");
  const pad = 14;
  let left = x + pad;
  let top = y + pad;
  const w = t.offsetWidth;
  const h = t.offsetHeight;
  if (left + w > window.innerWidth - 8) left = x - w - pad;
  if (top + h > window.innerHeight - 8) top = y - h - pad;
  t.style.left = `${left}px`;
  t.style.top = `${top}px`;
}
function hideTip() {
  if (tipEl) tipEl.classList.remove("on");
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// Bind a tooltip to any DOM node; htmlFn returns the tooltip inner HTML.
export function bindHover(node, htmlFn) {
  node.addEventListener("mousemove", (ev) => showTip(htmlFn(), ev.clientX, ev.clientY));
  node.addEventListener("mouseleave", hideTip);
}

// ---- nice axis ticks ------------------------------------------------------

function niceTicks(min, max, count = 5) {
  const span = max - min || 1;
  const step0 = span / count;
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const norm = step0 / mag;
  let step;
  if (norm < 1.5) step = 1;
  else if (norm < 3) step = 2;
  else if (norm < 7) step = 5;
  else step = 10;
  step *= mag;
  const start = Math.ceil(min / step) * step;
  const ticks = [];
  for (let t = start; t <= max + step * 0.001; t += step) ticks.push(t);
  return ticks;
}

// ---- line chart (cumulative mean CRPS over time) --------------------------

export function lineChart(container, { series, yLabel }) {
  const vbW = 720;
  const vbH = 360;
  // 6 method lines exceed the direct-label threshold of 4 and overlap heavily,
  // so identity comes from the legend + crosshair tooltip, not end labels.
  const m = { top: 16, right: 24, bottom: 34, left: 56 };
  const svg = makeSvg(container, vbW, vbH);
  const iw = vbW - m.left - m.right;
  const ih = vbH - m.top - m.bottom;

  const allX = [];
  const allY = [];
  for (const s of series) {
    for (const p of s.points) {
      allX.push(Date.parse(`${p.x}T00:00:00Z`));
      allY.push(p.y);
    }
  }
  if (allX.length === 0) {
    el("text", { x: vbW / 2, y: vbH / 2, "text-anchor": "middle", class: "tick" }, svg).textContent =
      "No resolved scores yet.";
    return;
  }
  const xMin = Math.min(...allX);
  const xMax = Math.max(...allX);
  const yMin = 0;
  const yMax = Math.max(...allY) * 1.08;

  const xPos = (t) => m.left + ((t - xMin) / (xMax - xMin || 1)) * iw;
  const yPos = (v) => m.top + ih - ((v - yMin) / (yMax - yMin || 1)) * ih;

  // gridlines + y ticks
  for (const t of niceTicks(yMin, yMax, 5)) {
    const y = yPos(t);
    el("line", { class: "grid", x1: m.left, y1: y, x2: m.left + iw, y2: y }, svg);
    el("text", { class: "tick", x: m.left - 8, y: y + 3, "text-anchor": "end" }, svg).textContent = t.toFixed(3);
  }
  // x axis with a few date ticks
  el("line", { class: "axis", x1: m.left, y1: m.top + ih, x2: m.left + iw, y2: m.top + ih }, svg);
  const xTickCount = 5;
  for (let i = 0; i <= xTickCount; i++) {
    const t = xMin + ((xMax - xMin) * i) / xTickCount;
    const x = xPos(t);
    el("text", { class: "tick", x, y: m.top + ih + 16, "text-anchor": "middle" }, svg).textContent = fmtDate(
      new Date(t).toISOString().slice(0, 10),
    );
  }
  el("text", { class: "axis-title", x: 4, y: m.top - 4 }, svg).textContent = yLabel;

  // lines + end labels
  for (const s of series) {
    const pts = s.points
      .slice()
      .sort((a, b) => Date.parse(a.x) - Date.parse(b.x))
      .map((p) => `${xPos(Date.parse(`${p.x}T00:00:00Z`))},${yPos(p.y)}`)
      .join(" ");
    el("polyline", { class: "series-line", points: pts, stroke: cssVar(s.colorVar) }, svg);
  }

  // crosshair + tooltip overlay
  const crosshair = el("line", { class: "axis", x1: 0, y1: m.top, x2: 0, y2: m.top + ih, opacity: 0 }, svg);
  const overlay = el("rect", { x: m.left, y: m.top, width: iw, height: ih, fill: "transparent" }, svg);
  const sortedDates = [...new Set(allX)].sort((a, b) => a - b);

  overlay.addEventListener("mousemove", (ev) => {
    const rect = svg.getBoundingClientRect();
    const scale = vbW / rect.width;
    const localX = (ev.clientX - rect.left) * scale;
    // nearest date
    let nearest = sortedDates[0];
    for (const t of sortedDates) {
      if (Math.abs(xPos(t) - localX) < Math.abs(xPos(nearest) - localX)) nearest = t;
    }
    crosshair.setAttribute("x1", xPos(nearest));
    crosshair.setAttribute("x2", xPos(nearest));
    crosshair.setAttribute("opacity", 0.5);
    const iso = new Date(nearest).toISOString().slice(0, 10);
    let rows = "";
    for (const s of series) {
      const p = s.points.find((q) => Date.parse(q.x) === nearest);
      if (!p) continue;
      rows += `<div class="tt-row"><span><span class="swatch" style="background:${cssVar(
        s.colorVar,
      )}"></span>${s.label}</span><span>${p.y.toFixed(4)}</span></div>`;
    }
    showTip(`<div class="tt-title">${fmtDate(iso)}</div>${rows}`, ev.clientX, ev.clientY);
  });
  overlay.addEventListener("mouseleave", () => {
    crosshair.setAttribute("opacity", 0);
    hideTip();
  });
}

// ---- fan comparison (predictive distributions at one origin/horizon) ------

export function fanComparison(container, { rows, realized, unitLabel }) {
  const rowH = 34;
  const vbW = 720;
  const m = { top: 26, right: 20, bottom: 40, left: 178 };
  const vbH = m.top + m.bottom + rows.length * rowH;
  const svg = makeSvg(container, vbW, vbH);
  const iw = vbW - m.left - m.right;

  // x domain across all quantiles + realized
  const vals = [];
  for (const r of rows) {
    vals.push(r.q[0.05], r.q[0.95]);
  }
  if (realized !== null && realized !== undefined) vals.push(realized);
  const lo = Math.min(...vals);
  const hi = Math.max(...vals);
  const pad = (hi - lo) * 0.08 || 0.01;
  const xMin = lo - pad;
  const xMax = hi + pad;
  const xPos = (v) => m.left + ((v - xMin) / (xMax - xMin || 1)) * iw;

  // x axis (percent)
  el("line", { class: "axis", x1: m.left, y1: m.top + rows.length * rowH, x2: m.left + iw, y2: m.top + rows.length * rowH }, svg);
  for (const t of niceTicks(xMin, xMax, 6)) {
    const x = xPos(t);
    el("line", { class: "grid", x1: x, y1: m.top, x2: x, y2: m.top + rows.length * rowH }, svg);
    el("text", { class: "tick", x, y: m.top + rows.length * rowH + 16, "text-anchor": "middle" }, svg).textContent =
      fmtPct(t, 1);
  }
  el("text", { class: "axis-title", x: m.left, y: m.top + rows.length * rowH + 32 }, svg).textContent = unitLabel;

  // realized reference line spanning all rows
  if (realized !== null && realized !== undefined) {
    const rx = xPos(realized);
    el("line", { class: "realized-line", x1: rx, y1: m.top - 6, x2: rx, y2: m.top + rows.length * rowH }, svg);
    el("text", { class: "tick", x: rx, y: m.top - 10, "text-anchor": "middle", fill: cssVar("--text-primary") }, svg).textContent =
      `realized ${fmtPct(realized, 2)}`;
  }

  rows.forEach((r, i) => {
    const cy = m.top + i * rowH + rowH / 2;
    const color = cssVar(r.colorVar);
    // row label
    el("text", { class: "tick", x: m.left - 10, y: cy + 3, "text-anchor": "end", fill: cssVar("--text-secondary") }, svg).textContent =
      r.label;
    // outer band 0.05-0.95
    el(
      "rect",
      { class: "band-outer", x: xPos(r.q[0.05]), y: cy - 7, width: xPos(r.q[0.95]) - xPos(r.q[0.05]), height: 14, rx: 4, fill: color },
      svg,
    );
    // inner band 0.20-0.80
    el(
      "rect",
      { class: "band-inner", x: xPos(r.q[0.2]), y: cy - 7, width: xPos(r.q[0.8]) - xPos(r.q[0.2]), height: 14, rx: 4, fill: color },
      svg,
    );
    // median dot
    el("circle", { class: "median-dot", cx: xPos(r.q[0.5]), cy, r: 4.5, fill: color }, svg);

    // hover target
    const hit = el("rect", { x: m.left, y: cy - rowH / 2, width: iw, height: rowH, fill: "transparent" }, svg);
    hit.addEventListener("mousemove", (ev) => {
      const crpsRow =
        r.crps === null || r.crps === undefined
          ? '<div class="tt-row tt-muted"><span>CRPS</span><span>pending</span></div>'
          : `<div class="tt-row"><span>CRPS</span><span>${fmtCrps(r.crps)}</span></div>`;
      showTip(
        `<div class="tt-title"><span class="swatch" style="background:${color}"></span>${r.fullLabel}</div>` +
          `<div class="tt-row"><span>median</span><span>${fmtPct(r.q[0.5])}</span></div>` +
          `<div class="tt-row tt-muted"><span>80% band</span><span>${fmtPct(r.q[0.1])} … ${fmtPct(r.q[0.9])}</span></div>` +
          `<div class="tt-row tt-muted"><span>90% band</span><span>${fmtPct(r.q[0.05])} … ${fmtPct(r.q[0.95])}</span></div>` +
          crpsRow,
        ev.clientX,
        ev.clientY,
      );
    });
    hit.addEventListener("mouseleave", hideTip);
  });
}
