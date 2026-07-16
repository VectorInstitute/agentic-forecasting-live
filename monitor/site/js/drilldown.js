// Single-forecast drill-down: all methods' predictive distributions at one
// origin against the realized value, plus the committed agent rationale,
// curated trace summary, and Langfuse trace id.

import { HORIZONS, METHOD_ORDER, METHOD_LABEL, METHOD_VAR, shortModelLabel } from "./config.js";
import { fanComparison, fmtCrps, fmtPct, fmtDate } from "./charts.js";
import { loadForecastBundle } from "./data.js";

let state = { manifest: null, horizon: 5, originDate: null, bundle: null, rationaleKey: null };
let hosts = {};

export function initDrilldown(root, { manifest }) {
  state.manifest = manifest;
  root.innerHTML = "";

  const card = document.createElement("section");
  card.className = "card";
  card.innerHTML =
    "<h2>Single-forecast drill-down</h2>" +
    '<p class="note">Every method &amp; model\'s predictive distribution at one origin, drawn against the realized outcome. Bands show the 90% (0.05-0.95) and 80% (0.10-0.90) intervals; the dot is the median.</p>';

  const controls = document.createElement("div");
  controls.className = "controls";

  const originLabel = document.createElement("label");
  originLabel.textContent = "Origin:";
  const originSelect = document.createElement("select");
  originSelect.addEventListener("change", () => {
    state.originDate = originSelect.value;
    loadAndRender();
  });
  originLabel.appendChild(originSelect);

  const hLabel = document.createElement("label");
  hLabel.textContent = "Horizon:";
  const seg = document.createElement("div");
  seg.className = "seg";
  HORIZONS.forEach((h) => {
    const b = document.createElement("button");
    b.textContent = `h = ${h}`;
    b.setAttribute("aria-pressed", String(h === state.horizon));
    b.addEventListener("click", () => {
      state.horizon = h;
      seg.querySelectorAll("button").forEach((bb, i) => bb.setAttribute("aria-pressed", String(HORIZONS[i] === h)));
      renderBundle();
    });
    seg.appendChild(b);
  });
  hLabel.appendChild(seg);

  controls.appendChild(originLabel);
  controls.appendChild(hLabel);
  card.appendChild(controls);

  const fanHost = document.createElement("figure");
  fanHost.className = "chart";
  card.appendChild(fanHost);

  const legend = document.createElement("div");
  legend.className = "legend";
  card.appendChild(legend);

  const rationaleCard = document.createElement("section");
  rationaleCard.className = "card";

  root.appendChild(card);
  root.appendChild(rationaleCard);
  hosts = { originSelect, seg, fanHost, legend, rationaleCard };

  // populate origins (most recent first)
  const origins = manifest.origins.slice().reverse();
  originSelect.innerHTML = origins
    .map((o) => `<option value="${o.origin_date}">${o.origin_date}${o.resolved_horizons.length === 0 ? " (unresolved)" : ""}</option>`)
    .join("");
  // default to most recent origin whose selected horizon has resolved
  const def = origins.find((o) => o.resolved_horizons.includes(state.horizon)) || origins[0];
  state.originDate = def.origin_date;
  originSelect.value = def.origin_date;
  loadAndRender();
}

async function loadAndRender() {
  hosts.fanHost.innerHTML = '<p class="note">Loading…</p>';
  try {
    state.bundle = await loadForecastBundle(state.originDate);
  } catch (err) {
    hosts.fanHost.innerHTML = `<p class="note">Could not load forecast bundle: ${err.message}</p>`;
    return;
  }
  renderBundle();
}

function qmap(horizonForecast) {
  const m = {};
  for (const p of horizonForecast.quantiles) m[p.quantile] = p.value;
  return m;
}

function renderBundle() {
  const { bundle, horizon } = state;
  if (!bundle) return;

  const realizedEntry = bundle.realized.find((r) => r.horizon === horizon);
  const realized = realizedEntry ? realizedEntry.realized_value : null;

  const crpsFor = (method, model) => {
    const r = bundle.resolutions.find((x) => x.method === method && x.model === model && x.horizon === horizon);
    return r ? r.crps : null;
  };

  const rows = [];
  const methodRank = (m) => METHOD_ORDER.indexOf(m);
  const preds = bundle.predictions
    .filter((p) => p.horizons.some((h) => h.horizon === horizon))
    .sort((a, b) => methodRank(a.method) - methodRank(b.method) || String(a.model).localeCompare(String(b.model)));

  for (const p of preds) {
    const hf = p.horizons.find((h) => h.horizon === horizon);
    rows.push({
      method: p.method,
      model: p.model,
      label: `${METHOD_LABEL[p.method]}${p.model ? ` · ${shortModelLabel(p.model)}` : ""}`,
      fullLabel: `${METHOD_LABEL[p.method]}${p.model ? ` · ${p.model}` : ""} · h=${horizon}`,
      colorVar: METHOD_VAR[p.method],
      q: qmap(hf),
      crps: crpsFor(p.method, p.model),
    });
  }
  // resolved rows sorted best-CRPS first; unresolved rows keep method order after.
  rows.sort((a, b) => {
    if (a.crps === null && b.crps === null) return 0;
    if (a.crps === null) return 1;
    if (b.crps === null) return -1;
    return a.crps - b.crps;
  });

  fanComparison(hosts.fanHost, {
    rows,
    realized,
    unitLabel: `cumulative log return at h=${horizon} (%)`,
  });

  hosts.legend.innerHTML =
    METHOD_ORDER.map(
      (m) => `<span class="item"><span class="key" style="background:var(${METHOD_VAR[m]})"></span>${METHOD_LABEL[m]}</span>`,
    ).join("") +
    (realized === null
      ? '<span class="item tt-muted">horizon not yet resolved — no realized line</span>'
      : `<span class="item"><span class="key" style="background:var(--text-primary)"></span>realized ${fmtPct(realized, 2)}</span>`);

  renderRationale();
}

function renderRationale() {
  const { bundle } = state;
  const card = hosts.rationaleCard;
  const withRationale = bundle.predictions.filter((p) => p.rationale && p.rationale.trim().length > 0);

  card.innerHTML = "<h2>Agent rationale &amp; curated trace</h2>";
  if (withRationale.length === 0) {
    card.innerHTML += '<p class="rationale empty">No agent-authored rationale at this origin (conventional methods only).</p>';
    appendPolicy(card);
    return;
  }

  const controls = document.createElement("div");
  controls.className = "controls";
  const label = document.createElement("label");
  label.textContent = "Method:";
  const select = document.createElement("select");
  select.innerHTML = withRationale
    .map((p) => `<option value="${p.predictor_id}">${METHOD_LABEL[p.method]}${p.model ? ` · ${p.model}` : ""}</option>`)
    .join("");
  const keys = withRationale.map((p) => p.predictor_id);
  if (!keys.includes(state.rationaleKey)) state.rationaleKey = keys[0];
  select.value = state.rationaleKey;
  select.addEventListener("change", () => {
    state.rationaleKey = select.value;
    fillRationale(card, withRationale);
  });
  label.appendChild(select);
  controls.appendChild(label);
  card.appendChild(controls);

  const body = document.createElement("div");
  body.id = "rationale-body";
  card.appendChild(body);
  fillRationale(card, withRationale);
  appendPolicy(card);
}

function fillRationale(card, preds) {
  const body = card.querySelector("#rationale-body");
  const p = preds.find((x) => x.predictor_id === state.rationaleKey) || preds[0];
  const tools = (p.curated_trace_summary && p.curated_trace_summary.tool_calls) || [];
  const toolHtml = tools.length
    ? `<ul class="trace-list">${tools
        .map((t) => `<li><span class="tool">${t.tool}</span> — <span class="q">${t.query_title}</span></li>`)
        .join("")}</ul>`
    : '<p class="rationale empty">No tool calls in this run.</p>';
  const trace = p.langfuse_trace_id
    ? `<span class="meta-chip" title="Langfuse trace id — authorized users can open the full trace internally">trace ${p.langfuse_trace_id.slice(0, 12)}…</span>`
    : "";
  body.innerHTML =
    `<p class="rationale">${p.rationale}</p>` +
    `<div style="margin-top:10px"><strong style="font-size:0.85rem">Curated trace summary</strong> ${trace}</div>` +
    toolHtml;
}

function appendPolicy(card) {
  const div = document.createElement("div");
  div.className = "policy";
  div.innerHTML =
    "<strong>Curation policy.</strong> Agent-authored rationales and tool-call summaries (tool names + query titles) are PUBLIC. " +
    "Raw retrieved article text and internal prompt scaffolding are NOT public. Langfuse trace ids are shown so authorized users can open full traces internally.";
  card.appendChild(div);
}
