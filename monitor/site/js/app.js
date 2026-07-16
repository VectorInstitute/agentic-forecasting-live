// App entry: load data, wire the mock banner + freshness, route between views.

import { loadManifest, loadLeaderboard, loadGaps, isMock } from "./data.js";
import { initOverview } from "./overview.js";
import { initDrilldown } from "./drilldown.js";
import { fmtDate } from "./charts.js";

const views = {
  overview: { el: null, rendered: false },
  drilldown: { el: null, rendered: false },
  twins: { el: null, rendered: false },
  calibration: { el: null, rendered: false },
};

let data = { manifest: null, leaderboard: null, gaps: null };

async function boot() {
  for (const name of Object.keys(views)) views[name].el = document.getElementById(`view-${name}`);

  try {
    const [manifest, leaderboard, gaps] = await Promise.all([loadManifest(), loadLeaderboard(), loadGaps()]);
    data = { manifest, leaderboard, gaps };
  } catch (err) {
    document.getElementById("view-overview").innerHTML =
      `<section class="card"><h2>Could not load data</h2><p class="note">${err.message}</p>` +
      "<p class='note'>Serve this directory over HTTP (see the README): <code>python -m http.server</code> from <code>monitor/site</code>.</p></section>";
    return;
  }

  // Mock banner + freshness
  if (isMock(data.manifest, data.leaderboard, data.gaps)) {
    document.getElementById("mock-banner").classList.add("on");
  }
  document.getElementById("freshness").textContent =
    `Latest origin ${fmtDate(data.manifest.latest_origin)} · aggregate generated ${data.manifest.generated_at.replace("T", " ").replace("Z", " UTC")}`;

  wireTabs();
  showView(location.hash.replace("#", "") || "overview");
}

function wireTabs() {
  document.querySelectorAll("nav.tabs button").forEach((btn) => {
    btn.addEventListener("click", () => showView(btn.dataset.view));
  });
}

function showView(name) {
  if (!views[name]) name = "overview";
  document.querySelectorAll("nav.tabs button").forEach((btn) => {
    btn.setAttribute("aria-selected", String(btn.dataset.view === name));
  });
  for (const [key, v] of Object.entries(views)) v.el.hidden = key !== name;
  history.replaceState(null, "", `#${name}`);

  const v = views[name];
  if (v.rendered) return;
  if (name === "overview") initOverview(v.el, data);
  else if (name === "drilldown") initDrilldown(v.el, data);
  else if (name === "twins") renderTwinsStub(v.el);
  else if (name === "calibration") renderCalibrationStub(v.el);
  v.rendered = true;
}

function renderTwinsStub(root) {
  root.innerHTML =
    '<section class="card"><div class="stub">' +
    '<div class="badge">COMING SOON</div>' +
    "<h3>Frozen vs. learning twin</h3>" +
    "<p>Once the adaptive twins deploy (stage 2c), this view plots the two arms' cumulative CRPS side by side — " +
    "the frozen control against the learning twin — with the trailing score gap that the circuit breaker watches.</p>" +
    "<p>Strategy-mutation events (observation → hypothesis → behavioral, with their gate outcomes) will be annotated " +
    "on the timeline, each linking to its version-controlled rationale. The mutation-event schema and mock fixtures already exist " +
    "(<code>data/mutations.json</code>), so this view is a rendering task, not a data one.</p>" +
    "</div></section>";
}

function renderCalibrationStub(root) {
  root.innerHTML =
    '<section class="card"><div class="stub">' +
    '<div class="badge">COMING SOON</div>' +
    "<h3>Calibration deep-dive</h3>" +
    "<p>The overview already surfaces 90% interval coverage per cell. This view will add reliability diagrams " +
    "(nominal vs. empirical quantile coverage) and PIT histograms per method &amp; horizon, plus coverage-over-time.</p>" +
    "<p>All inputs come from the same resolution records already in the contract — no new schema needed.</p>" +
    "</div></section>";
}

boot();
