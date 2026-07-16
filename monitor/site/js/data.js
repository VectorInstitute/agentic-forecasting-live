// Data loading. Everything is fetched from relative paths under ./data/,
// so the site is a pure static reader of the committed artifacts — no backend.

const BASE = "./data";

async function getJSON(path) {
  const res = await fetch(path, { cache: "no-cache" });
  if (!res.ok) {
    throw new Error(`Failed to load ${path}: ${res.status} ${res.statusText}`);
  }
  return res.json();
}

export function loadManifest() {
  return getJSON(`${BASE}/manifest.json`);
}

export function loadLeaderboard() {
  return getJSON(`${BASE}/leaderboard.json`);
}

export function loadGaps() {
  return getJSON(`${BASE}/gaps.json`);
}

export function loadMutations() {
  return getJSON(`${BASE}/mutations.json`);
}

export function loadForecastBundle(originDate) {
  return getJSON(`${BASE}/forecasts/${originDate}.json`);
}

// A payload is mock when any envelope it came from carries generated_by: "mock".
export function isMock(...payloads) {
  return payloads.some((p) => p && p.generated_by === "mock");
}
