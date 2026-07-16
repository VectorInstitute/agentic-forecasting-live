// Shared configuration: method ordering, labels, and color roles.
// Method colors follow the dataviz categorical palette in FIXED ORDER
// (never cycled); each maps to a CSS custom property defined in style.css.

// Schema 1.1.0: one method enum value per deployed rung.
export const METHOD_ORDER = [
  "naive",
  "ets",
  "kalman",
  "autoarima",
  "lightgbm",
  "lightgbm_cov",
  "llm_process",
  "llm_process_cov",
  "agent_news",
  "agent_code",
];

export const METHOD_LABEL = {
  naive: "Naive floor",
  ets: "ETS",
  kalman: "Kalman",
  autoarima: "AutoARIMA",
  lightgbm: "LightGBM",
  lightgbm_cov: "LightGBM +cov",
  llm_process: "LLM-Process",
  llm_process_cov: "LLM-Process +cov",
  agent_news: "News agent",
  agent_code: "Code agent",
  adaptive_frozen: "Twin (frozen)",
  adaptive_learning: "Twin (learning)",
};

// CSS variable holding each method's categorical hue. Method *families* share
// a hue (the palette stays at six fixed categorical colors); variants within a
// family are distinguished by their row label, not by color.
export const METHOD_VAR = {
  naive: "--m-naive",
  ets: "--m-classical",
  kalman: "--m-classical",
  autoarima: "--m-classical",
  lightgbm: "--m-lightgbm",
  lightgbm_cov: "--m-lightgbm",
  llm_process: "--m-llm_process",
  llm_process_cov: "--m-llm_process",
  agent_news: "--m-analyst_agent",
  agent_code: "--m-code_agent",
};

export const CONVENTIONAL = new Set(["naive", "ets", "kalman", "autoarima", "lightgbm", "lightgbm_cov"]);

export const HORIZONS = [1, 5, 21];

// Representative model used for the cumulative overview lines so the chart
// stays at 6 series (one per method) rather than the full model matrix.
export const REPRESENTATIVE_MODEL = "gemini-3.5-flash";

export function methodColor(method) {
  const varName = METHOD_VAR[method] || "--text-secondary";
  return `var(${varName})`;
}

export function modelLabel(model) {
  return model === null || model === undefined ? "—" : model;
}

// Compact model name for tight labels (row labels in the fan chart).
const SHORT_MODEL = {
  "gemini-3.1-flash-lite-preview": "gemini-3.1-lite",
  "gemini-3.5-flash": "gemini-3.5",
  "claude-haiku-4.5": "claude-haiku",
};
export function shortModelLabel(model) {
  if (model === null || model === undefined) return "";
  return SHORT_MODEL[model] || model;
}
