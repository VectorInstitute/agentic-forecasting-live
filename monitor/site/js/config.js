// Shared configuration: method ordering, labels, and color roles.
// Method colors follow the dataviz categorical palette in FIXED ORDER
// (never cycled); each maps to a CSS custom property defined in style.css.

export const METHOD_ORDER = [
  "naive",
  "classical",
  "lightgbm",
  "llm_process",
  "analyst_agent",
  "code_agent",
];

export const METHOD_LABEL = {
  naive: "Naive floor",
  classical: "Classical",
  lightgbm: "LightGBM",
  llm_process: "LLM-Process",
  analyst_agent: "Analyst agent",
  code_agent: "Code agent",
  twin_frozen: "Twin (frozen)",
  twin_learning: "Twin (learning)",
};

// CSS variable holding each method's categorical hue.
export const METHOD_VAR = {
  naive: "--m-naive",
  classical: "--m-classical",
  lightgbm: "--m-lightgbm",
  llm_process: "--m-llm_process",
  analyst_agent: "--m-analyst_agent",
  code_agent: "--m-code_agent",
};

export const CONVENTIONAL = new Set(["naive", "classical", "lightgbm"]);

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
