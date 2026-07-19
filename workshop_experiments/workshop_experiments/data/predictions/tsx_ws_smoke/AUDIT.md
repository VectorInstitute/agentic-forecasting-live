# Trace audit log — tsx_ws_smoke

Per plan §2.2b: no method's results are trusted until at least one full Langfuse
trace per method has been read — rendered prompt in, completion out, usage recorded,
tool trees present, correct span linkage. Audited by the orchestrator.

| Method | Model | Trace id | What was verified | Date |
|---|---|---|---|---|
| llmp_qgrid (precision-5) | gemini-3.1-flash-lite-preview | `1240eea25af775d86f5a616b7b7ca0fc` | GENERATION nested under named span; 5-decimal history in user message; usage 1763/3333 | 2026-07-16 |
| llmp_qgrid (post-tracing-fix verification) | gemini-3.1-flash-lite-preview | `e1d72dfc14dcdb775b92c963a3227a58` | Full prompt+completion captured; usage 1568/170; parentage correct | 2026-07-16 |
| agent_news | claude-sonnet-4-6 | `2e5eb1d6c66859bb626087d0f2f8e6bc` | Clean run: parallel search batch, tool tree, rationale, submit; 2–3 LLM calls | 2026-07-16 |
| agent_code (capped) | gemini-3.5-flash | `66d307d2e79aae708491464c3b72a2d3`, `44159bbfa7a45164220332855db3099c` | Cap hit → graceful final turn → valid 11-quantile forecast; 5–10 searches + 7–10 code runs before submit | 2026-07-16 |
| agent_news (proxy-bug failure mode) | gemini-3.5-flash | `08cc9d60ff8463047b68c1455d49e364` | Blocking 400 (function_response.name) documented; mitigations merged (serial-tool instruction + narrow retry); gap refilled post-fix | 2026-07-16 |

Known limitations at audit time: LLMP prediction-metadata trace ids are null (traces
correct; linkage fix tracked); curated_trace_summary population depends on the
tool-call capture path (merged; verified in live runs as they accumulate).

Pre-fix LLMP smoke artifacts (quantized precision-2 inputs) were force-refreshed on
2026-07-16; every committed LLMP prediction in this store is precision-5.

**Stage-2 (retrospective weekly LLMP) reads, 2026-07-17:**

| Method | Model | Trace id | What was verified | Date |
|---|---|---|---|---|
| llmp_qgrid | claude-sonnet-5 | `a0fe96b8aafba13018f77d4163b7c95a` | 5-decimal history; thinking-heavy completion (usage 2002/14463); late-Mar-2026 war-drawdown origin, coherent rebound-shaped quantiles; nesting correct | 2026-07-17 |
| llmp_qgrid_cov | gpt-5.4 | `96a580199319ef70b1d970fbf4483165` | Covariate block rendered (VIX et al.); 5-decimal history; usage 10978/1736; nesting correct | 2026-07-17 |

Outstanding per-method reads to append as their stages run: llmp_qgrid on the
remaining matrix models (incl. sonnet-5, opus-4-8 spot reads), agent_code
(sonnet-4-6), adaptive study transcript + one pre/post eval trace.

**Track-2 scenario + judge reads, 2026-07-18:**

| Artifact | Model | Trace id | What was verified | Date |
|---|---|---|---|---|
| scenario writeup 2026-03-31 (war low) | gemini-3.1-flash-lite-preview | `cc78e786ff43fc5b0129386abfb1b067` | Full read: 3 scenarios, probabilities sum to 1.0, ranged outlooks; base case (0.55, +3–5%) got direction right via the WRONG mechanism (persistent friction vs. actual ceasefire relief rally) — flagship right-direction/wrong-mechanism exhibit | 2026-07-18 |
| scenario writeup 2025-04-01 (tariff eve) | gemini-3.1-flash-lite-preview | — (meta.yaml) | Full read: retrieved facts check out vs. timeline (25% non-CUSMA tariffs, Mar-12 BoC cut); under-reaction to the publicly scheduled Apr-2 escalation — bear case had right driver at 0.15/−5–8% vs. realized −12.8% window | 2026-07-18 |
| judge verdicts, all 4 origins | claude-sonnet-4-6 | judge.yaml per origin | First post-PR#25 run: 4/4 verdicts written, integer 1–5 scores valid, justifications grounded in realized-return context; drivers never scored >3 (judge correctly withholds mechanism credit it cannot verify from returns alone — mechanism attribution requires the human timeline comparison) | 2026-07-18 |

**Adaptive study (tsx_study) — incident + first-look facts, 2026-07-18:**

- Study completed overnight: 50/50 turns, checkpoints every 10, 4,824 input /
  143,810 output tokens, 4,298s wall. Transcript: study_runs/tsx_study/transcript.jsonl.
- **Incident:** the morning re-invocation (retro-day.sh attempt 1) force-reseeded
  the trained strategy before its resume check, wiping 25 calibration corrections /
  28 hypotheses / 82 observations / 48 version entries while replaying 0 turns.
  Recovered in full from `.history/skill_state_20260718T085038Z.yaml` (restored via
  the store's own writer; restore is itself versioned). Guard shipped as the
  study-resume-no-reseed PR. Off-repo backup taken.
- **Study trace read (Langfuse) COMPLETE, 2026-07-18** — 50 traces, one per turn
  (name `tsx-adaptive-study`; sampled first `5e16587a…`, mid `93964196…`, distill
  `01c6332a…`). Verified:
  (1) the transcript's "4.8k input tokens" is driver-prompt-only accounting — the
  actual LLM calls carry the full session (usage grows 5k → 533k input by the
  distill turn); tool results DID enter context. Transcript accounting should
  report real LLM usage (minor fix, noted for #8 scope).
  (2) Empirical work is REAL: turn 1 alone runs ~20 `run_code` E2B executions
  (multi-KB code + results) after loading its skills; the sampled mid-study
  computation is a "Canadian Holiday catch-up study" — yfinance download +
  calendar alignment of TSX-open/US-closed days (independently discovering the
  same cross-calendar seam as today's covariate-panel fix), erroring and
  iterating like a genuine analysis loop. One `search_web` observed in turn 1.
  (3) Mutations flow through the gated interface: record_observation →
  open_hypothesis → 3× record_hypothesis_outcome → graduate_hypothesis; the k=3
  letter of the gate is enforced.
  (4) **Finding for §5:** the 3 confirmations are recorded back-to-back within
  the same turn with identical tiny payloads (~52 bytes each) — the gate's
  letter is satisfied by one analysis asserting thrice, not by independent
  evidence over time. 28 graduations in one session pattern-matches the
  overzealous-self-update risk in its structural form. Study model:
  gemini-3.5-flash (hence untouched by the Anthropic schema-bug window).
- Adaptive-eval traces owed after that stage runs.

Judge design note: the judge's realized-outcome context is returns-only (5/21/60d),
deliberately not a curated news summary — so calibration/specificity verdicts are
fully automated while mechanism attribution (drivers) is conservative by
construction. Right-direction/wrong-mechanism cases are caught by the human audit
read, not the judge; this division of labor is a finding, not a gap.
