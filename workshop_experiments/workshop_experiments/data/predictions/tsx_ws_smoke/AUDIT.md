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
