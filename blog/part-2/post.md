# Forecasting agents that read the news — and what it takes to trust them

**Ethan Jackson, Ali Kore, Behnoosh Zamanlooy & Shayaan Mehdi**

*Part 2 of 2.*

## The staircase

Part 1 ended on a frozen LLM that could emit a genuine predictive distribution
but saw only numbers and a short description of the series — quiet exactly where
the series was about to get loud, because the cause of every regime break was
written in language, not in the price history. The next rung hands the same
forecaster the ability to go out and read the news for itself.

The analyst agent keeps the task identical — a full quantile grid for the log
return at 1, 5, and 21 business days — but before it answers it runs web
searches, reads what it finds, and writes a rationale. The hard part is leakage,
and our guard against it is two-layered and honest about its limits. Each
forecast carries a harness-authoritative *as-of* date the agent cannot override,
and every query is scoped to it; a second LLM then verifies that each retrieved
snippet predates that cutoff and discards anything that doesn't. It is
best-effort, not a proof — a model can still infer the future from context it was
trained on — but it keeps the blatant leaks out. One rung further, the
code-executing analyst gets a sandboxed Python environment and can compute its
own diagnostics — pull the series into a dataframe, measure the trailing
volatility and drawdown, check a claim before committing to it — rather than
eyeballing numbers in a prompt.

What does one forecast actually look like? Take the analyst agent standing on
2026-03-30, at the trough of the war-driven drawdown. It fired six date-scoped
searches — Bank of Canada policy, CPI and the Labour Force Survey, oil and gold,
USD/CAD and GoC yields, US tariff spillovers, TSX sector earnings — then wrote a
rationale that reads like a desk note: an accommodative BoC on hold at 2.25%,
soft 1.8% CPI against a deteriorating labour market (−84K February jobs, 6.7%
unemployment), commodity volatility from Middle-East tension, and 25% US tariffs
as the structural headwind. From that it reasoned a distribution: a modest
positive median (+1%) off a deep mean-reversion anchor, elevated volatility, and
a deliberate negative skew — the left tail fattened for tariff escalation, the
right tail (Q0.95 at +9%) reserved for "tariff relief or commodity-driven surge."
The realized 21-day move, +5.1%, landed almost exactly on its Q0.8 — well inside
the distribution, but in the upper reaches the agent had explicitly reserved for
an outcome it considered unlikely. The rally that produced it lived in that
right tail.

![Anatomy of one agent forecast: search queries, rationale factors, and the
emitted quantile grid.](assets/fig1_agent_anatomy.png)

***Figure 1.** One forecast, end to end: the news analyst's six date-scoped
searches (left, paraphrased), the load-bearing factors from its written
rationale (center), and the 11-point quantile grid it emitted (right) — median
+1%, deliberate negative skew, and the realized 21-day move of +5.1% landing
essentially at its Q0.80.*

## The same scoreboard

An agent that reads is still just another predictor, so it earns its place the
way everything in Part 1 did: slotted into the same rolling-origin sweeps and
scored with CRPS beside the naive floor, the classical methods, LightGBM, and
the frozen LLMPs. Same origins, same cutoff, same referee.

The honest expectation, set by Part 1, is that reading the news does not
automatically win — and at h=1 it doesn't. On the protected 2026 eval the
Part 1 board clusters near CRPS 0.0050 against a naive floor of 0.0093,
and the two news-agent variants land mid-pack at 0.00507 and 0.00509,
indistinguishable from the frozen models: for a one-day-ahead return, where the
move is close to unforecastable, a news scan has little to add over a
well-shaped distribution. The one agent that thrives there is the
code-executing analyst, second of twenty-one, a hair behind
LightGBM-with-covariates — its eleven-code-runs-per-forecast style *is*
statistics, and at one day out, with the whole top of the board inside a 2%
spread and direction a coin flip, that is most of what there is to win. But
step through the horizons and the board reorganizes by *family*. At h=5 the
language models sweep: the top five methods are all LLM-based and the two
LightGBM configurations fall to 14th and 17th of twenty-one — the week horizon
appears to favour the LLMs' fatter, more skeptical distributions over
LightGBM's tight quantiles. At h=21 the guard changes again: the LightGBM
configurations retake the lead, but three of the top seven methods are agents,
led by the news agent (on the lighter model) at third — ahead of every frozen
LLMP on the board.

![Protected-eval leaderboard: all 21 methods at each horizon, agent rungs
highlighted.](assets/fig5_combined_leaderboard.png)

***Figure 2.** The complete protected-window scoreboard, every method family on
one board, ranked within each horizon. The guard changes at every horizon:
LightGBM at h=1, the LLM family sweeping h=5, LightGBM back on top at h=21 with
three agents in the top seven. Mean CRPS ×10⁻³ over n = 24/22/24 resolved
weekly origins at h = 1/5/21; chevrons hold the far-worse floors — the naive
method everywhere, ETS at h=5 and h=21 — off-scale so the ladder stays
legible.*

Two wrinkles keep that story honest. The ranking is model-dependent — the same
agent harness on a heavier model finishes mid-pack at h=21 — so "agents read
the news" is not yet a horizontal claim about agents; it is a configuration
that has to earn its rung, model by model. And the paired comparisons are
weaker than they look: at h=21 the code agent beats its own frozen base model
at eighteen of twenty-four origins, the largest paired count in the study, but
the margins behind the count are small, and a test that weighs them — or
respects how heavily the origins overlap — lands it inside noise. The code
agent also shows none of the news agent's break-window behaviour: it computes
from the same history the conventional methods see, and it fails where they
fail. What an agent *reads* determines where it wins.

Ranks, though, still average over the thing that matters. Split the h=21
origins into the ten inside the war window and the fourteen quiet ones and the
news agent's third place decomposes into two opposite averages: ~11% *better*
than LightGBM-with-covariates at the break, ~22% *worse* on quiet weeks, as
though it pays an LLM-noise tax whenever there is nothing to read. Pushed on,
the tidy story gives. The break-window edge rests on a single origin where
LightGBM blew up and the agent did not — remove that week and 11% becomes 2%,
and the agent records the *worse* score on six of the ten break origins. One
avoided blowup, not a dependable edge when the regime turns.

None of this is a defect in the methods; it is the window. Twenty-four weekly
origins at a one-month horizon overlap into roughly five independent
observations containing one regime event — not enough to resolve differences
this size, in any direction. These are observations to be checked, not results
to build on.

![War-window vs quiet-week CRPS against LightGBM, and paired same-model
deltas at h=21.](assets/fig6_where_agents_earn.png)

***Figure 3.** Left: the news agent versus LightGBM-with-covariates at h=21,
split into war-window (n = 10) and quiet (n = 14) origins — the average hides
two opposite directions, though the break-window edge rests heavily on a single
origin. Right: paired same-model deltas, frozen LLMP to agent, at h=21. The
code agent's 18-of-24 is the largest paired count in the study; the per-origin
margins behind it are small enough that it does not clear a stricter test. One
regime event sampled weekly, not ten independent breaks.*

One more signal hides in the comparison, and it may matter more than the
ranks: *when the agent disagrees with LightGBM*. The gap between the two
quantile grids behaves like an event detector — three of its four largest
values sit at origins bracketing the war drawdown, and the fourth fired on a
perceived risk (an early-June correction off a record high) that never
confirmed. Decompose the gap and the signal lives in the *width*: how much
wider the agent's 10–90 interval runs than LightGBM's tracks the size of the
move that follows, while how far it shifts its *median* tracks nothing at all.
The agent's useful signal is *how uncertain it says it is*, not *which way it
leans* — though the width signal leans heavily on the war window, and at
twenty-four overlapping origins we cannot separate it from luck. We also tried
a divergence-gated router that hands the forecast to the agent whenever the two
disagree sharply. It looks interesting, but there is no way this window has the
statistical power to tell whether it is a real mechanism, so we report it as an
idea rather than an edge.

The descriptive facts, though, stand on their own. At the war trough the
agent's interval ran **three times** LightGBM's — and at one break origin its
median matched LightGBM's almost exactly while its interval ran 2.5× wider: a
pure alarm, no directional bet. LightGBM's interval, built from trailing
volatility, barely moved all half-year. The mechanism on offer is not that the
agent predicts direction better; it is that the agent can notice, from the
news, that the quiet period may be ending — and say so by widening. That
suggests a concrete job in a production pipeline: the agent runs alongside the
conventional forecasters, and when its distribution diverges sharply from
theirs it raises an alert — kicking off a deeper investigation, or bringing in
a human expert with a stake in the prediction target. On this evidence that is
a hypothesis worth testing properly, not a finding to build on yet.

![Prediction intervals over time: LightGBM's band stays nearly constant while the
news agent's widens sharply through the war window.](assets/fig7_sentinel_bands.png)

***Figure 4.** The width signal, rolled out over time. Each band is a method's
10–90 prediction interval for the 21-day return, origin by origin across the
protected window, with the realized return overlaid and the war-window origins
shaded. LightGBM's band — built from trailing volatility — barely moves all
half-year, its width varying only 1.7× min-to-max. The agent's varies 3.5×:
consistently wider (1.63× LightGBM's at the median origin) and far more
responsive, peaking at 3.0× LightGBM's width at the war trough. Its median
width inside the war window is not elevated relative to quiet weeks; what
distinguishes the break is the spike.*

![Agent-vs-LightGBM divergence per origin across the protected window, war
window shaded, with router-vs-baselines CRPS bars inset.](assets/fig3_divergence_sentinel.png)

***Figure 5.** Divergence between the news agent's and LightGBM's quantile
grids, origin by origin; three of the four largest spikes bracket the war
window, and the fourth (2026-06-08) is the agent pricing a post-record-high
correction that never confirmed. Inset: mean h=21 CRPS ×10⁻³ of always-LightGBM
(17.18), always-agent (17.59), and the divergence-gated router (16.88), on a
zoomed axis starting at 16.5 — the three means differ by only about 4%.
Exploratory: 24 origins, and the router threshold is set in-sample.*

One cost aside, said plainly: an agent forecast runs on the order of 100× the
tokens of an LLMP call — tens to hundreds of thousands against a couple thousand —
so whatever it buys on the scoreboard, it buys at a price worth naming.


## What the score can't see

CRPS ranks distributions. It cannot tell you whether a forecaster was *right for
the right reasons* — and for an analyst agent, the reasoning is the product. So we
ran a second track: at four landmark origins, the agent writes a scenario
analysis — weighted scenarios, named drivers, return ranges — and an LLM judge,
given only the realized returns, scores it 1–5 on three axes:

- **Drivers** — did the forces the write-up named actually move the index? Five
  means the cited drivers are exactly what happened; one means they are unrelated.
- **Calibration** — did the *stated probabilities* put the most weight on the
  scenario that matched the realized direction? This is about the weighting, not
  about whether any one range was hit precisely.
- **Specificity** — is it concrete and checkable, with dated figures and named
  catalysts, or generic hedging that would fit almost any week?

Then we read the artifacts ourselves, against the event timeline.

The 2026 war low, 2026-03-31, is the set piece. The agent nailed the shape: its
base case, "Commodity-Led Defensive Rotation" at 0.55 probability, called +3% to
+5% and the market delivered +3.65% at 21 days, +6.35% at 60. The judge, seeing
only those returns, awarded calibration 5/5 — direction and range dead-on. But
the *mechanism* was wrong. The base case bet on persistent Middle-East friction
holding oil and gold up; the rally that actually arrived came from the *ceasefire*
— oil plunged, and equities rose on the relief. Right level, wrong engine. And
the judge, grounded only in realized returns, correctly withheld drivers credit
(3/5): it cannot confirm a causal chain it cannot see. Catching the inverted
mechanism took a human reading the write-up against what happened. That is the
whole point of the exercise.

The pattern repeats. On 2025-04-01, tariff eve, the agent had the right driver in
its bear case — a tariff-driven earnings downgrade — but sized it at 0.15
probability and a −5% to −8% magnitude, against a drawdown that ran to −12.8%: a
clear under-reaction to a publicly telegraphed catalyst (calibration 2/5). Lay
the four verdicts side by side and the rubric's dimensions visibly measure
different things:

| Origin | Moment | Drivers | Calibration | Specificity |
|---|---|:-:|:-:|:-:|
| 2025-04-01 | tariff eve | 3 | 2 | 4 |
| 2025-04-08 | rebound eve | 2 | 4 | 3 |
| 2026-02-25 | pre-drawdown | 3 | 2 | 3 |
| 2026-03-31 | war low | 3 | 5 | 4 |

Calibration swings from 2 to 5; drivers never clears 3 — because the judge,
grounded only in realized returns, systematically refuses to certify causal
chains it cannot see. A forecaster can land the number and miss the reason, or
read the world well and misweight it, and only scoring them separately shows
which.

The claim to take away: for analyst agents, the artifact is the value and the
score is the floor. Automated judging scales and human trace-reading catches what
the judge structurally can't — they are complements, not substitutes.

![The war-low scenario card: three weighted scenarios beside the judge's
verdict and the realized returns.](assets/fig2_scenario_card.png)

***Figure 6.** The war-low scenario set (issued 2026-03-31), graded against
what happened: the 0.55 base case called the direction and roughly the
magnitude — realized +3.65% at 21 days, +6.35% at 60 — earning calibration 5/5
through a mechanism that did not occur (drivers 3/5). The base case bet on
persistent Middle-East friction keeping oil bid; the rally came from the
ceasefire relief instead. Right call, wrong reason.*

## The honest limit

There is a ceiling here that no methodology fixes. An agent that reads the open
web to forecast a series cannot be perfectly firewalled from the future it is
predicting. Our as-of date and verifier keep out the obvious leaks, but retrieval
is porous, model training cutoffs are opaque, and a stray dated-wrong snippet or a
fact the model simply knows can tint a "2026-03-30" forecast with April's
hindsight. Every offline agent score is therefore optimistic at best — the same
caution Part 1 raised for the LLMs, now sharper, because the agent is actively
reaching for information rather than passively holding it. The protected 2026
window narrows the gap; it does not close it. A strong offline result is a reason
to look harder at an agent, not yet a reason to trust its number — which is
exactly why Track 2 reads the reasoning rather than resting on the CRPS.

The honest destination is the one ForecastBench pointed at in Part 1's opening:
**live evaluation** — scoring forecasts whose answers do not exist yet. When the
outcome hasn't happened, leakage is not a guard you hope holds; it is
structurally impossible. That is the only setting in which an agent's news-reading
skill can be measured without an asterisk. We come back to that at the end.

## The adaptive agent — pre/post on the TSX

Every agent so far was frozen. The last rung asks what happens when the agent is
allowed to study. We gave the analyst one self-directed session — fifty turns,
about seventy minutes — with the full TSX history in a sandbox, a hypothesis
ledger, and an evidence gate: nothing enters its strategy file without recorded
confirmations. Then we evaluated the same agent on the protected window twice,
frozen both times, seed strategy versus studied strategy. The scoreboard did not
move: same origins, same referee, per-origin wins a coin flip at every horizon.

The session's real product is a finding about the gate itself. The transcript
shows genuine empirical work — dozens of sandboxed analyses over twenty-five
years of data, including an unprompted "Canadian holiday catch-up study" of the
exact calendar seam our own pipeline had once mishandled. But every hypothesis
that graduated collected its three required confirmations back-to-back, within
the same session, from the same study that proposed it: the letter of an
evidence gate, satisfied at a speed that hollows out its spirit. Anyone building
a self-improving agent will meet this failure mode.

A single session can only confirm hypotheses against the past that generated
them. Actual evidence requires hypotheses meeting forecasts whose outcomes don't
yet exist — an agent that keeps learning as its forecasts resolve, evaluated
against a frozen copy of itself. Evaluating adaptive agents that way — and
treating the harness itself as an optimizable system, in the lineage of ADAS,
the Darwin Gödel Machine, and [ALMA](https://arxiv.org/abs/2602.07755), which
meta-learns the agent's memory design rather than hand-engineering it — is where
our attention goes next.

## What's next

One problem, one series, one referee — from a naive floor to a frozen LLM to an
agent that reads the news, every method answered the same question the same way,
and each rung taught something the last couldn't.

It's worth saying plainly why the margins in this series are so thin: we chose
one of the hardest forecasting problems there is, on purpose. A major equity
index is the output of a market whose entire job is to price new information
before you can; when an agent reads a headline, it is racing the very mechanism
that generates its target. That the visible edges are small is the problem
talking as much as the paradigm. The same ladder pointed at a series no
efficient market prices — a demand curve, an operational load, a policy-linked
quantity — is where this machinery has real room, and that transfer is exactly
what the harness was built to make cheap. If you're starting tomorrow:

- **Begin with the naive floor and the classical methods.** They are nearly free,
  fully interpretable, and genuinely hard to beat — the honest bar everything else
  has to clear.
- **Add covariates and ML only where they earn their keep,** horizon by horizon;
  the panel that helps at one horizon can add noise at another.
- **Reach for agents to read the world, not to shave a decimal** — and evaluate
  them accordingly: judge the artifact, not just the score.

But the idea we find most interesting is the one this retrospective could only
*raise*, never settle. The agent's distinctive behaviour was not forecasting
the direction better but reacting to context — widening when it read something
unsettling, diverging from the conventional methods exactly when the world was
moving. If that holds up, a production pipeline for something like a market
index gets built as a *mix* rather than a contest: cheap, well-calibrated
models carrying the ordinary weeks, an agentic layer alongside them raising an
alert when the weeks stop being ordinary. We have not shown this works — we
have seen one regime event through a two-dozen-origin window that behaves like
five. What we have is something narrower and, we think, more useful: a
hypothesis precise enough to pre-register. The alert rule can be stated today,
committed to before the outcomes exist, and judged by live data. Whatever
credibility we have in proposing it comes from having tried to break it and
reported the break.

Why not just run a longer backtest and settle it now? Because for an agentic
forecaster, that door is closed — and seeing why may be the most durable thing
this study has to teach. To resolve a regime-conditional claim you need many
regime events, so you must reach further back into history. But the further
back you reach, the more likely the model has already read that history, and
leakage inflates exactly the methods under test. Protect against leakage with a
recent post-cutoff window and you are back to roughly five independent
observations. **You cannot buy statistical power with history when your
forecaster may have memorized the history.** Note who escapes the squeeze:
LightGBM can be backtested to 2005 without a qualm. The bind is specific to
forecasters that read, which is why agentic forecasting does not just prefer a
different evaluation protocol — it requires one. Live evaluation, scoring
forecasts whose outcomes do not yet exist, is the only setting where leakage is
structurally impossible and where the independent windows, run forward long
enough, stop being five. That is the destination ForecastBench pointed at in
Part 1's opening, and it is where this experiment goes next.

Everything behind this series — the harness, the data pipeline, the methods, the
evaluation — is open at
[github.com/VectorInstitute/agentic-forecasting](https://github.com/VectorInstitute/agentic-forecasting),
the repository we built for Vector's 2026 Agentic Forecasting Bootcamps. The
experiments in these two posts, and the live work that follows them, all began
as a [fork](https://github.com/VectorInstitute/agentic-forecasting-live) of it. We invite you to follow the process we used in the bootcamp: fork the repository, point it at your data, add your own extensions,
and see what the ladder tells you.
