# Forecasting agents that read the news — and what it takes to trust them

**By the Vector AI Engineering team**

*Part 2 of two.*

## The staircase

Part 1 ended on a frozen LLM that could emit a genuine predictive distribution
but saw only numbers — quiet exactly where the series was about to get loud,
because the cause of every regime break was written in language, not in the price
history. The next rung hands the same forecaster the news.

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

![Anatomy of one agent forecast: six search queries, four load-bearing rationale
factors, and the emitted quantile grid with the realized move marked at
Q0.80.](assets/fig1_agent_anatomy.png)

*One forecast, end to end: the news analyst's six date-scoped searches (left),
the load-bearing factors from its written rationale (center), and the quantile
grid it emitted (right) — median +1%, deliberate negative skew, and the realized
21-day move of +5.1% landing essentially at its Q0.80.*

## The same scoreboard

An agent that reads is still just another predictor, so it earns its place the
same way everything in Part 1 did: identical origins, identical cutoff, identical
referee. We slot the news agent into the exact rolling-origin sweeps — the 2025
backtest and the protected 2026 eval — and score its quantiles with CRPS beside
the naive floor, the classical methods, LightGBM, and the frozen LLMP models. No
special pleading, and no thumb on the scale for the method that took the most
work to build.

The honest expectation, set by Part 1, is that reading the news does not
automatically win. On the protected 2026 eval at h=1 the numbers-only board still
clusters near CRPS 0.0050 — LightGBM-with-covariates at 0.00497, a lightweight
LLMP call at 0.00501 — while the naive floor sits at 0.0093. For a one-day-ahead
return, where the move is close to unforecastable, a news scan has little to add
over a well-shaped distribution, and the agents' rows say exactly that: 0.00507
and 0.00509 for the two news-agent variants, mid-pack and indistinguishable from
the frozen models. But step through the horizons and the board reorganizes by
*family*. At h=5 the language models sweep: the top five methods are all
LLM-based, and the two LightGBM configurations fall to 14th and 17th of
twenty-one — the week horizon appears to sit in a pocket where an LLM's fatter,
more skeptical distributions fit the return process better than the trees'
tight quantiles.
At h=21 the guard changes again: the trees retake the lead, but three of the
top seven methods are agents, led by the news agent (on the lighter model) at
third, 0.01759 — behind only the two LightGBM configurations and ahead of every
frozen LLMP on the board.

![The full protected-eval leaderboard by horizon, all 21 methods, with the
agent rungs highlighted against the numbers-only ladder and the LLMP
matrix.](assets/fig5_combined_leaderboard.png)

*The complete protected-window scoreboard, every method family on one board.
The guard changes at every horizon: trees at h=1, the LLM family sweeping h=5,
trees back on top at h=21 with three agents in the top seven.*

One more honest wrinkle: that ranking is model-dependent — the same agent
harness on a heavier model finishes mid-pack at h=21 — so "agents read the
news" is not yet a horizontal claim about agents; it is a configuration that
has to earn its rung, model by model. The code-executing analyst completes the
picture with an inverted profile: at
h=1 it lands second of twenty-one at 0.00499 — a hair behind
LightGBM-with-covariates — because its eleven-code-runs-per-forecast style *is*
statistics, and one-day forecasting is purely a calibration game. At h=21 it
sits mid-pack, yet beats its own frozen base model at eighteen of twenty-four
origins — the clearest paired evidence in the study that agency improves a
model — while showing none of the news agent's break-window advantage: it
computes from the same history the trees see, and it fails where they fail.
What an agent *reads* determines where it wins.

Ranks, though, still average over the thing that matters. Split the h=21
origins into the ten inside the war window and the fourteen quiet ones, and the
news agent's third place decomposes into two opposite results: at the break it
is ~11% *better* than LightGBM-with-covariates — largely by sidestepping the
tree's worst blowups — and on quiet weeks it is ~22% *worse*, paying an
LLM-noise tax whenever there is nothing to read. The trees win the average
because averages are mostly quiet weeks; the agent earns its fee precisely at
the moments Part 1 showed the numbers-only ladder to be blind. And the paired,
same-model comparisons say agency itself — not just model scale — is doing
work: the code agent beats its own frozen base at eighteen of twenty-four
origins, the strongest paired result in the study, while the news agent's gain
over its base is positive but within noise. With one regime event per year, no
retrospective can grade these conditional claims properly — an honest limit we
return to below.

![Where agents earn their keep: war-window vs quiet-week CRPS against
LightGBM, and paired same-model agency deltas at
h=21.](assets/fig6_where_agents_earn.png)

*Left: the news agent versus LightGBM-with-covariates at h=21, split into
war-window and quiet origins — the average hides two opposite results. Right:
paired same-model deltas, frozen LLMP → agent, at h=21; the code agent's
18-of-24 is the study's only statistically significant paired win.*

There is one more signal hiding in the comparison, and it may matter more than
the ranks: *when the agent disagrees with the trees*. Measure the gap between
the agent's quantile grid and LightGBM's at each origin, and on the protected
window that divergence behaves like an event detector — it correlates with the
size of the realized move (ρ≈0.5) and with the tree's own error, and three of
its four largest values sit at the origins bracketing the 2026 war drawdown.
(The fourth is an early-June origin where the agent sharply widened its tails
after a correction off a record high — the sentinel fires on perceived regime
risk, not only on breaks that confirm.) A naive rule that trusts the agent only
when divergence runs above its median beats either method alone: CRPS 0.0169
against 0.0172 for always-LightGBM and 0.0176 for always-agent. We report this
as exploratory, not established: twenty-four origins, an in-sample threshold,
and the same construction on the 2025 backtest fires at the tariff window but
does not pay — that agent's divergent forecasts were the wrong ones. A
hypothesis to test prospectively, not a result.

Decompose the divergence and the mechanism gets sharper — and more
interesting. The gap between the two forecasters has two parts: the agent
moving its *center* away from the tree's, and the agent changing its *width*.
It is the width that carries the signal. The ratio of the agent's 10–90
interval to the tree's predicts the size of the realized move (ρ=0.52,
p=0.01); the median shift predicts nothing (ρ=0.20). At the war trough the
agent's interval ran **three times** the tree's — and at one break origin its
median matched the tree's almost exactly while its interval ran 2.5× wider: a
pure alarm, no directional bet. The tree's interval, built from trailing
volatility, barely moved all half-year. The agent doesn't beat the
numbers-only methods by predicting direction better; it notices, from the
news, when the quiet period is ending — and says so by widening. That
reframes what an agent is for: not a replacement for the cheap methods, a
*sentinel* alongside them — conventional forecasters for the nominal periods,
and an agent watching for the moment the period stops being nominal.

![Agent-vs-tree divergence per origin across the protected window, war window
shaded, with router-vs-baselines CRPS bars inset.](assets/fig3_divergence_sentinel.png)

*Divergence between the news agent's and LightGBM's quantile grids, origin by
origin. The spikes are the war window; the inset compares always-tree,
always-agent, and the divergence-gated router.*

One cost aside, said plainly: an agent forecast runs on the order of 100× the
tokens of an LLMP call — tens to hundreds of thousands against a couple thousand —
so whatever it buys on the scoreboard, it buys at a price worth naming.


## What the score can't see

CRPS ranks distributions. It cannot tell you whether a forecaster was *right for
the right reasons* — and for an analyst agent, the reasoning is the product. So we
ran a second track: at four landmark origins, the agent writes a scenario
analysis — weighted scenarios, named drivers, return ranges — and an LLM judge,
given only the realized returns, scores it on calibration, drivers, and
specificity. Then we read the artifacts ourselves, against the event timeline.

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

![The 2026-03-31 scenario card: three weighted scenarios beside the judge's
verdict and the realized returns, with the mechanism mismatch
annotated.](assets/fig2_scenario_card.png)

*The war-low scenario set, graded against what happened: the 0.55 base case
called the direction and roughly the magnitude (calibration 5/5) through a
mechanism that did not occur (drivers 3/5) — right call, wrong reason.*

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

Every agent so far was frozen: the same prompt, the same tools, forecast after
forecast, learning nothing from its own record. The last rung asks what happens
when the agent is allowed to study. We gave the analyst one self-directed study
session — fifty turns, about seventy minutes — with the full TSX history in a
sandbox, a hypothesis ledger, and one rule: nothing enters its strategy file
without passing an evidence gate (open a hypothesis, record confirmations,
graduate it). Then we evaluated the *same* agent twice on the protected 2026
window, frozen both times: once with its untouched seed strategy, once with the
studied one.

Read the artifacts before the scoreboard and the session is genuinely
impressive. The transcript shows real empirical work, not narration: dozens of
sandboxed analyses over twenty-five years of daily data, iterating through
errors — including, we note with some delight, an unprompted "Canadian holiday
catch-up study" of days when Toronto trades while US markets close, the exact
calendar seam our own covariate pipeline had mishandled until the same week.
The distilled strategy holds twenty-five conditional calibration rules
(widen intervals this much in a low-volatility regime, that much after a 2.5σ
shock) and — the best scientific hygiene in the file — several recorded
negative results: day-of-week effects tested and explicitly ruled out. But the
trace also shows the gate's weakness: every graduated hypothesis collected its
three required confirmations back-to-back, within the same session, from the
same study that proposed it. The letter of the evidence gate, satisfied at a
speed that hollows out its spirit.

The scoreboard then says what it should say: nothing moved. Pre versus post,
same origins, same referee — 0.00522 versus 0.00527 at h=1, 0.01206 versus
0.01193 at h=5, 0.01857 versus 0.01876 at h=21, per-origin wins a coin flip at
every horizon. (Both arms, it's worth saying, are respectable forecasters — at h=5 the
studied agent is the best agent on the entire board, fourth overall, and the
seed sits ninth. Note what that means: a *hand-written* strategy skill —
martingale medians, volatility-scaled intervals, check-the-calendar discipline
— already lifts an agent past every free-form analyst we ran. Most of the
recipe is the discipline, not the study; which is itself a finding about where
harness design ends and learning begins.) The one suggestive trace: inside the war window the studied agent is
directionally better at h=5 and h=21 — precisely where widen-under-stress
corrections should bite — at magnitudes twenty-four origins cannot separate
from luck.

We read this null as the study's most useful output. A single session can only
confirm hypotheses against the past that generated them; it produces real
knowledge and cannot, by construction, produce *evidence* — that requires
hypotheses meeting forecasts whose outcomes don't yet exist. Which is the
design this whole series has been walking toward: an agent that keeps learning
as its forecasts resolve, evaluated against a frozen copy of itself, the gap
between them the measured value of experience. Evaluating adaptive agents that
way — and treating the agent harness itself as an optimizable system, in the
lineage of ADAS and the Darwin Gödel Machine — is where our attention goes
next. Stay tuned.

![Pre/post adaptive evaluation: paired CRPS by horizon with war-window split,
beside an excerpt of the learned strategy file.](assets/fig4_adaptive_prepost.png)

*The pre/post scoreboard beside what was learned: paired mean CRPS at each
horizon (with the war-window cut), and an excerpt of the strategy file the
study produced — including a graduated correction and a recorded negative
result.*

## What's next

One problem, one series, one referee — from a naive floor to a frozen LLM to an
agent that reads the news, every method answered the same question the same way,
and each rung taught something the last couldn't.

It's worth saying plainly why the margins in this series are so thin: we chose
one of the hardest forecasting problems there is, on purpose. A major equity
index is the output of a market whose entire job is to price new information
before you can — millions of participants compressing the news into the close,
every day. When an agent reads a headline, it is racing the very mechanism that
generates its target. That an agent finds *any* conditional edge here is
notable; that the edge is small is the problem talking, not the paradigm. The
same ladder, the same referee, and the same agents pointed at a series no
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

But the findings we care about most are the ones this retrospective could only
*raise*, not settle. The agent earned its keep at the regime break and signalled
it by widening its intervals — yet with roughly one such break a year, two dozen
origins can suggest that edge without ever confirming it, and no offline protocol
can fully firewall a model that reads the open web from the future it is scored
against. The honest way to answer the questions we've raised is to stop grading
forecasts against a past the models may already know, and start scoring them
against a future that does not yet exist — where leakage is not a guard you hope
holds but a physical impossibility, and where "divergence is an alarm" and
"agents pay at the breaks" become predictions you commit to *before* the outcome
rather than patterns you notice after. That is the destination ForecastBench
pointed at in Part 1's opening, and it is where we are taking the S&P/TSX
experiment next.

The full harness, data pipeline, and methods behind this series are open at
[github.com/VectorInstitute/agentic-forecasting](https://github.com/VectorInstitute/agentic-forecasting).
