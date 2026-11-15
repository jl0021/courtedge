---
date: 2026-11-15
tour: ATP+WTA
surface: All
legs: 0
combo_hit_prob: null
edge_pct: null
status: active
---

# Active Playbook

Heuristics that sit on top of the simulator. They are refined weekly, and only under
the 30-trade sample rule in [[Hard_Guardrails]]: a change needs a pattern that spans
at least 30 settled legs and improves the Brier score.

## Sample status

| Metric | Value |
|---|---|
| Settled legs | 0 |
| Settled combos | 0 |
| Eligible for heuristic changes | No (needs 30 settled legs) |

## Active heuristics

None yet. No legs have settled, so there is no evidence to base a heuristic on.
Until the sample reaches 30 settled legs, combos are selected by the locked gate alone.

## Candidate patterns under observation

Patterns go here while their sample is still below 30 settled legs. They do not affect selection.

| Pattern | Settled legs | Brier with | Brier without | Status |
|---|---|---|---|---|
| (none) | 0 | n/a | n/a | n/a |

## Change log

| Date | Change | Settled legs in sample | Brier before | Brier after |
|---|---|---|---|---|
| 2026-11-15 | Playbook created. No heuristics active. | 0 | n/a | n/a |
