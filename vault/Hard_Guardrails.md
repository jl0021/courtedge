---
date: 2026-11-15
tour: ATP+WTA
surface: All
legs: 0
combo_hit_prob: null
edge_pct: null
status: locked
---

# Hard Guardrails

These parameters are locked. No script, backtest, or playbook update may change them.
The numeric values below are mirrored as constants in `src/risk_gate.py`, and
`tests/test_risk_gate.py` fails if the two ever disagree.

## 1. Read-only MVP

- Never call Kalshi `POST` order or RFQ endpoints.
- Scan public markets with `GET` only.
- Log paper trades only. Bankroll is $10 with a $1 flat unit.

## 2. Combo hit-probability and edge gate

| Parameter | Value | Meaning |
|---|---|---|
| `MIN_COMBO_HIT_PROB` | 0.30 | Every combo must have `fair_combo_prob >= 0.30`. |
| `MIN_LEG_EDGE` | 0.05 | Each leg needs `simulated_prob - kalshi_ask >= 0.05`. |
| `MIN_COMBO_EDGE` | 0.06 | Joint combo probability must beat Kalshi's implied combo price by `>= 0.06`. |

There is no fixed ceiling on the number of legs. The 30% floor is what rules out moonshots.

## 3. Data sources and settlement

| Parameter | Value | Meaning |
|---|---|---|
| `MIN_NAME_MATCH_CONFIDENCE` | 0.98 | Minimum `player_lookup` match confidence. Below this, the leg is vetoed. |
| `VETO_LAST_RESULTS` | RET, W/O | Veto any player who lost their last match by retirement or walkover. The opponent who advanced is not vetoed. |

- Match history is seeded from `Aneeshers/tennis-sackmann-archive`, seasons 2024–2026.
  The upstream `JeffSackmann/tennis_atp` returns 404 and must not be used.
- Kalshi tennis settlement: once one ball is played, a mid-match retirement (`RET`)
  resolves the advancing player as `Yes` and the retiring player as `No`.
  Walkovers (`W/O`) have no ball played and are excluded from model scoring.

## 4. Obsidian formatting

Every generated note in `vault/` starts with YAML frontmatter containing
`date`, `tour`, `surface`, `legs`, `combo_hit_prob`, `edge_pct`, and `status`,
and links players as `[[First Last]]`.

## 5. 30-trade sample rule

| Parameter | Value | Meaning |
|---|---|---|
| `MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE` | 30 | A heuristic in [[Active_Playbook]] may change only if the pattern spans at least 30 settled legs and improves the Brier score. |

## 6. Milestone 1 verification gate

Both conditions must hold on the blind 2025–2026 walk-forward (tour-level main draws)
before live Kalshi polling is wired up.

| Parameter | Value | Meaning |
|---|---|---|
| `MAX_BRIER_SCORE` | 0.215 | Regression ceiling. The Brier score must be below this. |
| `MAX_CALIBRATION_GAP` | 0.035 | Calibration. In every probability bucket, the predicted and actual win rates must be within 3.5 percentage points. |
| `CALIBRATION_BUCKETS` | 5 | Buckets of the favourite's predicted win probability: 0.5–0.6, 0.6–0.7, 0.7–0.8, 0.8–0.9, 0.9–1.0. |

The original target of 0.185 was replaced on 2026-11-15. It was below the 0.196 that
Pinnacle's odds score on comparable matches, so no pre-match model could meet it.
