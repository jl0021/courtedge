"""Risk gate: the locked rules every paper combo must pass.

The constants here mirror vault/Hard_Guardrails.md. They are not configurable at
runtime, and tests/test_risk_gate.py fails if this file and the vault note disagree.

This module only evaluates candidate combos. It never talks to Kalshi and has no
code path that can place an order.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

# ---- Locked guardrails (see vault/Hard_Guardrails.md) ----
MIN_COMBO_HIT_PROB = 0.30
MIN_LEG_EDGE = 0.05
MIN_COMBO_EDGE = 0.06
MIN_NAME_MATCH_CONFIDENCE = 0.98
VETO_LAST_RESULTS = ("RET", "W/O")
MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE = 30
# Milestone 1 gate: both conditions must hold on the blind 2025-2026 walk-forward.
MAX_BRIER_SCORE = 0.215          # regression ceiling
MAX_CALIBRATION_GAP = 0.035      # largest allowed |predicted - actual| win rate in any bucket
CALIBRATION_BUCKETS = 5          # favourite probability: 0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0

_EPS = 1e-9  # tolerance so that 0.55 - 0.50 counts as an edge of exactly 0.05


@dataclass(frozen=True)
class Leg:
    player: str                      # canonical "First Last"
    match_key: str                   # identifies the match, so two legs on one match are caught
    simulated_prob: float            # model probability that this leg resolves Yes
    kalshi_ask: float                # Kalshi Yes ask, in dollars (0-1)
    name_match_confidence: float = 1.0
    last_result: str | None = None   # how the player's previous match ended: COMPLETED, RET, W/O...
    last_result_won: bool | None = None  # True if the player won that match (advanced), False if they lost

    @property
    def edge(self) -> float:
        return self.simulated_prob - self.kalshi_ask


@dataclass
class GateDecision:
    approved: bool
    fair_combo_prob: float
    combo_edge: float
    reasons: list[str] = field(default_factory=list)


def check_leg(leg: Leg) -> list[str]:
    """Return the reasons a single leg fails. An empty list means it passes."""
    reasons = []
    if not (0.0 < leg.simulated_prob < 1.0) or not (0.0 < leg.kalshi_ask < 1.0):
        reasons.append(f"{leg.player}: probability and ask must be strictly between 0 and 1")
        return reasons
    if leg.name_match_confidence < MIN_NAME_MATCH_CONFIDENCE:
        reasons.append(f"{leg.player}: name match confidence {leg.name_match_confidence:.3f} "
                       f"is below {MIN_NAME_MATCH_CONFIDENCE}")
    if is_vetoed_by_last_result(leg.last_result, leg.last_result_won):
        reasons.append(f"{leg.player}: lost last match by {leg.last_result}")
    if leg.edge < MIN_LEG_EDGE - _EPS:
        reasons.append(f"{leg.player}: leg edge {leg.edge:+.3f} is below +{MIN_LEG_EDGE}")
    return reasons


def is_vetoed_by_last_result(last_result: str | None, last_result_won: bool | None) -> bool:
    """Veto the player who retired or gave the walkover, not the opponent who advanced.

    If it is unknown which side the player was on, the veto applies.
    """
    return last_result in VETO_LAST_RESULTS and last_result_won is not True


def fair_combo_probability(legs: list[Leg]) -> float:
    """Joint probability of all legs, assuming legs on different matches are independent."""
    return math.prod(leg.simulated_prob for leg in legs)


def evaluate_combo(legs: list[Leg], kalshi_combo_price: float) -> GateDecision:
    """Apply every locked rule to a candidate combo.

    kalshi_combo_price: Kalshi's implied price for the whole combo, in dollars (0-1).
    There is no cap on the number of legs. The 30% hit-probability floor does that job.
    """
    reasons: list[str] = []
    if not legs:
        return GateDecision(False, 0.0, 0.0, ["combo has no legs"])
    if not (0.0 < kalshi_combo_price < 1.0):
        reasons.append("combo price must be strictly between 0 and 1")

    for leg in legs:
        reasons.extend(check_leg(leg))

    match_keys = [leg.match_key for leg in legs]
    if len(set(match_keys)) != len(match_keys):
        reasons.append("two legs are on the same match, so they are not independent")

    fair = fair_combo_probability(legs) if all(0.0 < leg.simulated_prob < 1.0 for leg in legs) else 0.0
    combo_edge = fair - kalshi_combo_price
    if fair < MIN_COMBO_HIT_PROB - _EPS:
        reasons.append(f"fair combo probability {fair:.3f} is below {MIN_COMBO_HIT_PROB}")
    if combo_edge < MIN_COMBO_EDGE - _EPS:
        reasons.append(f"combo edge {combo_edge:+.3f} is below +{MIN_COMBO_EDGE}")

    return GateDecision(approved=not reasons, fair_combo_prob=fair, combo_edge=combo_edge, reasons=reasons)


def playbook_change_allowed(settled_legs: int, brier_before: float, brier_after: float) -> bool:
    """30-trade sample rule for vault/Active_Playbook.md."""
    return settled_legs >= MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE and brier_after < brier_before


def brier_gate_passed(brier: float) -> bool:
    """Regression ceiling half of the Milestone 1 gate."""
    return brier < MAX_BRIER_SCORE


def calibration_gate_passed(max_calibration_gap: float) -> bool:
    """Calibration half of the Milestone 1 gate."""
    return max_calibration_gap <= MAX_CALIBRATION_GAP


def milestone1_gate_passed(brier: float, max_calibration_gap: float) -> bool:
    """Live Kalshi polling stays off until this is True."""
    return brier_gate_passed(brier) and calibration_gate_passed(max_calibration_gap)
