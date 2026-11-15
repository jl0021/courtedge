import inspect
import re
from pathlib import Path

import pytest

from src import ingest, risk_gate
from src.risk_gate import Leg, evaluate_combo

VAULT = Path(__file__).resolve().parent.parent / "vault"


def leg(player="A Player", match="m1", prob=0.70, ask=0.60, **kwargs):
    return Leg(player=player, match_key=match, simulated_prob=prob, kalshi_ask=ask, **kwargs)


def test_constants_match_the_vault_note():
    text = (VAULT / "Hard_Guardrails.md").read_text()
    for name in ("MIN_COMBO_HIT_PROB", "MIN_LEG_EDGE", "MIN_COMBO_EDGE", "MIN_NAME_MATCH_CONFIDENCE",
                 "MIN_SETTLED_LEGS_FOR_PLAYBOOK_CHANGE", "MAX_BRIER_SCORE", "MAX_CALIBRATION_GAP",
                 "CALIBRATION_BUCKETS"):
        found = re.search(rf"`{name}` \| ([0-9.]+) \|", text)
        assert found, f"{name} missing from Hard_Guardrails.md"
        assert float(found.group(1)) == getattr(risk_gate, name)
    assert risk_gate.VETO_LAST_RESULTS == ("RET", "W/O")
    assert ingest.MIN_NAME_MATCH_CONFIDENCE == risk_gate.MIN_NAME_MATCH_CONFIDENCE


def test_locked_values():
    assert risk_gate.MIN_COMBO_HIT_PROB == 0.30
    assert risk_gate.MIN_LEG_EDGE == 0.05
    assert risk_gate.MIN_COMBO_EDGE == 0.06
    assert risk_gate.MAX_BRIER_SCORE == 0.215
    assert risk_gate.MAX_CALIBRATION_GAP == 0.035


def test_good_combo_is_approved():
    legs = [leg("A One", "m1", 0.72, 0.62), leg("B Two", "m2", 0.68, 0.60)]
    decision = evaluate_combo(legs, kalshi_combo_price=0.40)
    assert decision.approved and decision.reasons == []
    assert decision.fair_combo_prob == pytest.approx(0.72 * 0.68)


def test_leg_edge_boundary():
    assert evaluate_combo([leg(prob=0.55, ask=0.50)], 0.45).approved          # exactly +5%
    assert not evaluate_combo([leg(prob=0.549, ask=0.50)], 0.45).approved


def test_combo_below_thirty_percent_is_rejected_regardless_of_edge():
    legs = [leg(f"P{i}", f"m{i}", 0.60, 0.40) for i in range(3)]      # 0.216 joint
    decision = evaluate_combo(legs, kalshi_combo_price=0.05)
    assert not decision.approved
    assert any("below 0.3" in r for r in decision.reasons)


def test_no_leg_count_ceiling():
    legs = [leg(f"P{i}", f"m{i}", 0.90, 0.80) for i in range(8)]      # 0.43 joint
    assert evaluate_combo(legs, kalshi_combo_price=0.30).approved


def test_combo_edge_must_be_six_percent():
    legs = [leg("A One", "m1", 0.72, 0.62), leg("B Two", "m2", 0.68, 0.60)]
    fair = 0.72 * 0.68
    assert evaluate_combo(legs, fair - 0.06).approved
    assert not evaluate_combo(legs, fair - 0.059).approved


@pytest.mark.parametrize("last_result", ["RET", "W/O"])
def test_veto_applies_to_the_player_who_retired_or_withdrew(last_result):
    decision = evaluate_combo([leg(last_result=last_result, last_result_won=False)], 0.50)
    assert not decision.approved
    assert any(last_result in r for r in decision.reasons)


@pytest.mark.parametrize("last_result", ["RET", "W/O"])
def test_advancing_winner_is_not_vetoed(last_result):
    assert evaluate_combo([leg(last_result=last_result, last_result_won=True)], 0.50).approved


def test_veto_applies_when_the_side_is_unknown():
    assert not evaluate_combo([leg(last_result="RET")], 0.50).approved
    assert risk_gate.is_vetoed_by_last_result("W/O", None)


def test_ordinary_loss_is_not_vetoed():
    assert evaluate_combo([leg(last_result="COMPLETED", last_result_won=False)], 0.50).approved


def test_name_confidence_veto():
    assert not evaluate_combo([leg(name_match_confidence=0.979)], 0.50).approved
    assert evaluate_combo([leg(name_match_confidence=0.98)], 0.50).approved


def test_same_match_legs_rejected():
    legs = [leg("A One", "m1", 0.72, 0.62), leg("B Two", "m1", 0.68, 0.60)]
    assert not evaluate_combo(legs, 0.30).approved


def test_empty_and_invalid_inputs():
    assert not evaluate_combo([], 0.5).approved
    assert not evaluate_combo([leg(prob=1.0, ask=0.5)], 0.5).approved
    assert not evaluate_combo([leg()], 1.2).approved


def test_playbook_rule_and_brier_gate():
    assert not risk_gate.playbook_change_allowed(29, 0.22, 0.20)
    assert not risk_gate.playbook_change_allowed(30, 0.20, 0.20)
    assert risk_gate.playbook_change_allowed(30, 0.22, 0.20)
    assert risk_gate.brier_gate_passed(0.2149) and not risk_gate.brier_gate_passed(0.215)
    assert risk_gate.calibration_gate_passed(0.035) and not risk_gate.calibration_gate_passed(0.0351)
    assert risk_gate.milestone1_gate_passed(0.212, 0.030)
    assert not risk_gate.milestone1_gate_passed(0.212, 0.040)      # calibrated badly
    assert not risk_gate.milestone1_gate_passed(0.216, 0.010)      # Brier regression


def test_no_order_placing_code_in_src():
    """Read-only MVP: nothing under src/ may issue an HTTP write or touch order endpoints."""
    banned = re.compile(r"requests\.(post|put|delete|patch)|\.post\(|method\s*=\s*['\"]POST|/portfolio/orders|/rfq", re.I)
    for path in (Path(inspect.getfile(risk_gate)).parent).glob("*.py"):
        assert not banned.search(path.read_text()), f"write call found in {path.name}"
