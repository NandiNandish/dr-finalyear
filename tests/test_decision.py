"""
Unit tests for the Clinical Decision Support Logic and safety invariants.
"""

import pytest
from src.decision import DecisionSupportEngine, ScreeningDecision
from src.quality.gate import QualityResult


@pytest.fixture
def engine():
    return DecisionSupportEngine(confidence_threshold=0.60)


def test_ungradable_rejection_safety_invariant(engine):
    """
    CRITICAL: Ungradable image must NEVER output 'No DR' or reassuring recommendation.
    Must always trigger RECAPTURE.
    """
    q_failed = QualityResult(
        is_gradable=False,
        rejection_code="UNGRA_BLURRED",
        reason="Severe motion blur obscures microaneurysms.",
        recommendation="RECAPTURE",
    )
    decision = engine.evaluate(quality_result=q_failed)

    assert decision.is_gradable is False
    assert decision.predicted_grade is None
    assert decision.recommendation == "RECAPTURE"
    assert decision.urgency_tier == "MANDATORY_RECAPTURE"
    assert decision.requires_human_review is True
    assert "RECAPTURE" in decision.action_plan.upper()
    assert "SCREENING RECOMMENDATION AID" in decision.disclaimer


def test_grade_0_routine_followup(engine):
    q_pass = QualityResult(is_gradable=True)
    decision = engine.evaluate(quality_result=q_pass, predicted_grade=0, confidence=0.92)

    assert decision.is_gradable is True
    assert decision.predicted_grade == 0
    assert decision.is_referable is False
    assert decision.recommendation == "ROUTINE_FOLLOWUP"
    assert decision.urgency_tier == "ROUTINE_12_MONTHS"
    assert decision.requires_human_review is False


def test_grade_1_mild_followup(engine):
    q_pass = QualityResult(is_gradable=True)
    decision = engine.evaluate(quality_result=q_pass, predicted_grade=1, confidence=0.88)

    assert decision.predicted_grade == 1
    assert decision.is_referable is False
    assert decision.recommendation == "ROUTINE_FOLLOWUP"
    assert decision.urgency_tier == "ROUTINE_6_TO_12_MONTHS"


def test_grade_2_moderate_prompt_referral(engine):
    q_pass = QualityResult(is_gradable=True)
    decision = engine.evaluate(quality_result=q_pass, predicted_grade=2, confidence=0.85)

    assert decision.predicted_grade == 2
    assert decision.is_referable is True
    assert decision.recommendation == "PROMPT_REFERRAL"
    assert decision.urgency_tier == "PROMPT_REFERRAL_4_TO_6_WEEKS"


def test_grade_3_and_4_urgent_referral(engine):
    q_pass = QualityResult(is_gradable=True)

    # Severe NPDR
    d3 = engine.evaluate(quality_result=q_pass, predicted_grade=3, confidence=0.91)
    assert d3.is_referable is True
    assert d3.recommendation == "URGENT_REFERRAL"
    assert d3.urgency_tier == "URGENT_EVALUATION_1_TO_2_WEEKS"

    # PDR
    d4 = engine.evaluate(quality_result=q_pass, predicted_grade=4, confidence=0.94)
    assert d4.is_referable is True
    assert d4.recommendation == "URGENT_REFERRAL"
    assert d4.urgency_tier == "IMMEDIATE_EVALUATION_1_WEEK"


def test_low_confidence_flagged_for_human_review(engine):
    q_pass = QualityResult(is_gradable=True)
    # Low confidence (0.45 < 0.60)
    decision = engine.evaluate(quality_result=q_pass, predicted_grade=0, confidence=0.45)

    assert decision.requires_human_review is True
    assert decision.recommendation == "FLAGGED_HUMAN_REVIEW"
    assert any("confidence" in r.lower() for r in decision.human_review_reasons)


def test_gradcam_outside_retina_flagged(engine):
    q_pass = QualityResult(is_gradable=True)
    # Heatmap peak falls outside retina
    decision = engine.evaluate(
        quality_result=q_pass,
        predicted_grade=0,
        confidence=0.89,
        gradcam_outside_retina=True,
    )

    assert decision.requires_human_review is True
    assert decision.recommendation == "FLAGGED_HUMAN_REVIEW"
    assert any("retinal boundary" in r.lower() for r in decision.human_review_reasons)
