"""
Clinical Decision Support Engine for Diabetic Retinopathy Screening.

Maps:
- Gradability (Quality Gate)
- 5-level DR Grade (0 to 4)
- Model Confidence
- Grad-CAM Anatomical Plausibility Flag

To standardized clinical screening recommendations:
- RECAPTURE
- ROUTINE_FOLLOWUP
- PROMPT_REFERRAL
- URGENT_REFERRAL
- FLAGGED_HUMAN_REVIEW

Safety Invariant:
An ungradable image must NEVER produce a reassuring "No DR" result.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from src.data.label_maps import DRGrade, LABEL_TO_NAME, is_referable_dr
from src.quality.gate import QualityResult


MANDATORY_CLINICAL_DISCLAIMER = (
    "SCREENING RECOMMENDATION AID ONLY, NOT A DIAGNOSTIC DEVICE. "
    "This automated screening system is intended to assist healthcare professionals in "
    "tele-ophthalmology screening, triage, and prioritization. It does not replace clinical "
    "examination by an ophthalmologist or optometrist. Definitive diagnostic and treatment "
    "decisions must always be made by a qualified clinician."
)


@dataclass
class ScreeningDecision:
    is_gradable: bool
    predicted_grade: Optional[int]
    grade_name: str
    confidence: float
    probabilities: Dict[str, float]
    is_referable: bool
    recommendation: str
    urgency_tier: str
    action_plan: str
    requires_human_review: bool
    human_review_reasons: list
    disclaimer: str = MANDATORY_CLINICAL_DISCLAIMER

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_gradable": self.is_gradable,
            "predicted_grade": self.predicted_grade,
            "grade_name": self.grade_name,
            "confidence": round(self.confidence, 4),
            "probabilities": self.probabilities,
            "is_referable": self.is_referable,
            "recommendation": self.recommendation,
            "urgency_tier": self.urgency_tier,
            "action_plan": self.action_plan,
            "requires_human_review": self.requires_human_review,
            "human_review_reasons": self.human_review_reasons,
            "disclaimer": self.disclaimer,
        }


class DecisionSupportEngine:
    """
    Rule-based clinical decision support engine.
    """

    def __init__(self, confidence_threshold: float = 0.60):
        self.confidence_threshold = float(confidence_threshold)

    def evaluate(
        self,
        quality_result: QualityResult,
        predicted_grade: Optional[int] = None,
        confidence: float = 0.0,
        probabilities: Optional[Dict[str, float]] = None,
        gradcam_outside_retina: bool = False,
    ) -> ScreeningDecision:
        """
        Synthesizes quality gate result, DR grade, confidence, and explainability check.
        """
        # RULE 1: Quality Gate Failure Invariant
        # An ungradable image must NEVER produce a reassuring "No DR" result!
        if not quality_result.is_gradable:
            return ScreeningDecision(
                is_gradable=False,
                predicted_grade=None,
                grade_name="Ungradable (Quality Gate Rejected)",
                confidence=0.0,
                probabilities={},
                is_referable=True,  # Treat as clinically referable / action required
                recommendation="RECAPTURE",
                urgency_tier="MANDATORY_RECAPTURE",
                action_plan=(
                    f"Recapture colour fundus photograph. Reason: {quality_result.reason}. "
                    "Ensure adequate pupillary dilation, stable fixation, and correct camera focus."
                ),
                requires_human_review=True,
                human_review_reasons=[f"Image ungradable: {quality_result.rejection_code}"],
            )

        # Grade is gradable
        if predicted_grade is None:
            raise ValueError("predicted_grade must be provided for gradable images.")

        grade = int(predicted_grade)
        grade_name = LABEL_TO_NAME.get(grade, f"Grade {grade}")
        referable = is_referable_dr(grade)
        probs = probabilities or {}

        review_reasons = []
        # Check low confidence
        if confidence < self.confidence_threshold:
            review_reasons.append(
                f"Model prediction confidence ({confidence*100:.1f}%) is below clinical safety threshold ({self.confidence_threshold*100:.1f}%)."
            )

        # Check explainability anatomical plausibility
        if gradcam_outside_retina:
            review_reasons.append(
                "Grad-CAM salience peak falls outside the retinal boundary (potential scanner artefact or peripheral distraction)."
            )

        requires_human_review = len(review_reasons) > 0

        # Grade-to-Recommendation Mapping
        if grade == DRGrade.NO_DR:
            recommendation = "ROUTINE_FOLLOWUP"
            urgency_tier = "ROUTINE_12_MONTHS"
            action_plan = (
                "No diabetic retinopathy lesions detected. Schedule routine annual tele-retinal screening in 12 months. "
                "Emphasize strict glycemic (HbA1c < 7%) and blood pressure control with primary care physician."
            )

        elif grade == DRGrade.MILD_NPDR:
            recommendation = "ROUTINE_FOLLOWUP"
            urgency_tier = "ROUTINE_6_TO_12_MONTHS"
            action_plan = (
                "Mild Non-Proliferative Diabetic Retinopathy (microaneurysms only). "
                "Schedule repeat screening fundus photography in 6 to 12 months. Notify patient's endocrinologist."
            )

        elif grade == DRGrade.MODERATE_NPDR:
            recommendation = "PROMPT_REFERRAL"
            urgency_tier = "PROMPT_REFERRAL_4_TO_6_WEEKS"
            action_plan = (
                "Referable Diabetic Retinopathy detected (Moderate NPDR: exudates / blot hemorrhages). "
                "Prompt referral to ophthalmology clinic within 4 to 6 weeks for complete dilated slit-lamp fundus examination and OCT."
            )

        elif grade == DRGrade.SEVERE_NPDR:
            recommendation = "URGENT_REFERRAL"
            urgency_tier = "URGENT_EVALUATION_1_TO_2_WEEKS"
            action_plan = (
                "Severe Non-Proliferative Diabetic Retinopathy detected (high progression risk). "
                "Urgent ophthalmologist evaluation required within 1 to 2 weeks for fluorescein angiography and staging."
            )

        elif grade == DRGrade.PROLIFERATIVE_DR:
            recommendation = "URGENT_REFERRAL"
            urgency_tier = "IMMEDIATE_EVALUATION_1_WEEK"
            action_plan = (
                "Proliferative Diabetic Retinopathy (PDR) detected (neovascularization / hemorrhage hazard). "
                "Immediate retinal specialist evaluation within 1 week for consideration of panretinal photocoagulation (PRP) or anti-VEGF therapy."
            )

        else:
            recommendation = "FLAGGED_HUMAN_REVIEW"
            urgency_tier = "UNKNOWN_TIER"
            action_plan = "Unrecognized severity grade. Manual ophthalmology review mandatory."

        # If flagged for human review, adjust primary recommendation accordingly
        if requires_human_review and recommendation == "ROUTINE_FOLLOWUP":
            recommendation = "FLAGGED_HUMAN_REVIEW"
            action_plan = f"FLAGGED FOR SPECIALIST REVIEW: {'; '.join(review_reasons)}. " + action_plan

        return ScreeningDecision(
            is_gradable=True,
            predicted_grade=grade,
            grade_name=grade_name,
            confidence=confidence,
            probabilities=probs,
            is_referable=referable,
            recommendation=recommendation,
            urgency_tier=urgency_tier,
            action_plan=action_plan,
            requires_human_review=requires_human_review,
            human_review_reasons=review_reasons,
        )
