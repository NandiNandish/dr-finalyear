"""
End-to-End Single-Image Screening Pipeline.

Executes strictly in the mandated pipeline sequence:
1. Image Acquisition: Loads image from file, stream, or NumPy array.
2. Quality Gate: Rejects blurred, too dark, too bright, or clipped-retina images as 'UNGRA_RECAPTURE'.
   Safety Invariant: Ungradable images NEVER produce a reassuring 'No DR' result.
3. Preprocessing: Crops black borders, enforces circular field, resizes (224px), applies CLAHE.
4. Backbone: Frozen EfficientNet-B0 feature extraction.
5. Interchangeable Head: Evaluates selected active head ('classical' | 'hybrid_quantum' | 'matched_dense').
6. Grading Scale: 5-class severity classification (0 to 4).
7. Explainability: Grad-CAM on backbone conv layer + anatomical plausibility check.
8. Decision Support: Maps grade, confidence, and gradability to structured screening recommendation.
"""

from dataclasses import dataclass
from typing import Union, Dict, Any, Optional
import io
import base64
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from PIL import Image

from src.quality.gate import QualityGate, QualityResult
from src.data.transforms import FundusPreprocessor
from src.models.full_model import DRScreeningModel
from src.explain.gradcam import GradCAM, GradCAMResult
from src.decision import DecisionSupportEngine, ScreeningDecision
from src.data.label_maps import LABEL_TO_NAME, CLASS_NAMES


@dataclass
class ScreeningReport:
    is_gradable: bool
    rejection_code: Optional[str]
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
    anatomical_check_passed: bool
    quality_metrics: Dict[str, Any]
    original_image_base64: str
    gradcam_overlay_base64: Optional[str]
    disclaimer: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_gradable": self.is_gradable,
            "rejection_code": self.rejection_code,
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
            "anatomical_check_passed": self.anatomical_check_passed,
            "quality_metrics": self.quality_metrics,
            "disclaimer": self.disclaimer,
        }


def _ndarray_to_base64(img_rgb: np.ndarray) -> str:
    success, buffer = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    if not success:
        return ""
    return base64.b64encode(buffer).decode("utf-8")


class ScreeningPipeline:
    """
    End-to-end clinical screening pipeline.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        active_head: str = "hybrid_quantum",
        backbone_name: str = "efficientnet_b0",
        image_size: int = 224,
        confidence_threshold: float = 0.60,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.active_head = active_head
        self.image_size = image_size

        # Step 2: Quality Gate
        self.quality_gate = QualityGate()

        # Step 3: Preprocessor
        self.preprocessor = FundusPreprocessor(image_size=image_size, keep_circular=True, use_clahe=True)

        # Step 4 & 5: Model with switchable head
        self.model = DRScreeningModel(
            backbone_name=backbone_name,
            active_head=active_head,
            variational_layers=3,
            diff_method="backprop",
            freeze_backbone=True,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            ckpt = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"], strict=False)
            print(f"Loaded screening checkpoint from: {checkpoint_path}")

        self.model.to(self.device)
        self.model.eval()

        # Step 7: Explainability
        self.gradcam = GradCAM(self.model)

        # Step 8: Decision Support Engine
        self.decision_engine = DecisionSupportEngine(confidence_threshold=confidence_threshold)

    def screen(self, image_input: Union[str, np.ndarray, Image.Image, bytes]) -> ScreeningReport:
        """
        Executes end-to-end screening workflow.
        """
        # 1. Image Acquisition
        if isinstance(image_input, bytes):
            nparr = np.frombuffer(image_input, np.uint8)
            img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, str):
            img_bgr = cv2.imread(image_input)
            if img_bgr is None:
                raise FileNotFoundError(f"Cannot open image: {image_input}")
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        elif isinstance(image_input, Image.Image):
            img_rgb = np.array(image_input.convert("RGB"))
        elif isinstance(image_input, np.ndarray):
            img_rgb = image_input if image_input.ndim == 3 else cv2.cvtColor(image_input, cv2.COLOR_GRAY2RGB)
        else:
            raise TypeError(f"Unsupported image type: {type(image_input)}")

        orig_b64 = _ndarray_to_base64(img_rgb)

        # 2. Quality Gate Assessment
        q_result: QualityResult = self.quality_gate.assess(img_rgb)

        # INVARIANT: If ungradable, immediately return RECAPTURE report.
        # NEVER proceed to classification or output "No DR".
        if not q_result.is_gradable:
            decision: ScreeningDecision = self.decision_engine.evaluate(quality_result=q_result)
            return ScreeningReport(
                is_gradable=False,
                rejection_code=q_result.rejection_code,
                predicted_grade=None,
                grade_name="Ungradable (Quality Rejected)",
                confidence=0.0,
                probabilities={},
                is_referable=True,
                recommendation=decision.recommendation,
                urgency_tier=decision.urgency_tier,
                action_plan=decision.action_plan,
                requires_human_review=True,
                human_review_reasons=decision.human_review_reasons,
                anatomical_check_passed=False,
                quality_metrics=q_result.metrics,
                original_image_base64=orig_b64,
                gradcam_overlay_base64=None,
                disclaimer=decision.disclaimer,
            )

        # 3. Preprocessing (Crop borders, circular FOV, resize 224, CLAHE)
        proc_rgb, retina_mask = self.preprocessor.preprocess_numpy(img_rgb)
        img_tensor = self.preprocessor(img_rgb).unsqueeze(0).to(self.device)

        # 4, 5, 6. Backbone & Active Head Forward Pass
        with torch.no_grad():
            logits = self.model(img_tensor)
            probs = F.softmax(logits, dim=-1).cpu().numpy()[0]
            pred_grade = int(np.argmax(probs))
            confidence = float(probs[pred_grade])

        prob_dict = {CLASS_NAMES[i]: round(float(probs[i]), 4) for i in range(len(CLASS_NAMES))}

        # 7. Explainability: Grad-CAM through backbone
        # Requires gradient tracking on tensor
        img_tensor_grad = img_tensor.clone().detach().requires_grad_(True)
        cam_result: GradCAMResult = self.gradcam.generate(
            image_tensor=img_tensor_grad,
            original_rgb=proc_rgb,
            target_class=pred_grade,
            retina_mask=retina_mask,
        )
        cam_b64 = _ndarray_to_base64(cam_result.overlay_rgb)

        # 8. Decision Support Mapping
        decision: ScreeningDecision = self.decision_engine.evaluate(
            quality_result=q_result,
            predicted_grade=pred_grade,
            confidence=confidence,
            probabilities=prob_dict,
            gradcam_outside_retina=cam_result.lies_outside_retina,
        )

        return ScreeningReport(
            is_gradable=True,
            rejection_code=None,
            predicted_grade=pred_grade,
            grade_name=decision.grade_name,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            is_referable=decision.is_referable,
            recommendation=decision.recommendation,
            urgency_tier=decision.urgency_tier,
            action_plan=decision.action_plan,
            requires_human_review=decision.requires_human_review,
            human_review_reasons=decision.human_review_reasons,
            anatomical_check_passed=not cam_result.lies_outside_retina,
            quality_metrics=q_result.metrics,
            original_image_base64=orig_b64,
            gradcam_overlay_base64=cam_b64,
            disclaimer=decision.disclaimer,
        )
