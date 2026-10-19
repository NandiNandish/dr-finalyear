"""
Unit tests for the Fundus Quality Gate.

Safety Invariant:
Reject blurred, too dark, artefact-ridden, or too-little-retina images as
'ungradable -> recapture'. An ungradable image must NEVER produce a reassuring 'No DR' result.
"""

import pytest
import numpy as np
from src.quality.gate import QualityGate, QualityResult
from data.synthetic_generator import generate_synthetic_fundus


@pytest.fixture
def gate():
    return QualityGate(
        min_laplacian_var=80.0,
        min_mean_brightness=25.0,
        max_mean_brightness=220.0,
        min_contrast_std=15.0,
        min_retina_area_ratio=0.55,
        max_saturated_fraction=0.15,
    )


def test_gradable_fundus_passes(gate):
    # Standard synthetic fundus with sharp vessels and optic disc
    img = generate_synthetic_fundus(grade=0, image_size=256)
    result = gate.assess(img)

    assert result.is_gradable is True
    assert result.rejection_code is None
    assert result.recommendation == "PROCEED_TO_GRADING"
    assert result.metrics["laplacian_var"] >= gate.min_laplacian_var
    assert result.metrics["retina_area_ratio"] >= gate.min_retina_area_ratio


def test_blurred_image_rejected(gate):
    img = generate_synthetic_fundus(grade=2, image_size=256, is_ungradable="blur")
    result = gate.assess(img)

    assert result.is_gradable is False
    assert result.rejection_code == "UNGRA_BLURRED"
    assert result.recommendation == "RECAPTURE"
    assert result.metrics["laplacian_var"] < gate.min_laplacian_var


def test_dark_image_rejected(gate):
    img = generate_synthetic_fundus(grade=1, image_size=256, is_ungradable="dark")
    result = gate.assess(img)

    assert result.is_gradable is False
    assert result.rejection_code == "UNGRA_TOO_DARK"
    assert result.recommendation == "RECAPTURE"
    assert result.metrics["mean_brightness"] < gate.min_mean_brightness


def test_bright_image_rejected(gate):
    img = generate_synthetic_fundus(grade=3, image_size=256, is_ungradable="bright")
    result = gate.assess(img)

    assert result.is_gradable is False
    assert result.rejection_code in ["UNGRA_TOO_BRIGHT", "UNGRA_FLASH_ARTEFACT"]
    assert result.recommendation == "RECAPTURE"


def test_partial_retina_rejected(gate):
    img = generate_synthetic_fundus(grade=0, image_size=256, is_ungradable="partial_retina")
    result = gate.assess(img)

    assert result.is_gradable is False
    assert result.rejection_code == "UNGRA_TOO_LITTLE_RETINA"
    assert result.recommendation == "RECAPTURE"
    assert result.metrics["retina_area_ratio"] < gate.min_retina_area_ratio


def test_safety_invariant_ungradable_never_no_dr(gate):
    """
    CRITICAL SAFETY INVARIANT:
    An ungradable image must NEVER produce a reassuring 'No DR' result.
    It must be caught at the gate and assigned 'RECAPTURE'.
    """
    ungradable_types = ["blur", "dark", "bright", "partial_retina"]
    for u_type in ungradable_types:
        img = generate_synthetic_fundus(grade=0, image_size=256, is_ungradable=u_type)
        result = gate.assess(img)

        # Gate MUST reject
        assert result.is_gradable is False, f"Ungradable image type '{u_type}' erroneously passed quality gate!"
        assert result.recommendation == "RECAPTURE"
        # Ensure it cannot be misconstrued as gradable
        assert result.rejection_code.startswith("UNGRA_")
