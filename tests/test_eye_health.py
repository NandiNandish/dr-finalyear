"""
Unit tests for Eye Health Analytics and Downloadable Report endpoints.
"""

import numpy as np
import pytest
from src.analytics.eye_health import EyeHealthAnalyzer, EyeHealthBiomarkers
from app.app import app


def test_eye_health_biomarkers_calculation():
    analyzer = EyeHealthAnalyzer()
    dummy_img = (np.random.rand(224, 224, 3) * 255).astype(np.uint8)
    
    biomarkers = analyzer.analyze(dummy_img, dr_grade=1, confidence=0.88)
    
    assert isinstance(biomarkers, EyeHealthBiomarkers)
    assert 0 <= biomarkers.overall_score <= 100
    assert biomarkers.overall_status in ["Optimal", "Good", "Borderline", "Attention Required"]
    assert biomarkers.hypertensive_risk in ["Low", "Moderate", "Elevated"]
    assert biomarkers.macular_edema_risk in ["Low Risk", "Possible DME", "High Risk (CSME Likely)"]
    assert 0.20 <= biomarkers.estimated_cdr <= 0.85
    assert len(biomarkers.preventive_recommendations) >= 4
    
    data = biomarkers.to_dict()
    assert "overall_score" in data
    assert "vessel_density_pct" in data
    assert "vessel_tortuosity_index" in data


def test_download_report_endpoints():
    client = app.test_client()
    
    # Authenticate
    client.post("/login", data={"username": "clinician", "password": "doctor123"}, follow_redirects=True)
    
    # Seed cache
    from app.app import REPORTS_CACHE
    test_id = "test_unit_999"
    REPORTS_CACHE[test_id] = {
        "report_id": test_id,
        "patient_id": "PAT-UNIT-999",
        "exam_eye": "OD (Right Eye)",
        "active_head": "CLASSICAL",
        "timestamp": "2026-10-22 18:00:00 UTC",
        "is_gradable": True,
        "rejection_code": None,
        "predicted_grade": 0,
        "grade_name": "No Apparent Diabetic Retinopathy",
        "confidence": 0.94,
        "is_referable": False,
        "urgency_tier": "ROUTINE_12_MONTHS",
        "action_plan": "Routine follow-up in 12 months.",
        "probabilities": {"No DR": 0.94, "Mild DR": 0.04, "Moderate DR": 0.01, "Severe DR": 0.005, "PDR": 0.005},
        "original_image_base64": "",
        "gradcam_overlay_base64": None,
        "encrypted_file_ref": "ref_test.enc",
        "eye_health": {
            "overall_score": 92,
            "overall_status": "Optimal",
            "vessel_density_pct": 14.5,
            "vessel_tortuosity_index": 1.25,
            "hypertensive_risk": "Low",
            "vascular_summary": "Stable architecture.",
            "macular_edema_risk": "Low Risk",
            "macular_integrity_score": 24,
            "macular_summary": "Macula intact.",
            "estimated_cdr": 0.36,
            "glaucoma_susceptibility": "Low / Normal",
            "optic_nerve_score": 24,
            "optic_nerve_summary": "Clear margins.",
            "media_clarity_score": 24,
            "cataract_haze_index": "Clear",
            "media_summary": "Transparent media.",
            "preventive_recommendations": [{"title": "HbA1c Target", "detail": "Maintain < 7.0%"}],
        },
    }
    
    # Test JSON download
    res_json = client.get(f"/report/{test_id}/json")
    assert res_json.status_code == 200
    assert res_json.headers["Content-Type"] == "application/json"
    assert "attachment" in res_json.headers["Content-Disposition"]
    
    # Test Print / PDF render
    res_print = client.get(f"/report/{test_id}/print")
    assert res_print.status_code == 200
    assert b"PAT-UNIT-999" in res_print.data
    assert b"National Tele-Ophthalmology Screening Centre" in res_print.data
