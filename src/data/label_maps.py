"""
Label mapping and clinical severity definitions for Diabetic Retinopathy screening.
Follows the International Clinical Diabetic Retinopathy (ICDR) scale.
"""

from enum import IntEnum
from typing import Union, Dict, Any


class DRGrade(IntEnum):
    NO_DR = 0
    MILD_NPDR = 1
    MODERATE_NPDR = 2
    SEVERE_NPDR = 3
    PROLIFERATIVE_DR = 4


CLASS_NAMES = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe NPDR",
    "Proliferative DR",
]

LABEL_TO_NAME: Dict[int, str] = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe NPDR",
    4: "Proliferative DR",
}

NAME_TO_LABEL: Dict[str, int] = {
    "No DR": 0,
    "Mild": 1,
    "Moderate": 2,
    "Severe NPDR": 3,
    "Proliferative DR": 4,
    "PDR": 4,
    "No_DR": 0,
    "Mild_NPDR": 1,
    "Moderate_NPDR": 2,
    "Severe_NPDR": 3,
}

REFERABLE_THRESHOLD_GRADE = 2  # Grade >= 2 (Moderate, Severe, PDR) is Referable DR


def is_referable_dr(grade: Union[int, DRGrade]) -> bool:
    """
    Returns True if the diabetic retinopathy grade is clinically referable (>= Moderate).
    Grade 0 (No DR) and Grade 1 (Mild NPDR) are Non-referable.
    Grade 2 (Moderate NPDR), Grade 3 (Severe NPDR), and Grade 4 (PDR) are Referable.
    """
    return int(grade) >= REFERABLE_THRESHOLD_GRADE


def map_dataset_label(dataset_name: str, raw_label: Any) -> int:
    """
    Harmonizes dataset-specific labels into canonical 0-4 integer grades.
    """
    d_clean = dataset_name.lower().strip()
    
    # Handle string names
    if isinstance(raw_label, str) and raw_label.strip() in NAME_TO_LABEL:
        return NAME_TO_LABEL[raw_label.strip()]
        
    try:
        raw_int = int(float(raw_label))
    except (ValueError, TypeError):
        raise ValueError(f"Unrecognized raw label '{raw_label}' for dataset '{dataset_name}'")

    if d_clean in ["eyepacs", "idrid", "ddr", "aptos", "aptos2019", "synthetic", "demo"]:
        if 0 <= raw_int <= 4:
            return raw_int
        raise ValueError(f"Label {raw_int} out of bounds for standard 5-class dataset {dataset_name}")

    elif d_clean in ["messidor", "messidor-1", "messidor1"]:
        # Messidor original 0-3 grading
        # 0: 0 MAs -> 0 (No DR)
        # 1: 1-5 MAs -> 1 (Mild)
        # 2: 6-14 MAs -> 2 (Moderate)
        # 3: >=15 MAs or NV -> 3 (Severe)
        if raw_int == 0:
            return 0
        elif raw_int == 1:
            return 1
        elif raw_int == 2:
            return 2
        elif raw_int == 3:
            return 3
        else:
            raise ValueError(f"Messidor label {raw_int} out of expected [0, 3] range")

    elif d_clean in ["messidor-2", "messidor2"]:
        if 0 <= raw_int <= 4:
            return raw_int
        raise ValueError(f"Messidor-2 label {raw_int} out of expected range")

    else:
        # Default fallback
        if 0 <= raw_int <= 4:
            return raw_int
        raise ValueError(f"Unknown dataset '{dataset_name}' with label {raw_int}")
