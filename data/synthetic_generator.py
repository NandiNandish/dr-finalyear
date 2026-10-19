"""
Synthetic Fundus Image and Manifest Generator.

Generates anatomically plausible synthetic fundus images for 5 DR stages
(No DR, Mild, Moderate, Severe NPDR, PDR) and ungradable categories (blurred,
too dark, overexposed, low retina area).

This enables complete end-to-end training, testing, evaluation, and CI
without requiring multi-gigabyte external downloads.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
import cv2
import pandas as pd


def generate_synthetic_fundus(
    grade: int,
    image_size: int = 256,
    rng: np.random.Generator = None,
    is_ungradable: str = None,
) -> np.ndarray:
    """
    Generates a single synthetic colour fundus image.
    grade: 0 to 4
    is_ungradable: None | 'blur' | 'dark' | 'bright' | 'partial_retina'
    """
    if rng is None:
        rng = np.random.default_rng()

    h, w = image_size, image_size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Center & radius of retinal circle
    cx, cy = w // 2, h // 2
    radius = int(min(cx, cy) * 0.90)

    # 1. Circular mask and radial gradient background (fundus orange-red)
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    retina_mask = dist <= radius

    norm_dist = np.clip(dist / radius, 0.0, 1.0)
    # Base fundus tones: Orange-red (R: 180-230, G: 60-90, B: 20-40), darker at edge
    red_chan = np.clip(220 - norm_dist * 80 + rng.normal(0, 3, (h, w)), 0, 255)
    green_chan = np.clip(85 - norm_dist * 45 + rng.normal(0, 2, (h, w)), 0, 255)
    blue_chan = np.clip(30 - norm_dist * 18 + rng.normal(0, 2, (h, w)), 0, 255)

    img[:, :, 0] = (red_chan * retina_mask).astype(np.uint8)
    img[:, :, 1] = (green_chan * retina_mask).astype(np.uint8)
    img[:, :, 2] = (blue_chan * retina_mask).astype(np.uint8)

    # 2. Optic Disc (yellowish-white ellipse on nasal side)
    od_x = cx - int(radius * 0.45)
    od_y = cy - int(radius * 0.05)
    od_rx, od_ry = int(radius * 0.14), int(radius * 0.16)
    cv2.ellipse(img, (od_x, od_y), (od_rx, od_ry), 0, 0, 360, (255, 230, 170), -1)

    # 3. Retinal Blood Vessels (dark red branches emerging from optic disc)
    vessel_color = (110, 20, 10)
    for _ in range(8):
        # Arc paths
        pts = []
        cur_x, cur_y = od_x, od_y
        pts.append((cur_x, cur_y))
        angle = rng.uniform(-np.pi / 2, np.pi / 2) + (0 if rng.random() > 0.5 else np.pi)
        for step in range(6):
            cur_x += int(np.cos(angle) * (radius * 0.15))
            cur_y += int(np.sin(angle) * (radius * 0.15))
            angle += rng.uniform(-0.3, 0.3)
            pts.append((cur_x, cur_y))
        pts_arr = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(img, [pts_arr], isClosed=False, color=vessel_color, thickness=rng.integers(2, 4))

    # 4. Fovea / Macula (darker circular depression temporal to optic disc)
    fovea_x = cx + int(radius * 0.25)
    fovea_y = cy
    fovea_mask = ((X - fovea_x) ** 2 + (Y - fovea_y) ** 2) <= (radius * 0.22) ** 2
    img[fovea_mask, 0] = np.clip(img[fovea_mask, 0] * 0.75, 0, 255).astype(np.uint8)
    img[fovea_mask, 1] = np.clip(img[fovea_mask, 1] * 0.65, 0, 255).astype(np.uint8)

    # 5. Lesion synthesis based on Diabetic Retinopathy grade
    if grade >= 1:
        # Mild: Microaneurysms (tiny distinct red dots, 1-2 px)
        num_ma = rng.integers(4, 10) if grade == 1 else rng.integers(12, 30)
        for _ in range(num_ma):
            lx = int(cx + rng.uniform(-radius * 0.7, radius * 0.7))
            ly = int(cy + rng.uniform(-radius * 0.7, radius * 0.7))
            if ((lx - cx) ** 2 + (ly - cy) ** 2) < (radius * 0.8) ** 2:
                cv2.circle(img, (lx, ly), rng.integers(1, 3), (120, 10, 10), -1)

    if grade >= 2:
        # Moderate: Hard exudates (bright yellowish lipid clusters) and blot hemorrhages
        num_exudates = rng.integers(3, 8) if grade == 2 else rng.integers(8, 20)
        for _ in range(num_exudates):
            ex = int(cx + rng.uniform(-radius * 0.6, radius * 0.6))
            ey = int(cy + rng.uniform(-radius * 0.6, radius * 0.6))
            if ((ex - cx) ** 2 + (ey - cy) ** 2) < (radius * 0.8) ** 2:
                for _ in range(rng.integers(3, 8)):
                    off_x, off_y = rng.integers(-6, 7), rng.integers(-6, 7)
                    cv2.circle(img, (ex + off_x, ey + off_y), rng.integers(2, 4), (250, 245, 180), -1)

        # Dot-blot hemorrhages
        num_blots = rng.integers(4, 12) if grade == 2 else rng.integers(15, 35)
        for _ in range(num_blots):
            bx = int(cx + rng.uniform(-radius * 0.7, radius * 0.7))
            by = int(cy + rng.uniform(-radius * 0.7, radius * 0.7))
            if ((bx - cx) ** 2 + (by - cy) ** 2) < (radius * 0.8) ** 2:
                cv2.circle(img, (bx, by), rng.integers(3, 7), (90, 10, 10), -1)

    if grade >= 3:
        # Severe NPDR: Cotton wool spots (feathery soft exudates) & extensive 4-quadrant hemorrhages
        for _ in range(rng.integers(4, 9)):
            cwx = int(cx + rng.uniform(-radius * 0.6, radius * 0.6))
            cwy = int(cy + rng.uniform(-radius * 0.6, radius * 0.6))
            if ((cwx - cx) ** 2 + (cwy - cy) ** 2) < (radius * 0.75) ** 2:
                overlay = img.copy()
                cv2.circle(overlay, (cwx, cwy), rng.integers(6, 12), (240, 240, 240), -1)
                cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)

    if grade >= 4:
        # Proliferative DR: Neovascularization fronds (tortuous fine tangled vessels) & large preretinal pool
        for _ in range(rng.integers(3, 6)):
            nvx, nvy = od_x + rng.integers(-20, 21), od_y + rng.integers(-20, 21)
            for _ in range(10):
                nx2 = nvx + rng.integers(-15, 16)
                ny2 = nvy + rng.integers(-15, 16)
                cv2.line(img, (nvx, nvy), (nx2, ny2), (160, 20, 15), 1)
                nvx, nvy = nx2, ny2

        # Vitreous / Preretinal boat-shaped hemorrhage
        vhx = cx + int(radius * 0.3)
        vhy = cy + int(radius * 0.3)
        cv2.ellipse(img, (vhx, vhy), (int(radius * 0.2), int(radius * 0.1)), 15, 0, 180, (70, 5, 5), -1)

    # 6. Apply deliberate ungradable artefacts if requested
    if is_ungradable == "blur":
        # Severe defocus / motion blur
        img = cv2.GaussianBlur(img, (31, 31), 15.0)

    elif is_ungradable == "dark":
        # Underexposed / opaque media
        img = (img * 0.06).astype(np.uint8)

    elif is_ungradable == "bright":
        # Flash glare overexposure
        img = np.clip(img.astype(np.float32) + 210, 0, 255).astype(np.uint8)

    elif is_ungradable == "partial_retina":
        # Retinal disc clipped off (keep only small strip)
        partial_mask = np.zeros((h, w), dtype=bool)
        partial_mask[:, : int(w * 0.28)] = True
        img = img * partial_mask[:, :, None]

    return img


def create_demo_dataset(
    output_dir: str = "data/demo",
    manifest_path: str = "data/manifests/demo_manifest.csv",
    external_manifest_path: str = "data/manifests/external_test.csv",
    num_patients_train: int = 16,
    num_patients_val: int = 6,
    num_patients_test: int = 8,
    num_patients_external: int = 10,
    images_per_patient: int = 2,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates patient-disjoint synthetic dataset and manifest files.
    """
    rng = np.random.default_rng(seed)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    Path(manifest_path).parent.mkdir(parents=True, exist_ok=True)

    records = []

    def process_split_patients(patient_prefix: str, count: int, split_name: str, dataset_name: str):
        for i in range(count):
            p_id = f"{patient_prefix}_{i:03d}"
            # Patient has a primary DR severity
            # Bias distribution realistically: more 0s and 1s, fewer 3s and 4s
            p_grade = int(rng.choice([0, 1, 2, 3, 4], p=[0.40, 0.25, 0.15, 0.10, 0.10]))

            for eye_idx, eye in enumerate(["left", "right"][:images_per_patient]):
                # Eye grade may vary slightly (usually identical or +-1)
                grade_delta = int(rng.choice([-1, 0, 1], p=[0.1, 0.8, 0.1]))
                eye_grade = int(np.clip(p_grade + grade_delta, 0, 4))

                filename = f"{dataset_name}_{p_id}_{eye}.png"
                filepath = out_path / filename

                img = generate_synthetic_fundus(grade=eye_grade, image_size=256, rng=rng)
                # Save as RGB PNG
                cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                records.append({
                    "path": str(filepath.as_posix()),
                    "label": eye_grade,
                    "dataset": dataset_name,
                    "patient_id": p_id,
                    "eye": eye,
                    "split": split_name,
                    "is_ungradable": 0,
                })

    # Internal splits (EyePACS/IDRiD emulation)
    process_split_patients("PAT_TR", num_patients_train, "train", "eyepacs_demo")
    process_split_patients("PAT_VA", num_patients_val, "val", "eyepacs_demo")
    process_split_patients("PAT_TE", num_patients_test, "test", "eyepacs_demo")

    # Add deliberate ungradable samples to test and val for Quality Gate validation
    ungradable_types = ["blur", "dark", "bright", "partial_retina"]
    for idx, u_type in enumerate(ungradable_types):
        for s_idx, split_name in enumerate(["val", "test"]):
            p_id = f"PAT_UNG_{split_name}_{idx:02d}"
            filename = f"ungradable_{u_type}_{split_name}.png"
            filepath = out_path / filename
            img = generate_synthetic_fundus(grade=2, image_size=256, rng=rng, is_ungradable=u_type)
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            records.append({
                "path": str(filepath.as_posix()),
                "label": -1,  # -1 denotes clinically ungradable
                "dataset": "eyepacs_demo",
                "patient_id": p_id,
                "eye": "left" if s_idx == 0 else "right",
                "split": split_name,
                "is_ungradable": 1,
            })

    df_internal = pd.DataFrame(records)
    df_internal.to_csv(manifest_path, index=False)

    # External test set (Messidor/APTOS emulation)
    ext_records = []
    for i in range(num_patients_external):
        p_id = f"EXT_PAT_{i:03d}"
        p_grade = int(rng.choice([0, 1, 2, 3, 4], p=[0.35, 0.25, 0.20, 0.10, 0.10]))
        for eye in ["left", "right"]:
            filename = f"messidor_demo_{p_id}_{eye}.png"
            filepath = out_path / filename
            img = generate_synthetic_fundus(grade=p_grade, image_size=256, rng=rng)
            cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            ext_records.append({
                "path": str(filepath.as_posix()),
                "label": p_grade,
                "dataset": "messidor_demo",
                "patient_id": p_id,
                "eye": eye,
                "split": "external",
                "is_ungradable": 0,
            })

    df_external = pd.DataFrame(ext_records)
    df_external.to_csv(external_manifest_path, index=False)

    return df_internal, df_external


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic fundus demo dataset.")
    parser.add_argument("--output_dir", default="data/demo", help="Directory to save images")
    parser.add_argument("--manifest", default="data/manifests/demo_manifest.csv", help="Internal manifest path")
    parser.add_argument("--external_manifest", default="data/manifests/external_test.csv", help="External manifest path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    df_int, df_ext = create_demo_dataset(
        output_dir=args.output_dir,
        manifest_path=args.manifest,
        external_manifest_path=args.external_manifest,
        seed=args.seed,
    )
    print(f"Generated internal manifest: {len(df_int)} rows -> {args.manifest}")
    print(f"Generated external manifest: {len(df_ext)} rows -> {args.external_manifest}")
