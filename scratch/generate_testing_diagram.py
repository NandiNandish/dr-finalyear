"""
Generates visual testing and verification matrix diagram for Slide 4.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.5)
ax.axis('off')

cards = [
    {
        "x": 0.4, "w": 2.35, "title": "1. Automated Test Suite",
        "tag": "36 / 36 PASSED (100%)", "color": "#064e3b", "accent": "#10b981",
        "bullets": [
            "• Unit & integration tests",
            "• Quality gate blur checks",
            "• EfficientNet-B0 embedding",
            "• VQC forward/backward pass",
            "• Decision engine triage rules",
            "• Eye health score metrics"
        ]
    },
    {
        "x": 3.05, "w": 2.35, "title": "2. Safety Invariant Gate",
        "tag": "100% RECAPTURE RULE", "color": "#701a75", "accent": "#d946ef",
        "bullets": [
            "• Blurry / degraded images",
            "• Underexposure (< 35 lum)",
            "• Clipped FOV (< 42% area)",
            "• NEVER outputs 'No DR'",
            "• Zero false-reassurance",
            "• Clinically verified safety"
        ]
    },
    {
        "x": 5.7, "w": 2.35, "title": "3. Multi-Dataset Benchmarks",
        "tag": "3 DIVERSE COHORTS", "color": "#0c4a6e", "accent": "#0ea5e9",
        "bullets": [
            "• ROC Dataset: Microaneurysms",
            "  (Holdout QWK = 0.650)",
            "• DRIVE Dataset: FOV masks",
            "  (>98% spatial overlap)",
            "• RFMiD 2.0: 854 multi-disease",
            "  (100% sens, 81.2% spec)"
        ]
    },
    {
        "x": 8.35, "w": 2.35, "title": "4. Security & Interoperability",
        "tag": "ENTERPRISE TELE-MED", "color": "#431407", "accent": "#f97316",
        "bullets": [
            "• AES-GCM 256-bit encryption",
            "  for images stored at rest",
            "• Role-based authentication",
            "• HL7 / DICOM JSON export",
            "• ISO 13485 medical PDF",
            "• Restful /predict API"
        ]
    }
]

for c in cards:
    box = patches.FancyBboxPatch((c["x"], 0.25), c["w"], 3.9, boxstyle="round,pad=0.08,rounding_size=0.15",
                                 facecolor=c["color"], edgecolor=c["accent"], linewidth=2.2)
    ax.add_patch(box)
    ax.text(c["x"] + c["w"]/2, 3.75, c["title"], color=c["accent"], fontsize=8.8, weight="bold", ha="center")
    ax.text(c["x"] + c["w"]/2, 3.40, c["tag"], color="#ffffff", fontsize=7.5, weight="bold", ha="center")
    
    y = 2.95
    for b in c["bullets"]:
        ax.text(c["x"] + 0.15, y, b, color="#e2e8f0", fontsize=7.2, ha="left", va="top")
        y -= 0.42

plt.tight_layout()
out = Path("reports/fig5_testing_architecture.png")
fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print(f"Generated {out}")
