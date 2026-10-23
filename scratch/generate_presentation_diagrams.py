"""
Generates high-resolution publication-quality flowcharts, diagrams, and visual figures
for the Phase-II Major Project Review Presentation.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Set high DPI and aesthetic styling
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cbd5e1'
plt.rcParams['axes.linewidth'] = 1.2
Path("reports").mkdir(exist_ok=True)

# ==============================================================================
# 1. Pipeline Architecture Flowchart (Fig 1) for Slide 3
# ==============================================================================
def generate_pipeline_flowchart():
    fig, ax = plt.subplots(figsize=(12, 5.2), dpi=200)
    ax.set_facecolor('#0f172a')  # Dark slate
    fig.patch.set_facecolor('#0f172a')
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.5)
    ax.axis('off')

    # Flowchart boxes
    boxes = [
        {"x": 0.3, "y": 2.2, "w": 1.4, "h": 1.4, "title": "1. Fundus Image\nAcquisition", "sub": "45°/50° Color\nRGB Photo", "color": "#0284c7", "edge": "#38bdf8"},
        {"x": 2.0, "y": 2.2, "w": 1.4, "h": 1.4, "title": "2. Quality\nGate Filter", "sub": "Sharpness, FOV\nContrast Checks", "color": "#d97706", "edge": "#fbbf24"},
        {"x": 3.7, "y": 2.2, "w": 1.4, "h": 1.4, "title": "3. Pre-\nprocessing", "sub": "Circular Mask\n224×224 + CLAHE", "color": "#2563eb", "edge": "#60a5fa"},
        {"x": 5.4, "y": 2.2, "w": 1.4, "h": 1.4, "title": "4. Frozen CNN\nBackbone", "sub": "EfficientNet-B0\n1280-dim Features", "color": "#7c3aed", "edge": "#a78bfa"},
        {"x": 7.1, "y": 2.2, "w": 1.5, "h": 1.4, "title": "5. Hybrid\nQuantum Head", "sub": "4-Qubit VQC\nRy+CNOT (L=3)", "color": "#0d9488", "edge": "#2dd4bf"},
        {"x": 8.9, "y": 2.2, "w": 1.4, "h": 1.4, "title": "6. Grad-CAM\nExplainability", "sub": "Heatmap Overlay\nDisc Verification", "color": "#e11d48", "edge": "#fb7185"},
        {"x": 10.6, "y": 2.2, "w": 1.3, "h": 1.4, "title": "7. Decision\n& Triage", "sub": "5-Level ICDR\nUrgency Tiers", "color": "#059669", "edge": "#34d399"},
    ]

    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.08,rounding_size=0.15",
                                      facecolor=b["color"], edgecolor=b["edge"], linewidth=2, alpha=0.9)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.62, b["title"], color="white", fontsize=8.5, weight="bold",
                ha="center", va="center")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.25, b["sub"], color="#f1f5f9", fontsize=7.2,
                ha="center", va="center")

    # Connect forward arrows
    for i in range(len(boxes) - 1):
        ax.annotate('', xy=(boxes[i+1]["x"], boxes[i]["y"] + boxes[i]["h"]/2),
                    xytext=(boxes[i]["x"] + boxes[i]["w"], boxes[i]["y"] + boxes[i]["h"]/2),
                    arrowprops=dict(arrowstyle="-|>", color="#38bdf8", lw=2.5, mutation_scale=15))

    # Quality Gate Recapture branch (Bottom arrow)
    ax.annotate('', xy=(2.7, 0.7), xytext=(2.7, 2.2),
                arrowprops=dict(arrowstyle="-|>", color="#f43f5e", lw=2.5, mutation_scale=15))
    recapt_box = patches.FancyBboxPatch((1.8, 0.2), 1.8, 0.75, boxstyle="round,pad=0.06,rounding_size=0.12",
                                        facecolor="#881337", edgecolor="#f43f5e", linewidth=2)
    ax.add_patch(recapt_box)
    ax.text(2.7, 0.65, "FAILED QUALITY GATE", color="#fecdd3", fontsize=7.5, weight="bold", ha="center")
    ax.text(2.7, 0.40, "Mandatory Recapture Required\n(Invariant: NEVER outputs 'No DR')", color="white", fontsize=6.8, ha="center")

    # Title Banner
    ax.text(6.0, 4.9, "Proposed End-to-End Clinical Screening & Quantum Pipeline", color="#38bdf8",
            fontsize=13, weight="bold", ha="center")
    ax.text(6.0, 4.4, "Quality-Gated Input ➔ Preprocessing ➔ Frozen EfficientNet-B0 ➔ 4-Qubit VQC ➔ Grad-CAM ➔ Decision Triage",
            color="#94a3b8", fontsize=8.5, ha="center")

    plt.tight_layout()
    out = Path("reports/fig1_pipeline_flowchart.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out}")

# ==============================================================================
# 2. Performance Comparison Bar Chart (Fig 3 from Paper) for Slide 5
# ==============================================================================
def generate_performance_chart():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8), dpi=200)
    fig.patch.set_facecolor('#0f172a')

    # Subplot 1: Sensitivity & Specificity Comparison
    ax1.set_facecolor('#1e293b')
    studies = ['Gulshan et al.\n(JAMA 2016)', 'Ting et al.\n(JAMA 2017)', 'IDx-DR\n(npj 2018)', 'Proposed System\n(RFMiD Holdout)']
    sens = [90.3, 90.5, 87.2, 100.0]
    spec = [98.1, 91.6, 90.7, 81.2]
    
    x = np.arange(len(studies))
    width = 0.35

    rects1 = ax1.bar(x - width/2, sens, width, label='Sensitivity (%)', color='#0ea5e9', edgecolor='#38bdf8', linewidth=1.5)
    rects2 = ax1.bar(x + width/2, spec, width, label='Specificity (%)', color='#10b981', edgecolor='#34d399', linewidth=1.5)

    ax1.set_ylabel('Reported Performance (%)', color='white', fontsize=9.5, weight='bold')
    ax1.set_title('Referable-DR Clinical Diagnostic Screening', color='#38bdf8', fontsize=11, weight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(studies, color='white', fontsize=8)
    ax1.set_ylim(0, 115)
    ax1.tick_params(colors='white')
    ax1.legend(loc='lower right', facecolor='#0f172a', edgecolor='#475569', labelcolor='white', fontsize=8.5)
    ax1.grid(axis='y', linestyle='--', alpha=0.2, color='white')

    # Bar labels
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='#38bdf8', fontsize=7.5, weight='bold')
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='#34d399', fontsize=7.5, weight='bold')

    # Subplot 2: Controlled Architecture Head Ablation
    ax2.set_facecolor('#1e293b')
    heads = ['Classical Head\n(6,405 params)', 'Matched Dense\n(5,149 params)', 'Hybrid Quantum\n(5,161 params)']
    qwk = [0.650, 0.628, 0.638]
    colors = ['#6366f1', '#f59e0b', '#06b6d4']

    bars = ax2.bar(heads, qwk, width=0.52, color=colors, edgecolor='white', linewidth=1.2)
    ax2.set_ylabel('Holdout Quadratic Weighted Kappa (QWK)', color='white', fontsize=9.5, weight='bold')
    ax2.set_title('Controlled Ablation on Identical Frozen Features', color='#38bdf8', fontsize=11, weight='bold')
    ax2.set_ylim(0, 0.85)
    ax2.tick_params(colors='white')
    ax2.grid(axis='y', linestyle='--', alpha=0.2, color='white')

    for bar in bars:
        h = bar.get_height()
        ax2.annotate(f'QWK = {h:.3f}', xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', color='white', fontsize=8.5, weight='bold')

    plt.tight_layout()
    out = Path("reports/fig2_performance_charts.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out}")

# ==============================================================================
# 3. SDG Alignment Infographic for Slide 7
# ==============================================================================
def generate_sdg_infographic():
    fig, ax = plt.subplots(figsize=(11, 4.4), dpi=200)
    fig.patch.set_facecolor('#0f172a')
    ax.set_facecolor('#0f172a')
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.5)
    ax.axis('off')

    sdgs = [
        {
            "x": 0.4, "w": 3.2, "title": "SDG 3: GOOD HEALTH & WELL-BEING",
            "tag": "Targets 3.8 & 3.d", "color": "#4c0519", "accent": "#f43f5e",
            "icon": "❤️🏥",
            "bullets": [
                "• Prevents permanent blindness in working-age adults.",
                "• Early risk stratification for Diabetic Retinopathy.",
                "• Expands tele-screening capacity in primary clinics."
            ]
        },
        {
            "x": 3.9, "w": 3.2, "title": "SDG 9: INDUSTRY & INNOVATION",
            "tag": "Target 9.5", "color": "#431407", "accent": "#f97316",
            "icon": "⚛️🔬",
            "bullets": [
                "• Pioneers Quantum Machine Learning (QML) in health.",
                "• Dressed 4-Qubit Variational Quantum Circuit.",
                "• Parameter-shift optimization on medical imaging."
            ]
        },
        {
            "x": 7.4, "w": 3.2, "title": "SDG 10: REDUCED INEQUALITIES",
            "tag": "Targets 10.2 & 10.3", "color": "#3b0764", "accent": "#d946ef",
            "icon": "⚖️🌍",
            "bullets": [
                "• Bridges acute specialist shortage in rural areas.",
                "• Low-cost decentralized tele-ophthalmology triage.",
                "• No resident ophthalmologist required at point-of-care."
            ]
        }
    ]

    for s in sdgs:
        box = patches.FancyBboxPatch((s["x"], 0.3), s["w"], 3.8, boxstyle="round,pad=0.1,rounding_size=0.18",
                                     facecolor=s["color"], edgecolor=s["accent"], linewidth=2.5)
        ax.add_patch(box)
        ax.text(s["x"] + s["w"]/2, 3.65, s["title"], color=s["accent"], fontsize=9.2, weight="bold", ha="center")
        ax.text(s["x"] + s["w"]/2, 3.30, s["tag"], color="#f1f5f9", fontsize=8, weight="bold", ha="center")
        
        y_text = 2.75
        for b in s["bullets"]:
            ax.text(s["x"] + 0.2, y_text, b, color="#e2e8f0", fontsize=7.6, ha="left", va="top")
            y_text -= 0.65

    plt.tight_layout()
    out = Path("reports/fig3_sdg_infographic.png")
    fig.savefig(out, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"Generated {out}")

# ==============================================================================
# 4. Multi-Panel Screenshot Gallery for Slide 8
# ==============================================================================
def generate_screenshot_gallery():
    # Load composite or create multi-panel grid
    src = Path("reports/presentation_screenshots_composite.png")
    if src.exists():
        img = Image.open(src)
        # Resave as gallery
        out = Path("reports/fig4_screenshots_gallery.png")
        img.save(out, dpi=(200, 200))
        print(f"Saved {out}")

generate_pipeline_flowchart()
generate_performance_chart()
generate_sdg_infographic()
generate_screenshot_gallery()
