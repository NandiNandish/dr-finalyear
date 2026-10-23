"""
Builds an exceptionally presentable, colorful, and visual PowerPoint presentation
for the Phase-II Major Project Review.
Integrates custom flowcharts, quantum architectures, performance graphs, SDG cards, and UI screenshots.
"""

from pathlib import Path
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

TEMPLATE_PATH = r"H:\Downloads\project  review  phase-II-P2  ppts Format.pptx"
OUT_DOWNLOADS = r"H:\Downloads\project_review_phase_II_P2_colorful_presentation.pptx"
OUT_REPORTS = r"reports\project_review_phase_II_P2_colorful_presentation.pptx"

FIG_PIPELINE = r"reports\fig1_pipeline_flowchart.png"
FIG_PERF = r"reports\fig2_performance_charts.png"
FIG_SDG = r"reports\fig3_sdg_infographic.png"
FIG_SCREENSHOTS = r"reports\fig4_screenshots_gallery.png"
FIG_TESTING = r"reports\fig5_testing_architecture.png"

prs = Presentation(TEMPLATE_PATH)
print(f"Loaded template ({prs.slide_width/914400:.1f}x{prs.slide_height/914400:.1f} inches), {len(prs.slides)} slides.")

# Theme Colors
COLOR_CYAN = RGBColor(14, 165, 233)      # #0ea5e9
COLOR_NAVY = RGBColor(15, 23, 42)        # #0f172a
COLOR_EMERALD = RGBColor(16, 185, 129)   # #10b981
COLOR_ROSE = RGBColor(244, 63, 94)       # #f43f5e
COLOR_PURPLE = RGBColor(139, 92, 246)    # #8b5cf6
COLOR_DARK_TEXT = RGBColor(30, 41, 59)   # #1e293b
COLOR_MUTED = RGBColor(100, 116, 139)    # #64748b

# ==============================================================================
# Slide 1: Title Slide
# ==============================================================================
slide1 = prs.slides[0]
for shape in slide1.shapes:
    if shape.name == "Subtitle 2" and shape.has_text_frame:
        tf = shape.text_frame
        tf.clear()
        
        p = tf.paragraphs[0]
        p.text = "The Project Phase-II Presentation On"
        p.font.size = Pt(15)
        p.font.bold = True
        p.font.color.rgb = COLOR_MUTED
        p.alignment = PP_ALIGN.CENTER
        
        p_title = tf.add_paragraph()
        p_title.text = "An Intelligent Healthcare System for Early Diagnosis of Diabetic Retinopathy Using AI and Quantum-Enhanced Deep Learning"
        p_title.font.size = Pt(20)
        p_title.font.bold = True
        p_title.font.color.rgb = RGBColor(2, 132, 199)
        p_title.alignment = PP_ALIGN.CENTER
        
        p_pres = tf.add_paragraph()
        p_pres.text = "\nPRESENTED BY"
        p_pres.font.size = Pt(13)
        p_pres.font.bold = True
        p_pres.font.color.rgb = COLOR_NAVY
        p_pres.alignment = PP_ALIGN.CENTER
        
        students = [
            ("Sandeep Kumar", "1MV23CS139"),
            ("Saisaket Dhannure", "1MV23CS134"),
            ("Lalith Kumar K", "1MV23CS078"),
            ("E B Nandeesha", "1MV24CS405"),
        ]
        for name, usn in students:
            p_s = tf.add_paragraph()
            p_s.text = f"{name:<35} {usn:>20}"
            p_s.font.size = Pt(12)
            p_s.font.color.rgb = COLOR_DARK_TEXT
            p_s.alignment = PP_ALIGN.CENTER
            
        p_guid = tf.add_paragraph()
        p_guid.text = "\nUNDER THE GUIDANCE OF"
        p_guid.font.size = Pt(13)
        p_guid.font.bold = True
        p_guid.font.color.rgb = COLOR_NAVY
        p_guid.alignment = PP_ALIGN.CENTER
        
        p_names = tf.add_paragraph()
        p_names.text = "Dr. T N Anitha                     Dr. T N Anitha                  Dr. T N Anitha"
        p_names.font.size = Pt(12)
        p_names.font.bold = True
        p_names.font.color.rgb = RGBColor(2, 132, 199)
        p_names.alignment = PP_ALIGN.CENTER
        
        p_des = tf.add_paragraph()
        p_des.text = "Professor & HOD (Guide)           Project Co-ordinator            Professor & HOD"
        p_des.font.size = Pt(11)
        p_des.font.color.rgb = COLOR_MUTED
        p_des.alignment = PP_ALIGN.CENTER
        
        p_dept = tf.add_paragraph()
        p_dept.text = "Dept. of CSE, Sir MVIT            Dept. of CSE, Sir MVIT          Dept. of CSE, Sir MVIT"
        p_dept.font.size = Pt(11)
        p_dept.font.color.rgb = COLOR_MUTED
        p_dept.alignment = PP_ALIGN.CENTER

# Helper function to clear and set top summary text
def set_top_summary(slide, title_text, bullets):
    if slide.shapes.title:
        slide.shapes.title.text = title_text
        slide.shapes.title.text_frame.paragraphs[0].font.size = Pt(22)
        slide.shapes.title.text_frame.paragraphs[0].font.bold = True
        slide.shapes.title.text_frame.paragraphs[0].font.color.rgb = RGBColor(15, 23, 42)

    for shape in slide.shapes:
        if shape.is_placeholder and shape.placeholder_format.type != 1:
            shape.left = Inches(0.8)
            shape.top = Inches(1.1)
            shape.width = Inches(11.7)
            shape.height = Inches(1.4)
            tf = shape.text_frame
            tf.clear()
            for idx, (title_b, text_b) in enumerate(bullets):
                p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
                p.text = f"{title_b}: " if title_b else ""
                p.font.size = Pt(11.5)
                p.font.bold = True
                p.font.color.rgb = RGBColor(2, 132, 199)
                run = p.add_run()
                run.text = text_b
                run.font.bold = False
                run.font.color.rgb = COLOR_DARK_TEXT

# ==============================================================================
# Slide 2: AGENDA
# ==============================================================================
slide2 = prs.slides[1]
if slide2.shapes.title:
    slide2.shapes.title.text = "PRESENTATION AGENDA"
for shape in slide2.shapes:
    if shape.is_placeholder and shape.placeholder_format.type != 1:
        shape.left = Inches(0.8)
        shape.top = Inches(1.3)
        shape.width = Inches(11.7)
        shape.height = Inches(5.6)
        tf = shape.text_frame
        tf.clear()
        
        agenda = [
            ("I. Implementation", "End-to-End Pipeline Architecture, Quality Gate, EfficientNet-B0 Backbone, 4-Qubit VQC & Web Platform"),
            ("II. Testing", "36/36 Automated Pytest Suites, Patient Safety Invariant Gate, and Cross-Dataset Validation (ROC, DRIVE, RFMiD 2.0)"),
            ("III. Performance Analysis", "Controlled Architecture Ablation (QWK = 0.650), RFMiD Holdout (100% Sensitivity, 81.2% Specificity), and Latency"),
            ("IV. Publication / Conference", "Research Paper Details, Author Contributions, IEEE/Scopus Target Venue, and 3 Core Published Novelties"),
            ("V. Outcomes Mapped to SDG", "Direct Alignment with UN SDG 3 (Good Health), SDG 9 (Health Innovation), and SDG 10 (Reduced Inequalities)"),
            ("VI. Screenshots & Demonstration", "Interactive Tele-Ophthalmology Dashboard, Saliency Explorer with Opacity Slider, and Downloadable Medical Report"),
            ("VII. Summary & References", "Consolidated Project Deliverables, Translational Clinical Impact, Future Roadmap, and Academic Citations"),
        ]
        for idx, (head, desc) in enumerate(agenda):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"{head}  ➔  "
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = RGBColor(2, 132, 199)
            run = p.add_run()
            run.text = desc
            run.font.bold = False
            run.font.size = Pt(12)
            run.font.color.rgb = COLOR_DARK_TEXT
            p.space_after = Pt(8)

# ==============================================================================
# Slide 3: Implementation with Pipeline Flowchart
# ==============================================================================
slide3 = prs.slides[2]
slide3_bullets = [
    ("Pipeline Overview", "Clinical tele-screening system combining Quality Control, Frozen Feature Extraction, a 4-Qubit VQC, and Explainable AI."),
    ("Core Invariant", "The Quality Gate rejects degraded/blurred images as UNGRA_RECAPTURE; an ungradable image can NEVER output a reassuring 'No DR'."),
]
set_top_summary(slide3, "System Implementation & Pipeline Architecture", slide3_bullets)
if Path(FIG_PIPELINE).exists():
    slide3.shapes.add_picture(FIG_PIPELINE, left=Inches(0.8), top=Inches(2.5), width=Inches(11.7), height=Inches(4.6))
    print("Embedded Pipeline Flowchart on Slide 3.")

# ==============================================================================
# Slide 4: Testing with Verification Architecture Diagram
# ==============================================================================
slide4 = prs.slides[3]
slide4_bullets = [
    ("Rigorous Verification", "Multi-tier testing strategy combining automated unit/integration tests, invariant safety gates, and multi-disease benchmarks."),
    ("Safety Compliance", "100.0% compliance on quality gate safety invariant: zero degraded images misclassified as healthy across all evaluations."),
]
set_top_summary(slide4, "Testing, Validation & Safety Invariant Framework", slide4_bullets)
if Path(FIG_TESTING).exists():
    slide4.shapes.add_picture(FIG_TESTING, left=Inches(0.8), top=Inches(2.5), width=Inches(11.7), height=Inches(4.6))
    print("Embedded Testing Diagram on Slide 4.")

# ==============================================================================
# Slide 5: Performance Analysis with Comparison Charts
# ==============================================================================
slide5 = prs.slides[4]
slide5_bullets = [
    ("Controlled Head Ablation", "Fair evaluation on identical frozen features (Classical QWK = 0.650 vs Matched Control QWK = 0.628 vs Quantum VQC QWK = 0.638)."),
    ("Diagnostic Efficacy", "Referable DR Sensitivity: 100.0% (target >= 85%). Normal Specificity: 81.2% (target >= 70%). Inference turnaround < 1.2s per image."),
]
set_top_summary(slide5, "Performance Analysis: Benchmarks & Controlled Ablation", slide5_bullets)
if Path(FIG_PERF).exists():
    slide5.shapes.add_picture(FIG_PERF, left=Inches(0.8), top=Inches(2.5), width=Inches(11.7), height=Inches(4.6))
    print("Embedded Performance Charts on Slide 5.")

# ==============================================================================
# Slide 6: Publication / Conference
# ==============================================================================
slide6 = prs.slides[5]
if slide6.shapes.title:
    slide6.shapes.title.text = "Research Publication & Conference Dissemination"
for shape in slide6.shapes:
    if shape.is_placeholder and shape.placeholder_format.type != 1:
        shape.left = Inches(0.8)
        shape.top = Inches(1.3)
        shape.width = Inches(11.7)
        shape.height = Inches(5.6)
        tf = shape.text_frame
        tf.clear()
        
        pub_info = [
            ("Paper Title", "An Intelligent Healthcare System for Early Diagnosis of Diabetic Retinopathy Using AI and Quantum-Enhanced Deep Learning"),
            ("Authors", "1st Dr. T N Anitha (Professor & HOD, Guide), 2nd Sandeep Kumar (1MV23CS139), 3rd Saisaket Dhannure (1MV23CS134), 4th Lalith Kumar K (1MV23CS078), 5th E B Nandeesha (1MV24CS405)."),
            ("Affiliation", "Department of Computer Science & Engineering, Sir M. Visvesvaraya Institute of Technology, Bengaluru, India."),
            ("Target Indexing & Forum", "IEEE / Springer / Scopus-Indexed International Conference on Healthcare AI, Telemedicine & Quantum Computing."),
            ("Core Innovation 1", "Automated Quality Gate Safety Invariant: Guarantees zero false-reassuring discharges on ungradable photographs."),
            ("Core Innovation 2", "Controlled Hybrid Quantum Ablation: Isolates genuine quantum expressivity against matched classical capacity baselines."),
            ("Core Innovation 3", "Multi-Axis Ocular Health Analytics: Integrates Grad-CAM XAI with quantitative risk scoring for Vasculature, DME, CDR, and Lens clarity."),
        ]
        for idx, (label, val) in enumerate(pub_info):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"•  {label}: "
            p.font.size = Pt(13)
            p.font.bold = True
            p.font.color.rgb = RGBColor(2, 132, 199) if "Core" not in label else RGBColor(16, 185, 129)
            run = p.add_run()
            run.text = val
            run.font.bold = False
            run.font.size = Pt(12)
            run.font.color.rgb = COLOR_DARK_TEXT
            p.space_after = Pt(6)

# ==============================================================================
# Slide 7: Outcomes Mapped to SDG with Visual Infographic
# ==============================================================================
slide7 = prs.slides[6]
slide7_bullets = [
    ("Global Health Alignment", "Directly targets United Nations Sustainable Development Goals by closing the screening gap in preventable diabetic blindness."),
    ("Tele-Ophthalmology Reach", "Enables point-of-care risk stratification in rural clinics, eliminating costly travel and specialist referral bottlenecks."),
]
set_top_summary(slide7, "Outcomes Mapped to UN Sustainable Development Goals (SDG)", slide7_bullets)
if Path(FIG_SDG).exists():
    slide7.shapes.add_picture(FIG_SDG, left=Inches(0.8), top=Inches(2.5), width=Inches(11.7), height=Inches(4.6))
    print("Embedded SDG Infographic on Slide 7.")

# ==============================================================================
# Slide 8: Screenshots (High-Res Composite Gallery)
# ==============================================================================
slide8 = prs.slides[7]
if slide8.shapes.title:
    slide8.shapes.title.text = "System Demonstration: Tele-Ophthalmology Platform & Clinical Report"
for shape in list(slide8.shapes):
    if shape.is_placeholder and shape.placeholder_format.type != 1:
        shape.text_frame.clear()
        shape.left = Inches(0.8)
        shape.top = Inches(1.1)
        shape.width = Inches(11.7)
        shape.height = Inches(0.4)
        shape.text_frame.paragraphs[0].text = "1. Quality Gate Screening  |  2. Grad-CAM Saliency Explorer  |  3. Multi-Axis Eye Health  |  4. Hospital-Grade Medical PDF"
        shape.text_frame.paragraphs[0].font.size = Pt(12)
        shape.text_frame.paragraphs[0].font.bold = True
        shape.text_frame.paragraphs[0].font.color.rgb = RGBColor(2, 132, 199)

if Path(FIG_SCREENSHOTS).exists():
    slide8.shapes.add_picture(FIG_SCREENSHOTS, left=Inches(0.8), top=Inches(1.6), width=Inches(11.7), height=Inches(5.4))
    print("Embedded High-Res Screenshot Gallery on Slide 8.")

# ==============================================================================
# Slide 9: SUMMARY
# ==============================================================================
slide9 = prs.slides[8]
if slide9.shapes.title:
    slide9.shapes.title.text = "PROJECT SUMMARY & FUTURE ROADMAP"
for shape in slide9.shapes:
    if shape.is_placeholder and shape.placeholder_format.type != 1:
        shape.left = Inches(0.8)
        shape.top = Inches(1.3)
        shape.width = Inches(11.7)
        shape.height = Inches(5.6)
        tf = shape.text_frame
        tf.clear()
        
        summary = [
            ("Consolidated Contributions", "Successfully translated research paper methodology into an inspectable, operational tele-ophthalmology screening platform."),
            ("Safety-First Engineering", "Enforced automated Quality Gate invariant: ungradable/blurred images strictly trigger mandatory recapture, eliminating false-reassuring negative discharges."),
            ("Hybrid Quantum Validation", "Conducted rigorous multi-seed ablation comparing 4-qubit VQC (12 quantum params) against matched classical dense baselines on identical frozen features."),
            ("Multi-Disease & Multi-Axis Analytics", "Demonstrated 100% referable DR sensitivity and 81.2% normal specificity on RFMiD holdout, augmented with quantitative scoring for Vasculature, DME, CDR, and Lens clarity."),
            ("Enterprise Tele-Medicine Platform", "Implemented web application with AES-GCM encryption at rest, 4 visual themes, 1-click EHR progress note export, and downloadable ISO 13485 medical PDF reports."),
            ("Future Roadmap - Hardware Execution", "Port variational quantum circuits to physical Quantum Processing Units (QPUs) via IBM Quantum / Amazon Braket quantum cloud infrastructure."),
            ("Future Roadmap - Clinical Trials", "Initiate multi-centre prospective observational studies in community health centers under licensed ophthalmologist supervision."),
        ]
        for idx, (head, body) in enumerate(summary):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"✔  {head}: "
            p.font.size = Pt(12.5)
            p.font.bold = True
            p.font.color.rgb = RGBColor(16, 185, 129) if "Roadmap" not in head else RGBColor(2, 132, 199)
            run = p.add_run()
            run.text = body
            run.font.bold = False
            run.font.size = Pt(11.5)
            run.font.color.rgb = COLOR_DARK_TEXT
            p.space_after = Pt(5)

# ==============================================================================
# Slide 10 & 11: References
# ==============================================================================
slide10 = prs.slides[9]
ref1 = [
    ("[1]", "Gulshan et al., 'Development and Validation of a Deep Learning Algorithm for Detection of DR in Retinal Photographs,' JAMA, vol. 316, pp. 2402-2410, 2016."),
    ("[2]", "Ting et al., 'Development and Validation of a Deep Learning System for DR and Related Eye Diseases in Multiethnic Populations,' JAMA, vol. 318, pp. 2211-2223, 2017."),
    ("[3]", "Abramoff et al., 'Pivotal Trial of an Autonomous AI-Based Diagnostic System for Detection of DR in Primary Care Offices,' npj Digital Medicine, vol. 1, art. 39, 2018."),
    ("[4]", "Alyoubi et al., 'Diabetic Retinopathy Detection Through Deep Learning Techniques: A Review,' Informatics in Medicine Unlocked, vol. 20, 100377, 2020."),
    ("[5]", "Pratt et al., 'Convolutional Neural Networks for Diabetic Retinopathy,' Procedia Computer Science, vol. 90, pp. 200-205, 2016."),
    ("[6]", "Romero-Oraa et al., 'Attention-Based Deep Learning Framework for Automatic Fundus Image Processing to Aid in DR Grading,' CMPB, vol. 249, 108160, 2024."),
    ("[7]", "Li et al., 'CANet: Cross-Disease Attention Network for Joint DR and Diabetic Macular Edema Grading,' IEEE TMI, vol. 39, pp. 1483-1493, 2020."),
    ("[8]", "Porwal et al., 'Indian Diabetic Retinopathy Image Dataset (IDRiD): A Database for Screening Research,' Data, vol. 3, 2018."),
    ("[9]", "Project Synopsis, 'An Intelligent Healthcare System for Early Diagnosis of DR Using AI,' Dept. of CSE, Sir MVIT, VTU Belagavi, 2026."),
    ("[10]", "Mir et al., 'Diabetic Retinopathy Detection Using Classical-Quantum Transfer Learning Approach,' CMC, vol. 71, pp. 3733-3746, 2022."),
    ("[11]", "Jain et al., 'Diabetic Retinopathy Detection Using Quantum Transfer Learning,' arXiv:2405.01734, 2024."),
]
if slide10.shapes.title:
    slide10.shapes.title.text = "Academic References (Part 1)"
for shape in slide10.shapes:
    if shape.is_placeholder and shape.placeholder_format.type != 1:
        shape.left = Inches(0.8)
        shape.top = Inches(1.3)
        shape.width = Inches(11.7)
        shape.height = Inches(5.6)
        tf = shape.text_frame
        tf.clear()
        for idx, (cite, text) in enumerate(ref1):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"{cite}  "
            p.font.size = Pt(10.5)
            p.font.bold = True
            p.font.color.rgb = RGBColor(2, 132, 199)
            run = p.add_run()
            run.text = text
            run.font.bold = False
            run.font.size = Pt(10)
            run.font.color.rgb = COLOR_DARK_TEXT
            p.space_after = Pt(3)

slide11 = prs.slides[10]
ref2 = [
    ("[12]", "Bali et al., 'QuantumNet: An Enhanced Diabetic Retinopathy Detection Model Using Classical DL-QTL,' MethodsX, vol. 14, 103185, 2025."),
    ("[13]", "Ikram & Imran, 'Fundus Imaging Based Quantum Computing: Grading of Vision Threatening Diabetic Retinopathy,' Informatics in Med. Unlocked, vol. 61, 2026."),
    ("[14]", "Chilukuri et al., 'Quantum Denoising Autoencoder Improves Retinal Fundus Image Quality for Early DR Screening,' Scientific Reports, vol. 16, art. 5970, 2026."),
    ("[15]", "Menaka et al., 'A Novel Diabetic Retinopathy Detection From Fundus Images Using Hybrid QCNN Models,' Scientific Reports, vol. 16, art. 20878, 2026."),
    ("[16]", "Mari et al., 'Transfer Learning in Hybrid Classical-Quantum Neural Networks,' Quantum, vol. 4, p. 340, 2020."),
    ("[17]", "Sim et al., 'Expressibility and Entangling Capability of Parameterized Quantum Circuits,' Advanced Quantum Technologies, vol. 2, 1900070, 2019."),
    ("[18]", "Hubregtsen et al., 'Evaluation of Parameterized Quantum Circuits: Accuracy, Expressibility, and Entangling Capability,' QMI, vol. 3, 2021."),
    ("[19]", "Schuld et al., 'Evaluating Analytic Gradients on Quantum Hardware,' Physical Review A, vol. 99, 032331, 2019."),
    ("[20]", "Bergholm et al., 'PennyLane: Automatic Differentiation of Hybrid Quantum-Classical Computations,' arXiv:1811.04968, 2018."),
    ("[21]", "Preskill, 'Quantum Computing in the NISQ Era and Beyond,' Quantum, vol. 2, p. 79, 2018."),
    ("[22]", "McClean et al., 'Barren Plateaus in Quantum Neural Network Training Landscapes,' Nature Communications, vol. 9, art. 4812, 2018."),
]
if slide11.shapes.title:
    slide11.shapes.title.text = "Academic References (Part 2)"
for shape in slide11.shapes:
    if shape.is_placeholder and shape.placeholder_format.type != 1:
        shape.left = Inches(0.8)
        shape.top = Inches(1.3)
        shape.width = Inches(11.7)
        shape.height = Inches(5.6)
        tf = shape.text_frame
        tf.clear()
        for idx, (cite, text) in enumerate(ref2):
            p = tf.paragraphs[0] if idx == 0 else tf.add_paragraph()
            p.text = f"{cite}  "
            p.font.size = Pt(10.5)
            p.font.bold = True
            p.font.color.rgb = RGBColor(2, 132, 199)
            run = p.add_run()
            run.text = text
            run.font.bold = False
            run.font.size = Pt(10)
            run.font.color.rgb = COLOR_DARK_TEXT
            p.space_after = Pt(3)

# Save presentation to both paths
prs.save(OUT_REPORTS)
prs.save(OUT_DOWNLOADS)
print(f"\n=======================================================")
print(f"Successfully generated visual & colorful presentation to:")
print(f" 1. {OUT_DOWNLOADS}")
print(f" 2. {OUT_REPORTS}")
print(f"=======================================================\n")
