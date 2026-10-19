"""
Flask Backend Application for Diabetic Retinopathy Screening System.

Features:
- Role-based authentication (Technician, Clinician, Administrator).
- Encrypted storage at rest for uploaded patient fundus images (AES-GCM).
- REST endpoint /predict for automated screening, Grad-CAM, and triage.
- Web UI displaying: uploaded fundus photo, gradability verdict, 5-level DR grade,
  confidence, Grad-CAM attention overlay, clinical recommendation, and disclaimer.
"""

import os
import sys
from pathlib import Path
import time
import uuid
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    session,
    redirect,
    url_for,
    flash,
)

# Ensure project root directory is in Python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from auth import authenticate_user, login_required
    from crypto import EncryptedStorage
except (ModuleNotFoundError, ImportError):
    from app.auth import authenticate_user, login_required
    from app.crypto import EncryptedStorage

from src.inference.pipeline import ScreeningPipeline, ScreeningReport
from src.data.label_maps import CLASS_NAMES


app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "quantum_dr_screening_secret_key_2026")

# Encrypted storage at rest
storage = EncryptedStorage(
    storage_dir="app/uploads_encrypted",
    key_path="app/storage_key.key",
)

# Check for trained checkpoint
def get_best_available_checkpoint(head: str = "hybrid_quantum") -> str:
    possible_ckpts = [
        f"checkpoints/ablation/checkpoint_{head}_seed42.pt",
        f"checkpoints/checkpoint_{head}_seed42.pt",
        f"checkpoints/ablation/checkpoint_matched_dense_seed42.pt",
        f"checkpoints/best_model.pt",
    ]
    for p in possible_ckpts:
        if Path(p).exists():
            return p
    return None


# Lazy cache of pipelines per head
PIPELINES = {}

def get_pipeline(head: str = "hybrid_quantum") -> ScreeningPipeline:
    head = head.lower().strip()
    if head not in PIPELINES:
        ckpt = get_best_available_checkpoint(head)
        print(f"Initializing screening pipeline for head: '{head}' with checkpoint: '{ckpt}'...")
        PIPELINES[head] = ScreeningPipeline(
            checkpoint_path=ckpt,
            active_head=head,
            device="cpu",
        )
    return PIPELINES[head]


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")
        user = authenticate_user(username, password)
        if user:
            session["user"] = user
            flash(f"Welcome, {user['name']} ({user['role'].title()})", "success")
            next_url = request.args.get("next") or url_for("index")
            return redirect(next_url)
        else:
            flash("Invalid credentials. Try: clinician / doctor123, technician / tech123, or admin / admin123.", "error")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.pop("user", None)
    flash("You have been securely logged out.", "info")
    return redirect(url_for("login"))


@app.route("/")
@login_required
def index():
    user = session.get("user", {})
    # Find available demo samples for quick 1-click test
    demo_dir = Path("data/demo")
    sample_images = []
    if demo_dir.exists():
        for p in list(demo_dir.glob("*.png"))[:8]:
            sample_images.append({
                "filename": p.name,
                "label": p.stem.replace("eyepacs_demo_", "").replace("messidor_demo_", ""),
            })

    return render_template("index.html", user=user, sample_images=sample_images)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
    """
    REST & Form endpoint for automated screening.
    Accepts:
    - file: multipart upload
    - head: 'hybrid_quantum' | 'matched_dense' | 'classical'
    """
    t_start = time.time()
    head = request.form.get("head", "hybrid_quantum").lower().strip()

    file = request.files.get("file")
    sample_name = request.form.get("sample_name")

    image_bytes = None
    original_filename = "upload.png"

    if file and file.filename != "":
        original_filename = file.filename
        image_bytes = file.read()
    elif sample_name:
        sample_path = Path("data/demo") / sample_name
        if sample_path.exists():
            original_filename = sample_name
            with open(sample_path, "rb") as f:
                image_bytes = f.read()

    if not image_bytes:
        return jsonify({"error": "No image file provided. Upload a fundus photo."}), 400

    # 1. ENCRYPT AT REST BEFORE DISK WRITE
    uid = uuid.uuid4().hex[:12]
    enc_filename = f"{uid}_{original_filename}"
    storage.save_encrypted(image_bytes, enc_filename)

    # 2. RUN SCREENING PIPELINE
    pipeline = get_pipeline(head=head)
    report: ScreeningReport = pipeline.screen(image_bytes)

    latency_ms = round((time.time() - t_start) * 1000.0, 1)

    response_data = report.to_dict()
    response_data["latency_ms"] = latency_ms
    response_data["active_head"] = head
    response_data["original_image_base64"] = report.original_image_base64
    response_data["gradcam_overlay_base64"] = report.gradcam_overlay_base64
    response_data["encrypted_file_ref"] = enc_filename + ".enc"

    # If requested via JSON API
    if request.is_json or request.headers.get("Accept") == "application/json":
        return jsonify(response_data)

    # Render HTML report template for interactive UI
    return render_template("report.html", report=response_data, user=session.get("user", {}))


@app.route("/health", methods=["GET"])
def health():
    """
    Health and capability check endpoint.
    """
    return jsonify({
        "status": "HEALTHY",
        "system": "Quantum-Enhanced Diabetic Retinopathy Screening Aid",
        "version": "1.0.0",
        "supported_heads": ["hybrid_quantum", "matched_dense", "classical"],
        "quantum_backend": "PennyLane default.qubit exact simulator",
        "qubits": 4,
        "storage_encryption": "AES-GCM (Fernet) active",
        "device_disclaimer": "Screening recommendation aid only, not a diagnostic device.",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
