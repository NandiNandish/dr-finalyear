"""
Unit tests for Flask Backend Web Application, Authentication, and Encrypted Storage.
"""

import pytest
import io
from pathlib import Path
from app.app import app, storage
from data.synthetic_generator import generate_synthetic_fundus
import cv2


@pytest.fixture
def client():
    app.config["TESTING"] = True
    app.config["WTF_CSRF_ENABLED"] = False
    with app.test_client() as client:
        yield client


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "HEALTHY"
    assert "PennyLane" in data["quantum_backend"]
    assert data["qubits"] == 4
    assert "AES-GCM" in data["storage_encryption"]


def test_unauthenticated_access_redirects(client):
    response = client.get("/")
    assert response.status_code == 302
    assert "/login" in response.headers["Location"]


def test_login_success_and_logout(client):
    # Test valid clinician credentials
    response = client.post("/login", data={"username": "clinician", "password": "doctor123"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Dr. Sarah Rao" in response.data

    # Test logout
    response_logout = client.get("/logout", follow_redirects=True)
    assert response_logout.status_code == 200
    assert b"Sign In" in response_logout.data


def test_login_invalid_credentials(client):
    response = client.post("/login", data={"username": "clinician", "password": "wrongpassword"}, follow_redirects=True)
    assert response.status_code == 200
    assert b"Invalid credentials" in response.data


def test_predict_endpoint_and_encrypted_storage(client):
    # Authenticate first
    client.post("/login", data={"username": "clinician", "password": "doctor123"})

    # Create dummy fundus image bytes
    img = generate_synthetic_fundus(grade=0, image_size=224)
    _, buf = cv2.imencode(".png", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    img_bytes = io.BytesIO(buf.tobytes())

    # Send POST request to /predict with Accept: application/json
    response = client.post(
        "/predict",
        data={
            "file": (img_bytes, "test_fundus.png"),
            "head": "matched_dense",
        },
        headers={"Accept": "application/json"},
        content_type="multipart/form-data",
    )

    assert response.status_code == 200
    data = response.get_json()

    assert "is_gradable" in data
    assert "recommendation" in data
    assert "disclaimer" in data
    assert "encrypted_file_ref" in data

    # Verify that the file stored on disk is encrypted (ciphertext, not valid PNG header)
    enc_ref = data["encrypted_file_ref"]
    enc_path = Path("app/uploads_encrypted") / enc_ref
    assert enc_path.exists()

    with open(enc_path, "rb") as f:
        file_bytes = f.read()

    # PNG magic bytes: \x89PNG\r\n\x1a\n
    assert not file_bytes.startswith(b"\x89PNG"), "Uploaded file was stored in plaintext! Encryption at rest failed."

    # Verify decrypt in-memory works
    decrypted = storage.read_decrypted(enc_ref)
    assert decrypted.startswith(b"\x89PNG"), "Decryption did not yield original PNG file."
