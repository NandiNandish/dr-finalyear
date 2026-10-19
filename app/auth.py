"""
Role-Based Access Control and Authentication.
Supported Roles:
- Screening Technician ('technician'): Can upload images, assess imageability, and request screening.
- Clinician ('clinician'): Full diagnostic screening access, Grad-CAM inspection, referral triage.
- Administrator ('admin'): Full system access, auditing, and configuration.
"""

from functools import wraps
from typing import Dict, Any, Optional
from flask import session, redirect, url_for, flash, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash


# Pre-configured secure role-based accounts (passwords hashed via PBKDF2/SHA256)
USERS_DB: Dict[str, Dict[str, Any]] = {
    "clinician": {
        "username": "clinician",
        "name": "Dr. Sarah Rao, MD (Ophthalmologist)",
        "password_hash": generate_password_hash("doctor123"),
        "role": "clinician",
    },
    "technician": {
        "username": "technician",
        "name": "Alex Chen (Retinal Screening Tech)",
        "password_hash": generate_password_hash("tech123"),
        "role": "technician",
    },
    "admin": {
        "username": "admin",
        "name": "System Administrator",
        "password_hash": generate_password_hash("admin123"),
        "role": "admin",
    },
}


def authenticate_user(username: str, password: str) -> Optional[Dict[str, Any]]:
    user = USERS_DB.get(username.lower().strip())
    if user and check_password_hash(user["password_hash"], password):
        return {
            "username": user["username"],
            "name": user["name"],
            "role": user["role"],
        }
    return None


def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user" not in session:
            if request.is_json or request.path.startswith("/predict"):
                return jsonify({"error": "Unauthorized. Please log in first."}), 401
            return redirect(url_for("login", next=request.url))
        return f(*args, **kwargs)
    return decorated_function


def roles_required(*allowed_roles):
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if "user" not in session:
                return redirect(url_for("login"))
            user_role = session["user"].get("role")
            if user_role not in allowed_roles:
                flash(f"Access denied. Requires role: {', '.join(allowed_roles)}.", "error")
                return redirect(url_for("index"))
            return f(*args, **kwargs)
        return decorated_function
    return decorator
