"""
auth.py — Login & Register Routes for OccupAI
===============================================
Endpoints:
  POST /auth/register  — create driver account
  POST /auth/login     — login, returns JWT
  GET  /auth/me        — current user + role profile
  PUT  /auth/profile   — update name/password
  POST /auth/logout    — logout

Note: Uses RealDictCursor — rows accessed as dicts (row['email'])
Admin/Owner accounts created manually in pgAdmin.
"""

from flask import Blueprint, request, jsonify
from flask_jwt_extended import (
    create_access_token, create_refresh_token,
    jwt_required, get_jwt_identity, get_jwt
)
import bcrypt
from datetime import datetime
from database import get_db

auth_bp = Blueprint("auth", __name__, url_prefix="/auth")


# ╔══════════════════════════════════════════════╗
# ║              REGISTER                       ║
# ╚══════════════════════════════════════════════╝
@auth_bp.route("/register", methods=["POST"])
def register():
    """
    POST /auth/register
    Body: { first_name, last_name, email, password }
    Role is always 'driver' — cannot be changed via API.
    """
    data = request.get_json()

    # Validate required fields
    for field in ["first_name", "last_name", "email", "password"]:
        if not data.get(field):
            return jsonify({
                "success": False,
                "message": f"'{field}' is required."
            }), 400

    first_name = data["first_name"].strip()
    last_name  = data["last_name"].strip()
    email      = data["email"].strip().lower()
    password   = data["password"]
    role       = "driver"  # always — never from client

    if "@" not in email or "." not in email:
        return jsonify({"success": False, "message": "Invalid email format."}), 400
    if len(password) < 6:
        return jsonify({"success": False, "message": "Password must be at least 6 characters."}), 400
    if len(first_name) < 2:
        return jsonify({"success": False, "message": "First name must be at least 2 characters."}), 400
    if len(last_name) < 2:
        return jsonify({"success": False, "message": "Last name must be at least 2 characters."}), 400

    password_hash = bcrypt.hashpw(
        password.encode("utf-8"), bcrypt.gensalt()
    ).decode("utf-8")

    try:
        conn = get_db()
        cur  = conn.cursor()

        # Check duplicate email
        cur.execute("SELECT user_id FROM users WHERE email = %s", (email,))
        if cur.fetchone():
            cur.close(); conn.close()
            return jsonify({
                "success": False,
                "message": "Email already registered. Please login."
            }), 409

        # Insert user
        cur.execute("""
            INSERT INTO users (first_name, last_name, email, password_hash, role)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING user_id, first_name, last_name, full_name, email, role, created_at
        """, (first_name, last_name, email, password_hash, role))

        new_user    = cur.fetchone()
        new_user_id = new_user["user_id"]

        # Auto-create drivers child row
        cur.execute("INSERT INTO drivers (user_id) VALUES (%s)", (new_user_id,))

        conn.commit()
        cur.close(); conn.close()

        access_token  = create_access_token(
            identity=str(new_user_id),
            additional_claims={"role": new_user["role"], "email": new_user["email"]}
        )
        refresh_token = create_refresh_token(identity=str(new_user_id))

        return jsonify({
            "success":       True,
            "message":       "Account created successfully!",
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "user": {
                "user_id":    new_user["user_id"],
                "first_name": new_user["first_name"],
                "last_name":  new_user["last_name"],
                "full_name":  new_user["full_name"],
                "email":      new_user["email"],
                "role":       new_user["role"],
                "created_at": str(new_user["created_at"]),
            }
        }), 201

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Registration failed: {str(e)}"
        }), 500


# ╔══════════════════════════════════════════════╗
# ║                 LOGIN                       ║
# ╚══════════════════════════════════════════════╝
@auth_bp.route("/login", methods=["POST"])
def login():
    """
    POST /auth/login
    Body: { email, password }
    Works for all roles. Role embedded in JWT from DB.
    """
    data     = request.get_json()
    email    = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password are required."
        }), 400

    try:
        conn = get_db()
        cur  = conn.cursor()

        cur.execute("""
            SELECT user_id, first_name, last_name, full_name,
                   email, password_hash, role, is_active
            FROM users WHERE email = %s
        """, (email,))
        user = cur.fetchone()

        if not user:
            cur.close(); conn.close()
            return jsonify({
                "success": False,
                "message": "Email not found. Please register first."
            }), 401

        if not user["is_active"]:
            cur.close(); conn.close()
            return jsonify({
                "success": False,
                "message": "Account is disabled. Contact administrator."
            }), 403

        if not bcrypt.checkpw(
            password.encode("utf-8"),
            user["password_hash"].encode("utf-8")
        ):
            cur.close(); conn.close()
            return jsonify({
                "success": False,
                "message": "Incorrect password. Please try again."
            }), 401

        cur.execute(
            "UPDATE users SET last_login = %s WHERE user_id = %s",
            (datetime.utcnow(), user["user_id"])
        )
        conn.commit()
        cur.close(); conn.close()

        access_token  = create_access_token(
            identity=str(user["user_id"]),
            additional_claims={"role": user["role"], "email": user["email"]}
        )
        refresh_token = create_refresh_token(identity=str(user["user_id"]))

        return jsonify({
            "success":       True,
            "message":       f"Welcome back, {user['first_name']}!",
            "access_token":  access_token,
            "refresh_token": refresh_token,
            "user": {
                "user_id":    user["user_id"],
                "first_name": user["first_name"],
                "last_name":  user["last_name"],
                "full_name":  user["full_name"],
                "email":      user["email"],
                "role":       user["role"],
            }
        }), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": f"Login failed: {str(e)}"
        }), 500


# ╔══════════════════════════════════════════════╗
# ║            GET CURRENT USER                 ║
# ╚══════════════════════════════════════════════╝
@auth_bp.route("/me", methods=["GET"])
@jwt_required()
def get_me():
    """
    GET /auth/me
    Header: Authorization: Bearer <token>
    Returns user + role-specific child data.
    """
    user_id = get_jwt_identity()

    try:
        conn = get_db()
        cur  = conn.cursor()

        cur.execute("""
            SELECT user_id, first_name, last_name, full_name,
                   email, role, is_active, created_at, last_login
            FROM users WHERE user_id = %s
        """, (user_id,))
        user = cur.fetchone()

        if not user:
            cur.close(); conn.close()
            return jsonify({"success": False, "message": "User not found."}), 404

        role_data = {}
        if user["role"] == "driver":
            cur.execute("""
                SELECT driver_id, license_number, vehicle_plate, vehicle_type
                FROM drivers WHERE user_id = %s
            """, (user["user_id"],))
            row = cur.fetchone()
            if row: role_data = dict(row)

        elif user["role"] == "admin":
            cur.execute("""
                SELECT admin_id, department, access_level
                FROM admins WHERE user_id = %s
            """, (user["user_id"],))
            row = cur.fetchone()
            if row: role_data = dict(row)

        elif user["role"] == "owner":
            cur.execute("""
                SELECT owner_id, business_name, lot_address
                FROM owners WHERE user_id = %s
            """, (user["user_id"],))
            row = cur.fetchone()
            if row: role_data = dict(row)

        cur.close(); conn.close()

        user_dict = dict(user)
        user_dict["created_at"] = str(user["created_at"])
        user_dict["last_login"] = str(user["last_login"]) if user["last_login"] else None

        return jsonify({
            "success": True,
            "user": {**user_dict, **role_data}
        }), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ╔══════════════════════════════════════════════╗
# ║           UPDATE PROFILE                    ║
# ╚══════════════════════════════════════════════╝
@auth_bp.route("/profile", methods=["PUT"])
@jwt_required()
def update_profile():
    """
    PUT /auth/profile
    Body: { first_name, last_name, password? }
    Role cannot be changed here.
    """
    user_id    = get_jwt_identity()
    data       = request.get_json()
    first_name = data.get("first_name", "").strip()
    last_name  = data.get("last_name",  "").strip()
    password   = data.get("password",   "")

    if not first_name or not last_name:
        return jsonify({
            "success": False,
            "message": "first_name and last_name are required."
        }), 400

    try:
        conn = get_db()
        cur  = conn.cursor()

        if password:
            if len(password) < 6:
                return jsonify({
                    "success": False,
                    "message": "Password must be at least 6 characters."
                }), 400
            password_hash = bcrypt.hashpw(
                password.encode("utf-8"), bcrypt.gensalt()
            ).decode("utf-8")
            cur.execute("""
                UPDATE users
                SET first_name=%s, last_name=%s,
                    password_hash=%s, updated_at=%s
                WHERE user_id=%s
            """, (first_name, last_name, password_hash, datetime.utcnow(), user_id))
        else:
            cur.execute("""
                UPDATE users
                SET first_name=%s, last_name=%s, updated_at=%s
                WHERE user_id=%s
            """, (first_name, last_name, datetime.utcnow(), user_id))

        conn.commit()
        cur.close(); conn.close()
        return jsonify({"success": True, "message": "Profile updated successfully!"}), 200

    except Exception as e:
        return jsonify({"success": False, "message": str(e)}), 500


# ╔══════════════════════════════════════════════╗
# ║              LOGOUT                         ║
# ╚══════════════════════════════════════════════╝
@auth_bp.route("/logout", methods=["POST"])
@jwt_required()
def logout():
    return jsonify({"success": True, "message": "Logged out successfully."}), 200


# ╔══════════════════════════════════════════════╗
# ║         ROLE-BASED ACCESS DECORATOR         ║
# ╚══════════════════════════════════════════════╝
def role_required(*roles):
    """
    Usage:
        @role_required("admin", "owner")
        @role_required("owner")
    """
    from functools import wraps
    def decorator(fn):
        @wraps(fn)
        @jwt_required()
        def wrapper(*args, **kwargs):
            claims    = get_jwt()
            user_role = claims.get("role", "")
            if user_role not in roles:
                return jsonify({
                    "success": False,
                    "message": f"Access denied. Required role: {', '.join(roles)}"
                }), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator