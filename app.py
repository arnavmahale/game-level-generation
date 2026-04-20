"""
Flask web app for interactive game level generation.

Serves a React frontend and exposes an API for generating levels with
three models — naive baseline, bigram transitions, and a conditional
Conv-VAE — plus a BFS playability repair post-pass.
"""

import sys
import os
import logging
from datetime import timedelta
from functools import wraps

import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from itsdangerous import URLSafeSerializer, BadSignature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Load .env before anything else so TURSO_* and FLASK_SECRET_KEY are available
# when the db and Flask session modules initialize.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from data_utils import get_model_path
from model_naive import NaiveBaseline
from model_classical import BigramModel
from model_deep import (
    ConvVAE, generate_levels as vae_generate,
    three_class_to_api, N_BUCKETS,
)
from repair import repair as repair_level, enforce_layout
from evaluate import (
    js_divergence, tile_distribution, structural_metrics, check_playability,
)
import db as db_mod

# ---------------------------------------------------------------------------
# Model registry — loads all models once at startup
# ---------------------------------------------------------------------------

class ModelRegistry:
    _instance = None

    def __init__(self):
        self.naive = NaiveBaseline()
        self.naive.load()

        self.bigram = BigramModel()
        self.bigram.load()

        self.device = torch.device("cpu")
        self.vae = ConvVAE(latent_dim=64).to(self.device)
        self.vae.load_state_dict(
            torch.load(get_model_path("cvae_best.pth"), map_location=self.device, weights_only=True)
        )
        self.vae.eval()

        # Reference tile distribution (pre-computed from the train split by
        # scripts/build_reference.py). Loading a ~100-byte histogram avoids
        # shipping the 150MB training JSONL with the deployed container.
        self.train_tile_dist = np.load(get_model_path("train_tile_dist.npy"))

    @classmethod
    def get(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance


# ---------------------------------------------------------------------------
# Difficulty mapping
# ---------------------------------------------------------------------------

def get_naive_params(difficulty):
    """Map difficulty 0-100 to height_std (1.0 to 16.0)."""
    height_std = 1.0 + (difficulty / 100.0) * 15.0
    return {"height_std": round(height_std, 2)}


def get_bigram_params(difficulty):
    """Map difficulty 0-100 to prior_weight (0.2-0.8) and trans_weight (0.4-0.1)."""
    t = difficulty / 100.0
    prior_weight = 0.2 + t * 0.6
    trans_weight = 0.4 - t * 0.3
    return {
        "prior_weight": round(prior_weight, 3),
        "trans_weight": round(trans_weight, 3),
    }


def get_vae_params(difficulty):
    """
    Map difficulty 0-100 to a conditioning bucket (0=easy, 1=med, 2=hard).
    The slider is partitioned into N_BUCKETS equal bands.
    """
    bucket = min(N_BUCKETS - 1, (difficulty * N_BUCKETS) // 101)
    labels = {0: "easy", 1: "medium", 2: "hard"}
    return {
        "bucket": int(bucket),
        "bucket_label": labels.get(bucket, str(bucket)),
        "guidance_scale": 2.0,
    }


# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------

app = Flask(__name__, static_folder="frontend/dist", static_url_path="")
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "dev-insecure-key-change-in-prod")
# On HF Spaces the app is served inside an iframe on huggingface.co, so its
# cookies are cross-site. SameSite=Lax cookies are suppressed in that context,
# which makes POSTs (score recording) fail auth even though login worked.
# Detect prod by presence of the $PORT env var the Docker image sets, and
# switch to SameSite=None; Secure so the cookie rides cross-site over HTTPS.
_is_prod_iframe = bool(os.environ.get("PORT"))
# Respect the proxy's X-Forwarded-Proto so Flask knows the request is HTTPS
# even though gunicorn is speaking plain HTTP on the loopback.
if _is_prod_iframe:
    from werkzeug.middleware.proxy_fix import ProxyFix
    app.wsgi_app = ProxyFix(app.wsgi_app, x_proto=1, x_host=1)
app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="None" if _is_prod_iframe else "Lax",
    SESSION_COOKIE_SECURE=_is_prod_iframe,
    PERMANENT_SESSION_LIFETIME=timedelta(days=30),
)
CORS(app, supports_credentials=True)


# HF Spaces embeds the app in a cross-site iframe, where Safari/Chrome block
# third-party cookies for auth. We issue a signed token on login/register and
# accept it via Authorization: Bearer header as a cookie-free fallback.
_token_serializer = URLSafeSerializer(app.config["SECRET_KEY"], salt="auth-token")


def _make_token(user_id: int) -> str:
    return _token_serializer.dumps({"uid": int(user_id)})


def _verify_token(token: str):
    try:
        data = _token_serializer.loads(token)
        return int(data["uid"])
    except (BadSignature, KeyError, ValueError, TypeError):
        return None


def _current_user_id():
    uid = session.get("user_id")
    if uid:
        return uid
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return _verify_token(auth[7:])
    return None


from werkzeug.exceptions import HTTPException
import traceback


@app.errorhandler(HTTPException)
def _http_error(err):
    return jsonify({"error": err.description}), err.code


@app.errorhandler(Exception)
def _unhandled_error(err):
    # Keep errors as JSON so the SPA's res.json() doesn't blow up with
    # Safari's "string did not match the expected pattern" on the
    # default HTML 500 page.
    tb = traceback.format_exc()
    app.logger.error("Unhandled error:\n%s", tb)
    original = getattr(err, "original_exception", None) or err
    return jsonify({"error": f"{type(original).__name__}: {original}"}), 500


def _require_auth(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        uid = _current_user_id()
        if not uid:
            return jsonify({"error": "not authenticated"}), 401
        return fn(uid, *args, **kwargs)
    return wrapper


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    if not username:
        return jsonify({"error": "username is required"}), 400
    if not password:
        return jsonify({"error": "password is required"}), 400
    try:
        if db_mod.get_user_by_username(username) is not None:
            return jsonify({"error": "username already taken"}), 409
        uid = db_mod.create_user(username, password)
    except Exception as e:
        app.logger.exception("register db error")
        return jsonify({"error": f"db error: {type(e).__name__}: {e}"}), 500
    if uid is None:
        return jsonify({"error": "could not create user"}), 500
    session.permanent = True
    session["user_id"] = uid
    return jsonify({
        "user": {"id": uid, "username": username},
        "token": _make_token(uid),
    })


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json() or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""
    user = db_mod.get_user_by_username(username)
    if user is None or not db_mod.verify_password(password, user["password_hash"]):
        return jsonify({"error": "invalid username or password"}), 401
    session.permanent = True
    session["user_id"] = user["id"]
    return jsonify({
        "user": {"id": user["id"], "username": user["username"]},
        "token": _make_token(user["id"]),
    })


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    session.clear()
    return jsonify({"ok": True})


@app.route("/api/auth/me", methods=["GET"])
def auth_me():
    uid = _current_user_id()
    if not uid:
        return jsonify({"user": None})
    # Don't session.clear() on a missed lookup: the embedded Turso replica
    # on this worker thread may simply be stale from another thread's
    # insert. Trust the signed cookie, fall back to a minimal user dict.
    try:
        user = db_mod.get_user_by_id(uid)
    except Exception:
        app.logger.exception("auth_me lookup failed")
        user = None
    if user is None:
        return jsonify({"user": {"id": uid, "username": "you"}})
    return jsonify({"user": user})


@app.route("/api/scores/completion", methods=["POST"])
@_require_auth
def score_completion(uid):
    data = request.get_json() or {}
    model = data.get("model")
    app.logger.info("POST /api/scores/completion uid=%s model=%s", uid, model)
    if model not in ("naive", "bigram", "vae"):
        return jsonify({"error": "invalid model"}), 400
    try:
        db_mod.record_completion(uid, model)
        return jsonify(db_mod.stats_for_user(uid))
    except Exception as e:
        app.logger.exception("score_completion failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/scores/endless", methods=["POST"])
@_require_auth
def score_endless(uid):
    data = request.get_json() or {}
    try:
        score = int(data.get("score", 0))
    except (TypeError, ValueError):
        return jsonify({"error": "invalid score"}), 400
    app.logger.info("POST /api/scores/endless uid=%s score=%s", uid, score)
    if score < 0 or score > 1_000_000:
        return jsonify({"error": "score out of range"}), 400
    try:
        db_mod.record_endless_score(uid, score)
        return jsonify(db_mod.stats_for_user(uid))
    except Exception as e:
        app.logger.exception("score_endless failed")
        return jsonify({"error": f"{type(e).__name__}: {e}"}), 500


@app.route("/api/stats/me", methods=["GET"])
@_require_auth
def stats_me(uid):
    return jsonify(db_mod.stats_for_user(uid))


@app.route("/api/leaderboard/vae", methods=["GET"])
def leaderboard_vae_route():
    return jsonify({"rows": db_mod.leaderboard_vae()})


@app.route("/api/leaderboard/endless", methods=["GET"])
def leaderboard_endless_route():
    return jsonify({"rows": db_mod.leaderboard_endless()})


@app.route("/api/generate", methods=["POST"])
def generate():
    data = request.get_json()
    model_name = data.get("model", "vae")
    difficulty = max(0, min(100, int(data.get("difficulty", 50))))
    seed = data.get("seed")
    if seed is not None:
        seed = int(seed)
    apply_repair = bool(data.get("repair", True))

    registry = ModelRegistry.get()
    params_used = {"model": model_name, "difficulty": difficulty}

    if model_name == "naive":
        p = get_naive_params(difficulty)
        params_used.update(p)
        original_std = registry.naive.height_std
        registry.naive.height_std = p["height_std"]
        levels = registry.naive.generate(n=1, seed=seed)
        registry.naive.height_std = original_std

    elif model_name == "bigram":
        p = get_bigram_params(difficulty)
        params_used.update(p)
        levels = registry.bigram.generate(
            n=1, seed=seed,
            prior_weight=p["prior_weight"],
            trans_weight=p["trans_weight"],
        )

    elif model_name == "vae":
        p = get_vae_params(difficulty)
        params_used.update(p)
        levels_3c = vae_generate(
            registry.vae, n=1, bucket=p["bucket"],
            seed=seed, device=registry.device,
            guidance_scale=p["guidance_scale"],
        )
        levels = np.stack([three_class_to_api(levels_3c[0])])

    else:
        return jsonify({"error": f"Unknown model: {model_name}"}), 400

    # Enforce the clean sky / floor silhouette so every level reads as a
    # 'platformer screen' rather than a field of random tiles.
    levels = np.stack([enforce_layout(levels[0])])

    params_used["repair"] = apply_repair
    if apply_repair:
        levels = np.stack([repair_level(levels[0])])

    # Collapse to 3 gameplay categories: 0=empty, 1=solid, 6=hazard.
    # The in-browser game already treats slope/platform as solid and
    # bonus/water/decoration as air, so we simplify the output to match.
    out = levels[0].copy()
    out[(out == 2) | (out == 3)] = 1
    out[(out == 4) | (out == 5) | (out == 7)] = 0
    levels = out[None, ...]

    # Compute metrics
    js = float(js_divergence(registry.train_tile_dist, tile_distribution(levels)))
    playable = bool(check_playability(levels[0]))
    struct = structural_metrics(levels)

    metrics = {
        "js_divergence": round(js, 4),
        "playable": playable,
        "ground_coverage": round(float(struct["ground_coverage"]), 4),
        "ground_contiguity": round(float(struct["ground_contiguity"]), 4),
        "hazard_density": round(float(struct["hazard_density"]), 4),
    }

    return jsonify({
        "level": levels[0].tolist(),
        "metrics": metrics,
        "params_used": params_used,
    })


@app.route("/api/models", methods=["GET"])
def list_models():
    return jsonify({
        "models": [
            {
                "id": "naive",
                "name": "Naive Baseline",
                "description": "Random ground fill with learned average height. Simple but no gaps or features.",
                "difficulty_param": "height_std",
            },
            {
                "id": "bigram",
                "name": "Bigram Transitions",
                "description": "Column-by-column generation using learned transition probabilities. Best playability.",
                "difficulty_param": "prior_weight / trans_weight",
            },
            {
                "id": "vae",
                "name": "Conditional Convolutional VAE",
                "description": "Deep generative model with 64-dim latent space, conditioned on a difficulty bucket (easy/medium/hard).",
                "difficulty_param": "bucket (easy/medium/hard)",
            },
        ]
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


# Serve React SPA
@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    print("Loading models...")
    ModelRegistry.get()
    print("Models loaded. Starting server...")
    debug = os.environ.get("FLASK_DEBUG", "1") != "0"
    app.run(debug=debug, port=5050)
