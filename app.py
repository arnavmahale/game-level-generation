"""
Flask web app for interactive game level generation.

Serves a React frontend and exposes an API for generating levels with
three models — naive baseline, bigram transitions, and a conditional
Conv-VAE — plus a BFS playability repair post-pass.
"""

import sys
import os
import numpy as np
import torch
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))

from data_utils import load_levels, train_val_test_split_levels, get_model_path
from model_naive import NaiveBaseline
from model_classical import BigramModel
from model_deep import (
    ConvVAE, generate_levels as vae_generate,
    three_class_to_api, N_BUCKETS,
)
from repair import repair as repair_level, enforce_layout
from evaluate import (
    distribution_similarity, structural_metrics, check_playability,
)

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

        # Load training data for JS divergence evaluation. Use the same
        # train partition the model saw, so JS reflects train-vs-generated.
        levels = load_levels()
        self.train_levels, _, _ = train_val_test_split_levels(levels)

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
CORS(app)


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
    js = float(distribution_similarity(registry.train_levels, levels))
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
    app.run(debug=True, port=5050)
