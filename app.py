from flask import Flask, request, jsonify, render_template
import numpy as np
from PIL import Image, ImageOps
import io
import os

app = Flask(__name__)

# ──────────────────────────────────────────────
# Load the Teachable Machine model at startup
# ──────────────────────────────────────────────
MODEL_PATH  = os.path.join("model", "keras_model.h5")
LABELS_PATH = os.path.join("model", "labels.txt")

model  = None
labels = []

def load_model():
    """Load the Keras model and class labels exported from Teachable Machine."""
    global model, labels

    # keras is imported here so the app still runs (for front-end work)
    # even when the model file hasn't been added yet.
    try:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load(MODEL_PATH, compile=False)
        with open(LABELS_PATH, "r") as f:
            labels = [line.strip() for line in f.readlines()]
        print(f"✅  Model loaded — classes: {labels}")
    except Exception as e:
        print(f"⚠️  Model not loaded yet: {e}")
        print("    Add keras_model.h5 and labels.txt to the /model folder.")


def prepare_image(image_bytes: bytes) -> np.ndarray:
    """
    Resize and normalise an uploaded image so it matches the
    224 × 224 input that Teachable Machine expects.
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
    arr   = np.asarray(image, dtype=np.float32)
    arr   = (arr / 127.5) - 1.0          # normalise to [-1, 1]
    return np.expand_dims(arr, axis=0)   # shape → (1, 224, 224, 3)


# ──────────────────────────────────────────────
# Routes
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive an image, run inference, return JSON with label + confidence."""

    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file  = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if model is None:
        # Model not loaded — return a friendly placeholder so the
        # front-end can still be tested before training is complete.
        return jsonify({
            "label":      "Model not loaded yet",
            "confidence": 0,
            "tip":        "Add your keras_model.h5 and labels.txt to the /model folder."
        }), 200

    try:
        image_bytes = file.read()
        data        = prepare_image(image_bytes)
        predictions = model.predict(data)[0]           # array of probabilities

        best_idx    = int(np.argmax(predictions))
        best_label  = labels[best_idx].split(" ", 1)[-1]  # strip the leading index
        confidence  = float(predictions[best_idx]) * 100

        # Build a readable tip based on the prediction
        tip = get_tip(best_label, confidence)

        return jsonify({
            "label":      best_label,
            "confidence": round(confidence, 1),
            "tip":        tip,
            "all_scores": {
                labels[i].split(" ", 1)[-1]: round(float(predictions[i]) * 100, 1)
                for i in range(len(labels))
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_tip(label: str, confidence: float) -> str:
    """Return a helpful human-readable tip based on the predicted label."""
    label_lower = label.lower()

    tips = {
        "ripe": [
            "Perfect timing! This fruit is ready to eat right now. 🍓",
            "Great pick! Enjoy it today for peak flavour.",
            "This one's ready — don't wait too long or it'll over-ripen!",
        ],
        "unripe": [
            "Not quite yet! Leave it out at room temperature for a day or two.",
            "Still needs some time. Keep it on the counter, away from direct sunlight.",
            "Patience! It'll be sweeter once it ripens fully.",
        ],
        "overripe": [
            "Past its prime for eating fresh, but perfect for smoothies or baking!",
            "Use it in a banana bread or blend it into a smoothie. Don't waste it!",
        ],
    }

    for key, tip_list in tips.items():
        if key in label_lower:
            import random
            return random.choice(tip_list)

    return "Scan complete! Check the confidence score above."


# ──────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    app.run(debug=True)
