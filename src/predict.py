# =============================================================================
# predict.py — CLI inference: predict mask status on a new image
# Usage: python src/predict.py --image path/to/image.jpg
#        python src/predict.py --image path/to/image.jpg --model models/final_model.keras
# =============================================================================

import os
import sys
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    IMG_SIZE, MODEL_BEST_PATH, OUTPUT_DIR, CLASSES
)


# ---------------------------------------------------------------------------
# Label styling
# ---------------------------------------------------------------------------
LABEL_CONFIG = {
    "with_mask": {
        "color": "#2ecc71",
        "emoji": "✅",
        "message": "Wearing a mask",
    },
    "without_mask": {
        "color": "#e74c3c",
        "emoji": "❌",
        "message": "Not wearing a mask",
    },
}


def load_model(model_path: str) -> tf.keras.Model:
    """Load a saved Keras model from disk."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = tf.keras.models.load_model(model_path)
    return model


def preprocess_image(image_path: str) -> np.ndarray:
    """Load and preprocess a single image for inference."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.mobilenet_v2.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)   # shape: (1, 224, 224, 3)


def predict_single(image_path: str, model: tf.keras.Model) -> dict:
    """
    Run inference on a single image.
    Returns a dict with label, confidence, and probabilities.
    """
    img_array = preprocess_image(image_path)
    prob_without_mask = float(model.predict(img_array, verbose=0)[0][0])
    prob_with_mask    = 1.0 - prob_without_mask

    predicted_class = CLASSES[1] if prob_without_mask >= 0.5 else CLASSES[0]
    confidence      = prob_without_mask if predicted_class == CLASSES[1] else prob_with_mask

    return {
        "label"             : predicted_class,
        "confidence"        : confidence,
        "prob_with_mask"    : prob_with_mask,
        "prob_without_mask" : prob_without_mask,
    }


def visualize_prediction(image_path: str, result: dict, save: bool = True) -> None:
    """Display and optionally save a prediction visualization."""
    cfg   = LABEL_CONFIG[result["label"]]
    color = cfg["color"]

    img = Image.open(image_path).convert("RGB")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#1a1a2e")

    # --- Left: image with prediction overlay ---
    axes[0].imshow(img)
    axes[0].set_title(
        f"{cfg['emoji']}  {cfg['message']}\n"
        f"Confidence: {result['confidence']*100:.1f}%",
        fontsize=14, fontweight="bold",
        color=color, pad=12,
    )
    axes[0].axis("off")
    for spine in axes[0].spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)

    # --- Right: probability bar chart ---
    labels = ["with_mask", "without_mask"]
    probs  = [result["prob_with_mask"], result["prob_without_mask"]]
    colors = [LABEL_CONFIG[l]["color"] for l in labels]

    bars = axes[1].barh(labels, probs, color=colors, edgecolor="white",
                        linewidth=0.8, height=0.5)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability", color="white", fontsize=11)
    axes[1].set_title("Class Probabilities", color="white",
                      fontsize=13, fontweight="bold")
    axes[1].tick_params(colors="white")
    axes[1].set_facecolor("#16213e")
    for spine in axes[1].spines.values():
        spine.set_edgecolor("#444")

    for bar, prob in zip(bars, probs):
        axes[1].text(
            min(prob + 0.02, 0.95), bar.get_y() + bar.get_height() / 2,
            f"{prob*100:.1f}%", va="center", color="white", fontsize=12,
        )

    plt.tight_layout(pad=2)

    if save:
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path  = os.path.join(OUTPUT_DIR, f"prediction_{base_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        print(f"  💾 Visualization saved: {out_path}")

    plt.show()
    plt.close()


def print_result(image_path: str, result: dict) -> None:
    """Print a formatted prediction summary to terminal."""
    cfg = LABEL_CONFIG[result["label"]]
    print("\n" + "=" * 50)
    print("  PREDICTION RESULTS")
    print("=" * 50)
    print(f"  Image : {os.path.basename(image_path)}")
    print(f"  Prediction : {cfg['emoji']}  {result['label']}")
    print(f"  Status : {cfg['message']}")
    print(f"  Confidence : {result['confidence']*100:.2f}%")
    print("-" * 50)
    print(f"  P(with_mask) : {result['prob_with_mask']*100:.2f}%")
    print(f"  P(without_mask) : {result['prob_without_mask']*100:.2f}%")
    print("=" * 50)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Face Mask Detection — Single Image Inference"
    )
    parser.add_argument(
        "--image", type=str, required=True,
        help="Path to the input image (jpg/png)"
    )
    parser.add_argument(
        "--model", type=str, default=MODEL_BEST_PATH,
        help=f"Path to the model file (default: {MODEL_BEST_PATH})"
    )
    parser.add_argument(
        "--no-save", action="store_true",
        help="Do not save the visualization to outputs/"
    )
    args = parser.parse_args()

    print("\n🔄 Loading model...")
    model = load_model(args.model)
    print(f"  ✅ Model loaded: {args.model}")

    print("\n🖼️  Processing image...")
    result = predict_single(args.image, model)

    print_result(args.image, result)
    visualize_prediction(args.image, result, save=not args.no_save)


if __name__ == "__main__":
    main()
