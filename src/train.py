# =============================================================================
# train.py — MobileNetV2 transfer learning with two-phase training
# =============================================================================

import os
import sys
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import (
    TRAIN_DIR, VAL_DIR, IMG_SIZE, BATCH_SIZE,
    EPOCHS_FROZEN, EPOCHS_FINETUNE,
    LEARNING_RATE, FINETUNE_LR, UNFREEZE_LAYERS,
    ROTATION_RANGE, WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE,
    ZOOM_RANGE, HORIZONTAL_FLIP,
    MODEL_BEST_PATH, MODEL_FINAL_PATH, OUTPUT_DIR, CLASSES
)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def build_data_generators():
    """Create train and validation data generators with augmentation."""

    train_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
        rotation_range=ROTATION_RANGE,
        width_shift_range=WIDTH_SHIFT_RANGE,
        height_shift_range=HEIGHT_SHIFT_RANGE,
        zoom_range=ZOOM_RANGE,
        horizontal_flip=HORIZONTAL_FLIP,
        fill_mode="nearest",
    )

    val_datagen = ImageDataGenerator(
        preprocessing_function=tf.keras.applications.mobilenet_v2.preprocess_input,
    )

    train_gen = train_datagen.flow_from_directory(
        TRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=True,
        seed=SEED,
    )

    val_gen = val_datagen.flow_from_directory(
        VAL_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="binary",
        classes=CLASSES,
        shuffle=False,
    )

    return train_gen, val_gen


def build_model() -> tf.keras.Model:
    """Build MobileNetV2-based binary classifier."""

    base_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False   # freeze backbone for phase 1

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)   # binary output

    model = models.Model(inputs, outputs, name="facemask_mobilenetv2")
    return model, base_model


def get_callbacks(phase: str) -> list:
    """Return training callbacks for a given phase."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(MODEL_BEST_PATH), exist_ok=True)

    cb_list = [
        callbacks.ModelCheckpoint(
            filepath=MODEL_BEST_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=5,
            restore_best_weights=True,
            verbose=1,
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-7,
            verbose=1,
        ),
        callbacks.CSVLogger(
            os.path.join(OUTPUT_DIR, f"history_{phase}.csv"),
            append=False,
        ),
    ]
    return cb_list


def save_history(history, phase: str) -> None:
    """Persist training history as JSON for later plotting."""
    path = os.path.join(OUTPUT_DIR, f"history_{phase}.json")
    with open(path, "w") as f:
        json.dump(history.history, f, indent=2)
    print(f"  💾 History saved: {path}")


def train() -> None:
    print("=" * 60)
    print("  FACE MASK DETECTION — TRAINING PIPELINE")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    print("\n📂 Loading data generator...")
    train_gen, val_gen = build_data_generators()
    print(f"  ✅ Train : {train_gen.samples} images | "
          f"{train_gen.num_classes} classes")
    print(f"  ✅ Val   : {val_gen.samples} images")
    print(f"  📌 Class indices: {train_gen.class_indices}")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    print("\n🏗️  Building the model...")
    model, base_model = build_model()
    model.summary()

    # ------------------------------------------------------------------
    # PHASE 1 — Frozen backbone
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("  TRAINING PHASE 1: Top layers only (backbone frozen)")
    print("=" * 60)

    model.compile(
        optimizer=optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history_frozen = model.fit(
        train_gen,
        epochs=EPOCHS_FROZEN,
        validation_data=val_gen,
        callbacks=get_callbacks("frozen"),
        verbose=1,
    )
    save_history(history_frozen, "frozen")

    # ------------------------------------------------------------------
    # PHASE 2 — Fine-tune top N layers of backbone
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  TRAINING PHASE 2: Fine-tuning (unfreeze {UNFREEZE_LAYERS} layers)")
    print("=" * 60)

    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAYERS]:
        layer.trainable = False

    trainable_count = sum(1 for l in model.layers if l.trainable)
    print(f"  🔓 Trainable layers: {trainable_count}")

    model.compile(
        optimizer=optimizers.Adam(learning_rate=FINETUNE_LR),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    history_finetune = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=get_callbacks("finetune"),
        verbose=1,
    )
    save_history(history_finetune, "finetune")

    # ------------------------------------------------------------------
    # Save final model
    # ------------------------------------------------------------------
    model.save(MODEL_FINAL_PATH)
    print(f"\n💾 Final model saved: {MODEL_FINAL_PATH}")
    print("\n🎉 Training completed!")


if __name__ == "__main__":
    train()
