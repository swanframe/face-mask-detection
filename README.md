# 😷 Face Mask Detection

Binary image classification to detect whether a person is **wearing a face mask or not**, using MobileNetV2 transfer learning.

## 📊 Results

| Metric | Score |
|--------|-------|
| Test Accuracy | **99.21%** |
| ROC-AUC | **0.9998** |
| Precision (with_mask) | 0.99 |
| Precision (without_mask) | 0.99 |
| F1-Score (macro avg) | 0.99 |

## 🛠️ Tech Stack

| Component | Detail |
|-----------|--------|
| Model | MobileNetV2 (Transfer Learning) |
| Framework | TensorFlow / Keras |
| Dataset | [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) — 7,553 images |
| Training | 2-phase: frozen backbone → fine-tuning |
| Platform | Google Colab (Tesla T4 GPU) |

## 🗂️ Project Structure
```
face-mask-detection/
├── src/
│   ├── config.py        # Hyperparameters & paths
│   ├── prepare_data.py  # Download & split dataset (80/10/10)
│   ├── train.py         # MobileNetV2 transfer learning + fine-tuning
│   ├── evaluate.py      # Metrics, confusion matrix, ROC curve
│   └── predict.py       # CLI inference on new images
├── models/              # Saved model weights (.keras)
├── outputs/             # Plots & evaluation reports
├── requirements.txt
├── .gitignore
└── README.md
```

## 🚀 Quickstart

### Google Colab
```bash
# 1. Setup Kaggle credentials (run interactively)
# 2. Run pipeline:
!python src/prepare_data.py
!python src/train.py
!python src/evaluate.py
!python src/predict.py --image path/to/image.jpg
```

### Local (venv)
```bash
git clone https://github.com/swanframe/face-mask-detection.git
cd face-mask-detection

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt

python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/predict.py --image path/to/image.jpg
```

## 🔍 Inference
```bash
# Basic usage
python src/predict.py --image image.jpg

# Use final model instead of best checkpoint
python src/predict.py --image image.jpg --model models/final_model.keras

# Skip saving visualization
python src/predict.py --image image.jpg --no-save
```

**Example output:**
```
==================================================
  PREDICTION RESULTS
==================================================
  Image : image.jpg
  Prediction : ✅  with_mask
  Status : Wearing a mask
  Confidence : 99.87%
--------------------------------------------------
  P(with_mask) : 99.87%
  P(without_mask) : 0.13%
==================================================
```

## 📈 Training Details

- **Phase 1** — Frozen backbone, train top layers only (10 epochs, LR=1e-3)
- **Phase 2** — Unfreeze top 30 layers for fine-tuning (10 epochs, LR=1e-5)
- **Callbacks** — ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## 📁 Dataset Split

| Split | with_mask | without_mask | Total |
|-------|-----------|--------------|-------|
| Train | 2,980 | 3,062 | 6,042 |
| Val   | 372   | 382   | 754   |
| Test  | 373   | 384   | 757   |

## 📄 License

MIT
