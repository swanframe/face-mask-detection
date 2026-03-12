# Face Mask Detection

Binary image classification to detect whether a person is wearing a face mask or not.

## Tech Stack
- **Model**: MobileNetV2 (Transfer Learning)
- **Framework**: TensorFlow / Keras
- **Dataset**: [Face Mask Dataset](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset) — 7,553 images

## Classes
| Label | Description |
|-------|-------------|
| `with_mask` | Person wearing a face mask |
| `without_mask` | Person not wearing a face mask |

## Project Structure
```
face-mask-detection/
├── src/
│   ├── config.py        # Hyperparameters & paths
│   ├── prepare_data.py  # Download & split dataset
│   ├── train.py         # Model training
│   ├── evaluate.py      # Evaluation & visualization
│   └── predict.py       # Inference on new images
├── models/              # Saved model weights
├── outputs/             # Plots & evaluation reports
├── requirements.txt
├── .gitignore
└── README.md
```

## Quickstart

### Google Colab
```bash
!python src/prepare_data.py
!python src/train.py
!python src/evaluate.py
!python src/predict.py --image image.jpg
```

### Local (venv)
```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python src/prepare_data.py
python src/train.py
python src/evaluate.py
python src/predict.py --image image.jpg
```

## Results
*(Will be updated after training)*

## License
MIT
