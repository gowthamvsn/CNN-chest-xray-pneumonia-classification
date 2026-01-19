# Chest X-Ray Pneumonia Classification with Deep Learning

This project classifies chest X-ray images as **Normal** or **Pneumonia**
using deep learning models built with TensorFlow/Keras.

The project includes model training, hyperparameter tuning,
model explainability (Grad-CAM and SHAP), and performance evaluation.

---

## Dataset
- Chest X-Ray Images (Pneumonia) dataset
- Original source: Kaggle / NIH
- Images categorized into:
  - Normal
  - Pneumonia
export CHEST_XRAY_DATASET=/path/to/chest_xray

> ⚠️ Dataset is not included due to size and licensing restrictions.

---

## Models Implemented

| # | Approach                          | Base Architecture     | Key Features                              | Result File                     |
|---|-----------------------------------|-----------------------|-------------------------------------------|---------------------------------|
| 1 | Custom CNN + Keras Tuner          | Sequential CNN        | Hyperparameter search, BatchNorm, Dropout | pneumonia_best_tuned_model.h5   |
| 2 | Transfer Learning + Fine-tuning   | MobileNetV2           | CLAHE preprocessing, class weights, label smoothing | mobilenetv2_best_model.h5  |
---

## Explainability
- **Grad-CAM** to visualize regions influencing predictions
- **SHAP** for model interpretability

---

## Evaluation Metrics
- Accuracy
- ROC-AUC
- Confusion Matrix
- Classification Report

---

## File Structure
- `train_and_explain_pneumonia.py` – Main training, explainability, and evaluation pipeline

---

## Hardware

Optimized for NVIDIA RTX 3050 GPU
Supports CPU fallback

## Project Structure
'''
chest-xray-pneumonia-detection/
├── data/
│ └── README.md # Explains dataset source & structure
├── notebooks/
│ ├── pneumonia_detection.ipynb # Custom CNN + Grad-CAM + SHAP
│ └── mobilenetv2_approach.ipynb # Transfer learning
├── models/
│ ├── pneumonia_best_tuned_model.h5
│ └── mobilenetv2_best_model.h5
├── pneumonia_tuning/ # Keras Tuner results
├── README.md
├── requirements.txt
└── .gitignore
'''

>  **Start here:** `notebooks/train_and_explain_pneumonia.py`

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
