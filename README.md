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

## How to Run
1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
