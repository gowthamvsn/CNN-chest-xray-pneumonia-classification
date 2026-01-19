# ==============================
# CONFIGURATION & IMPORTS
# ==============================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import cv2
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import keras_tuner as kt

# For Grad-CAM
import tensorflow.keras.backend as K

# For SHAP
import shap
# ==============================
# GPU CONFIGURATION
# ==============================

# GPU Configuration for RTX 3050
print("Setting up GPU configuration for RTX 3050...")
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU found: {gpus}")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU found, using CPU")
# ==============================
# GLOBAL SETTINGS & PATHS
# ==============================

# Configuration optimized for RTX 3050
model_path = "pneumonia_best_tuned_model.h5"
base_dir = os.environ.get("CHEST_XRAY_DATASET", "data/chest_xray")
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'val')
test_dir = os.path.join(base_dir, 'test')
img_size = (150, 150)  # Increased for better performance
batch_size = 16  # Optimized for RTX 3050
test_image_path = os.path.join(test_dir, 'PNEUMONIA', 'person19_virus_50.jpeg')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
# ==============================
# DATA LOADING & PREPROCESSING
# ==============================

print("Setting up data generators...")

# Data preprocessing with augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

# Data generators
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

print("Class indices:", train_generator.class_indices)
print(f"Training samples: {train_generator.samples}")
print(f"Validation samples: {val_generator.samples}")
print(f"Test samples: {test_generator.samples}")

# ==============================
# CUSTOM CNN + HYPERPARAMETER TUNING
# ==============================

# =============================================================================
# 1. HYPERPARAMETER TUNING WITH KERAS TUNER (RTX 3050 Optimized)
# =============================================================================

def build_model(hp):
    """Build model with hyperparameters to tune - optimized for RTX 3050"""
    model = Sequential()
    
    # First Conv Block
    model.add(Conv2D(
        filters=hp.Int('conv1_filters', min_value=32, max_value=64, step=16),
        kernel_size=hp.Choice('conv1_kernel', values=[3, 5]),
        activation='relu',
        input_shape=(150, 150, 3)
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Batch Normalization
    if hp.Boolean('use_batch_norm_1'):
        model.add(BatchNormalization())
    
    # Second Conv Block
    model.add(Conv2D(
        filters=hp.Int('conv2_filters', min_value=64, max_value=128, step=32),
        kernel_size=hp.Choice('conv2_kernel', values=[3, 5]),
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if hp.Boolean('use_batch_norm_2'):
        model.add(BatchNormalization())
    
    # Third Conv Block
    model.add(Conv2D(
        filters=hp.Int('conv3_filters', min_value=128, max_value=256, step=64),
        kernel_size=3,
        activation='relu'
    ))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    if hp.Boolean('use_batch_norm_3'):
        model.add(BatchNormalization())
    
    # Fourth Conv Block (optional)
    if hp.Boolean('use_conv4'):
        model.add(Conv2D(
            filters=hp.Int('conv4_filters', min_value=256, max_value=512, step=128),
            kernel_size=3,
            activation='relu'
        ))
        model.add(MaxPooling2D(pool_size=(2, 2)))
    
    # Flatten and Dense layers
    model.add(Flatten())
    
    # First Dense layer
    model.add(Dense(
        units=hp.Int('dense1_units', min_value=128, max_value=512, step=64),
        activation='relu'
    ))
    
    # Dropout
    model.add(Dropout(hp.Float('dropout1', min_value=0.3, max_value=0.7, step=0.1)))
    
    # Second Dense layer (optional)
    if hp.Boolean('use_dense2'):
        model.add(Dense(
            units=hp.Int('dense2_units', min_value=64, max_value=256, step=32),
            activation='relu'
        ))
        model.add(Dropout(hp.Float('dropout2', min_value=0.3, max_value=0.6, step=0.1)))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=hp.Float('learning_rate', min_value=1e-5, max_value=1e-2, sampling='LOG')),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def run_hyperparameter_tuning():
    """Run hyperparameter tuning using Keras Tuner"""
    print("Starting hyperparameter tuning...")
    
    # Initialize tuner with reduced trials for RTX 3050
    tuner = kt.RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=15,  # Reduced for RTX 3050
        directory='pneumonia_tuning',
        project_name='pneumonia_classification'
    )
    
    # Callbacks
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
    
    # Start tuning
    tuner.search(
        train_generator,
        epochs=10,  # Reduced epochs for faster tuning
        validation_data=val_generator,
        callbacks=[early_stop, reduce_lr],
        verbose=1
    )
    
    # Get best hyperparameters and model
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_model = tuner.hypermodel.build(best_hps)
    
    print("Best hyperparameters found:")
    for param, value in best_hps.values.items():
        print(f"{param}: {value}")
    
    return best_model, best_hps

def train_best_model(model):
    """Train the best model with more epochs"""
    print("Training best model...")
    
    # Callbacks
    checkpoint = ModelCheckpoint(
        model_path,
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)
    
    # Train model
    history = model.fit(
        train_generator,
        epochs=30,  # Reduced for RTX 3050
        validation_data=val_generator,
        callbacks=[checkpoint, early_stop, reduce_lr],
        verbose=1
    )
    
    return history

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.show()
# ==============================
# MODEL TRAINING OR LOADING
# ==============================

# Check if model exists, otherwise train it
if os.path.exists(model_path):
    print(f"Loading existing model from {model_path}...")
    model = load_model(model_path)
    print("Model loaded successfully!")
else:
    print("No existing model found. Starting training process...")
    # Run hyperparameter tuning
    best_model, best_hps = run_hyperparameter_tuning()
    # Train the best model
    history = train_best_model(best_model)
    # Plot training history
    plot_training_history(history)
    # Load the saved best model
    model = load_model(model_path)
    print("Model training completed and saved!")

# ==============================
# MODEL EXPLAINABILITY: GRAD-CAM
# ==============================

# =============================================================================
# 2. GRAD-CAM IMPLEMENTATION
# =============================================================================

class GradCAM:
    def __init__(self, model, class_idx, layer_name):
        self.model = model
        self.class_idx = class_idx
        self.layer_name = layer_name
        self.grad_model = self._get_grad_model()
    
    def _get_grad_model(self):
        """Create a model that maps the input image to the activations of the last conv layer"""
        return Model(
            inputs=[self.model.inputs],
            outputs=[self.model.get_layer(self.layer_name).output, self.model.output]
        )
    
    def generate_heatmap(self, img_array):
        """Generate Grad-CAM heatmap"""
        # Convert to tensor
        img_tensor = tf.cast(img_array, tf.float32)
        img_tensor = tf.expand_dims(img_tensor, axis=0)
        
        # Get the gradient of the top predicted class for our input image
        with tf.GradientTape() as tape:
            conv_outputs, predictions = self.grad_model(img_tensor)
            loss = predictions[:, self.class_idx]
        
        # Extract gradients
        grads = tape.gradient(loss, conv_outputs)
        
        # Global average pooling of gradients
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        
        # Weight the channels by their importance
        conv_outputs = conv_outputs[0]
        heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        
        # Normalize heatmap
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        
        return heatmap.numpy()
    
    def overlay_heatmap(self, img, heatmap, alpha=0.4):
        """Overlay heatmap on original image"""
        # Resize heatmap to match image size
        heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_resized = np.uint8(255 * heatmap_resized)
        
        # Apply colormap
        heatmap_colored = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        
        # Convert BGR to RGB
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        # Overlay
        overlayed_img = heatmap_colored * alpha + img * (1 - alpha)
        
        return overlayed_img.astype(np.uint8)

def find_last_conv_layer(model):
    """Find the last convolutional layer in the model"""
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            return layer.name
    return None

def visualize_gradcam(model, img_path, layer_name=None):
    """Visualize Grad-CAM for a given image"""
    # Find the last convolutional layer if not specified
    if layer_name is None:
        layer_name = find_last_conv_layer(model)
    
    if layer_name is None:
        print("No convolutional layer found in the model")
        return None, None
    
    print(f"Using layer: {layer_name}")
    
    # Load and preprocess image
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0
    
    # Get prediction
    pred = model.predict(np.expand_dims(img_array, axis=0))[0][0]
    class_prediction = "Pneumonia" if pred > 0.5 else "Normal"
    confidence = pred if pred > 0.5 else 1 - pred
    
    print(f"Prediction: {class_prediction} (Confidence: {confidence:.3f})")
    
    # Generate Grad-CAM
    class_idx = 0  # Binary classification, so we use class 0
    gradcam = GradCAM(model, class_idx, layer_name)
    heatmap = gradcam.generate_heatmap(img_array)
    
    # Overlay heatmap on original image
    overlayed = gradcam.overlay_heatmap(np.array(img), heatmap)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    axes[2].imshow(overlayed)
    axes[2].set_title(f'Overlay\n{class_prediction} ({confidence:.3f})')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return heatmap, overlayed

# Run Grad-CAM analysis on the test image
print("\n" + "="*50)
print("GRAD-CAM ANALYSIS")
print("="*50)

if os.path.exists(test_image_path):
    heatmap, overlay = visualize_gradcam(model, test_image_path)
else:
    print(f"Test image not found at: {test_image_path}")
    # Try to find an alternative test image
    pneumonia_dir = os.path.join(test_dir, 'PNEUMONIA')
    if os.path.exists(pneumonia_dir):
        test_files = [f for f in os.listdir(pneumonia_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
        if test_files:
            alternative_path = os.path.join(pneumonia_dir, test_files[0])
            print(f"Using alternative test image: {test_files[0]}")
            heatmap, overlay = visualize_gradcam(model, alternative_path)

# ==============================
# MODEL EXPLAINABILITY: SHAP
# ==============================

# =============================================================================
# 3. SHAP ANALYSIS
# =============================================================================

def create_shap_explainer(model, background_data):
    """Create SHAP explainer for the model"""
    explainer = shap.DeepExplainer(model, background_data)
    return explainer

def analyze_with_shap(model, test_images, num_samples=3):
    """Analyze model predictions using SHAP - reduced samples for RTX 3050"""
    print("Preparing SHAP analysis...")
    
    # Get background data (sample from training set)
    background_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=10,
        class_mode='binary',
        shuffle=True
    )
    
    # Get one batch for background
    bg_batch, _ = next(background_generator)
    background_data = bg_batch[:5]  # Reduced for RTX 3050
    
    print("Creating SHAP explainer...")
    # Create SHAP explainer
    explainer = create_shap_explainer(model, background_data)
    
    print("Computing SHAP values...")
    # Get SHAP values for test images
    shap_values = explainer.shap_values(test_images[:num_samples])
    
    print("Plotting SHAP values...")
    # Plot SHAP values
    shap.image_plot(shap_values[0], test_images[:num_samples])
    
    return shap_values

def get_sample_test_images(num_samples=3):
    """Get sample test images for SHAP analysis"""
    test_images = []
    test_labels = []
    
    # Get samples from test generator
    test_batch, test_batch_labels = next(test_generator)
    
    for i in range(min(num_samples, len(test_batch))):
        test_images.append(test_batch[i])
        test_labels.append(test_batch_labels[i])
    
    return np.array(test_images), np.array(test_labels)

# Run SHAP analysis
print("\n" + "="*50)
print("SHAP ANALYSIS")
print("="*50)

sample_images, sample_labels = get_sample_test_images(3)
shap_values = analyze_with_shap(model, sample_images)

# ==============================
# MODEL EVALUATION & METRICS
# ==============================

# =============================================================================
# 4. MODEL EVALUATION
# =============================================================================

def evaluate_model(model):
    """Comprehensive model evaluation"""
    print("Evaluating model...")
    
    # Get predictions
    test_generator.reset()
    predictions = model.predict(test_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, predictions)
    auc_score = roc_auc_score(y_true, predictions)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()
    
    print(f"\nModel Performance Summary:")
    print(f"AUC Score: {auc_score:.3f}")
    print(f"Accuracy: {np.mean(y_pred == y_true):.3f}")
    
    return predictions, y_pred, y_true

# Run model evaluation
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

predictions, y_pred, y_true = evaluate_model(model)

print("\n" + "="*50)
print("ANALYSIS COMPLETE!")
print("="*50)
print(f"Model saved at: {model_path}")
print("All analyses have been completed successfully!")






# ==============================
# EXPERIMENTAL METHODS (OPTIONAL)
# ==============================

###############################################################
## For additional method trials --

import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def apply_clahe(img):
    """Apply CLAHE to each channel of an image."""
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_yuv[:, :, 0] = clahe.apply(img_yuv[:, :, 0])
    img_clahe = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    return img_clahe

def clahe_preprocessing(img):
    img = img.astype(np.uint8)
    return apply_clahe(img) / 255.0

train_datagen = ImageDataGenerator(
    preprocessing_function=clahe_preprocessing,
    rotation_range=30,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=clahe_preprocessing)
test_datagen = ImageDataGenerator(preprocessing_function=clahe_preprocessing)
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.models import Model

base_model = MobileNetV2(input_shape=(150,150,3), include_top=False, weights='imagenet')
base_model.trainable = False  # Fine-tune later

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weights = dict(enumerate(class_weights))
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss=BinaryCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(patience=7, restore_best_weights=True, monitor='val_loss'),
    ModelCheckpoint('mobilenetv2_best_model.h5', save_best_only=True, monitor='val_accuracy'),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-7)
]

history = model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)
base_model.trainable = True
for layer in base_model.layers[:-50]:  # Unfreeze last 50 layers
    layer.trainable = False

model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss=BinaryCrossentropy(label_smoothing=0.05),
    metrics=['accuracy']
)

history_finetune = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    class_weight=class_weights,
    callbacks=callbacks
)
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_model(model, test_generator):
    # Reset generator to avoid skipping
    test_generator.reset()
    
    # Get predictions
    predictions = model.predict(test_generator, verbose=1)
    y_pred = (predictions > 0.5).astype(int)
    y_true = test_generator.classes
    
    # Classification report
    print("\n📋 Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Pneumonia'], 
                yticklabels=['Normal', 'Pneumonia'])
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()
    
    # ROC Curve & AUC
    fpr, tpr, _ = roc_curve(y_true, predictions)
    auc_score = roc_auc_score(y_true, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})', color='darkorange')
    plt.plot([0, 1], [0, 1], linestyle='--', color='navy')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    print(f" Final Accuracy: {(y_pred == y_true).mean():.3f}")
    print(f" AUC Score: {auc_score:.3f}")

# Run the evaluation
evaluate_model(model, test_generator)


def print_model_configuration():
    print("\n🛠️ Model Configuration Summary")
    print(f"Base Model: MobileNetV2 (pretrained on ImageNet)")
    print(f"Image Size: {img_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Optimizer: Adam")
    print(f"Initial Learning Rate: 1e-4")
    print(f"Loss: BinaryCrossentropy (with label_smoothing = 0.1)")
    print(f"Augmentation: rotation=30, zoom=0.2, CLAHE preprocessing")
    print(f"Class Weights: {class_weights}")
    print(f"Dropout Rate: 0.5 after GlobalAveragePooling2D")
    print(f"Model Saved At: mobilenetv2_best_model.h5")

print_model_configuration()
def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    
    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

# If you have history from model.fit()
plot_training_history(history)
