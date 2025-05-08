# final_testing_function.py

import tensorflow as tf
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                           f1_score, roc_auc_score, roc_curve,
                           precision_recall_curve, confusion_matrix,
                           classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from skimage import io as skio, morphology, exposure
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import cv2
from sklearn.model_selection import train_test_split

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
CSV_FILE = 'full_df.csv'
IMAGE_DIR = 'ODIR-5K/Training Images'
SAMPLE_SIZE = 12198  # Added SAMPLE_SIZE to match training

# --- Preprocessing Functions ---
def extract_g_channel(image_path):
    """Enhanced G-channel extraction with adaptive illumination correction"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))

        # Extract G channel with CLAHE for better contrast
        g_channel = img[:, :, 1]
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g_channel = clahe.apply(g_channel)

        # Adaptive illumination correction
        blurred = cv2.GaussianBlur(g_channel, (0, 0), sigmaX=30)
        g_channel = cv2.addWeighted(g_channel, 2.5, blurred, -1.5, 0)
        g_channel = np.clip(g_channel, 0, 255).astype(np.uint8)

        return g_channel
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def top_bottom_hat_transform(gray_image):
    """Enhanced contrast using multi-scale top/bottom hat transforms"""
    if gray_image is None:
        return None

    # Convert to float and normalize to [0,1] if needed
    if gray_image.dtype != np.float32:
        gray_image = gray_image.astype(np.float32) / 255.0

    # Multi-scale structuring elements
    selem_small = morphology.disk(5)
    selem_medium = morphology.disk(15)
    selem_large = morphology.disk(25)

    # Multi-scale top-hat
    top_hat_small = morphology.white_tophat(gray_image, selem_small)
    top_hat_medium = morphology.white_tophat(gray_image, selem_medium)
    top_hat_large = morphology.white_tophat(gray_image, selem_large)

    # Multi-scale bottom-hat
    bottom_hat_small = morphology.black_tophat(gray_image, selem_small)
    bottom_hat_medium = morphology.black_tophat(gray_image, selem_medium)
    bottom_hat_large = morphology.black_tophat(gray_image, selem_large)

    # Combine transforms with weighted sum
    enhanced = gray_image + \
              0.4*top_hat_small + 0.3*top_hat_medium + 0.3*top_hat_large - \
              0.4*bottom_hat_small - 0.3*bottom_hat_medium - 0.3*bottom_hat_large

    # Clip values to [0,1] range before adaptive equalization
    enhanced = np.clip(enhanced, 0, 1)

    # Adaptive histogram equalization (input must be in [0,1] range)
    enhanced = exposure.equalize_adapthist(enhanced, clip_limit=0.03)

    # Convert back to 8-bit
    enhanced = (enhanced * 255).astype(np.uint8)

    return enhanced

def trilateral_filter(image):
    """Advanced trilateral filter with edge-preserving smoothing"""
    if image is None:
        return None

    # Convert to float32
    img_float = image.astype(np.float32) / 255.0

    # Step 1: Joint bilateral filter
    bilateral = cv2.bilateralFilter(img_float, d=9, sigmaColor=0.2, sigmaSpace=0.2)

    # Step 2: Calculate gradient magnitude
    grad_x = cv2.Sobel(img_float, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img_float, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # Step 3: Adaptive filtering based on gradient
    alpha = 0.8  # Edge preservation factor
    filtered = (1 - alpha*grad_mag) * bilateral + alpha*grad_mag*img_float

    # Step 4: Contrast enhancement
    filtered = cv2.normalize(filtered, None, 0, 1, cv2.NORM_MINMAX)
    filtered = (filtered * 255).astype(np.uint8)

    return filtered

# --- Feature Extraction Functions ---
def extract_luminance_features(image):
    """Advanced luminance feature extraction with texture analysis"""
    if image is None:
        return None
    
    # Basic luminance features
    luminance = image.astype(np.float32)
    mean_lum = np.mean(luminance)
    std_lum = np.std(luminance)

    # Local binary patterns for texture
    lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
    lbp_hist, _ = np.histogram(lbp, bins=10, range=(0, 10))
    lbp_hist = lbp_hist / lbp_hist.sum()

    # GLCM texture features
    glcm = graycomatrix(image, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]

    # Regional luminance variations
    regions = []
    for i in range(3):
        for j in range(3):
            region = image[i*74:(i+1)*74, j*74:(j+1)*74]
            regions.append(np.mean(region))
    regional_variation = np.std(regions)

    features = {
        'mean_luminance': mean_lum,
        'std_luminance': std_lum,
        'lbp_features': lbp_hist,
        'contrast': contrast,
        'energy': energy,
        'homogeneity': homogeneity,
        'regional_variation': regional_variation
    }

    return features

def preprocess_for_resnet(image):
    """Prepare single-channel image for ResNet (convert to 3-channel)"""
    if image is None:
        return None

    # Stack single channel to create 3-channel image
    img_3channel = np.stack([image]*3, axis=-1)
    # Normalize to [0,1]
    img_3channel = img_3channel.astype('float32') / 255.0
    # Subtract mean and divide by std (approximate ImageNet stats)
    img_3channel[..., 0] -= 0.485
    img_3channel[..., 1] -= 0.456
    img_3channel[..., 2] -= 0.406
    img_3channel[..., 0] /= 0.229
    img_3channel[..., 1] /= 0.224
    img_3channel[..., 2] /= 0.225
    return img_3channel

def prepare_features_for_lstm(luminance_features, resnet_features):
    """Prepare combined feature vector for LSTM input"""
    if luminance_features is None or resnet_features is None:
        return None
    
    # Convert luminance features to vector
    lum_vector = [
        luminance_features['mean_luminance'],
        luminance_features['std_luminance'],
        luminance_features['contrast'],
        luminance_features['energy'],
        luminance_features['homogeneity'],
        luminance_features['regional_variation']
    ]
    lum_vector.extend(luminance_features['lbp_features'])

    # Combine with ResNet features
    combined_features = np.concatenate([
        np.array(lum_vector),
        resnet_features.flatten()
    ])

    # Reshape for LSTM (sequence length = 1, features = combined length)
    return combined_features.reshape(1, 1, -1)

# --- Load Model ---
model = tf.keras.models.load_model('final_cataract_model.h5')

# --- Load ResNet Feature Extractor ---
from tensorflow import keras
from keras._tf_keras.keras.applications import ResNet50V2
from keras._tf_keras.keras.models import Model as KerasModel
from keras._tf_keras.keras.layers import GlobalAveragePooling2D
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
resnet_feature_extractor = KerasModel(inputs=base_model.input, outputs=x)
resnet_feature_extractor.trainable = False # Freeze the layers

def evaluate_model(csv_file, image_dir, batch_size=32, sample_size=SAMPLE_SIZE): #Added the samplesize
    """Evaluates the model on the dataset and calculates performance metrics."""
    # Load validation data
    df = pd.read_csv(csv_file)

    # Sample data if sample_size is not None
    if sample_size is not None:
        df = df.sample(sample_size, random_state=42)

    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Reset index for test_df
    test_df = test_df.reset_index(drop=True)

    # Initialize lists to store predictions and labels
    all_predictions = []
    all_labels = []

    # Process images in batches
    for i in range(0, len(test_df), batch_size):
        batch_df = test_df.iloc[i:i + batch_size]

        # Preprocess images in the current batch
        for idx, row in batch_df.iterrows():
            img_path = os.path.join(image_dir, row['filename'])
            label = row['label']  # Assuming 'label' is the correct column name

            try:
                g_channel = extract_g_channel(img_path)
                enhanced = top_bottom_hat_transform(g_channel)
                filtered = trilateral_filter(enhanced)
                lum_features = extract_luminance_features(filtered)
                resnet_input = preprocess_for_resnet(filtered)
                resnet_features = resnet_feature_extractor.predict(np.expand_dims(resnet_input, axis=0))
                lstm_input = prepare_features_for_lstm(lum_features, resnet_features)

                # Make prediction
                prediction = model.predict(lstm_input)[0][0]
                all_predictions.append(prediction)
                all_labels.append(label)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}")
                continue

    # Convert lists to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    # Convert probabilities to binary predictions
    all_predictions_class = (all_predictions > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_predictions_class)
    precision = precision_score(all_labels, all_predictions_class)
    recall = recall_score(all_labels, all_predictions_class)
    f1 = f1_score(all_labels, all_predictions_class)
    roc_auc = roc_auc_score(all_labels, all_predictions)  # Use probabilities for ROC AUC

    # Print metrics
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")
    print(f"ROC AUC: {roc_auc}")

    # Plot confusion matrix
    cm = confusion_matrix(all_labels, all_predictions_class)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=['Normal', 'Cataract'],
                yticklabels=['Normal', 'Cataract'])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()

    # Plot ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_predictions)  # Use probabilities for ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    evaluate_model(CSV_FILE, IMAGE_DIR, batch_size=32, sample_size=SAMPLE_SIZE)