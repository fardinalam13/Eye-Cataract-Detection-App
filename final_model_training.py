# final_model_training.py

import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from skimage import io as skio, morphology, exposure
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.layers import Attention
from keras._tf_keras.keras.applications import ResNet50V2
from keras._tf_keras.keras.models import Model as KerasModel
from keras._tf_keras.keras.layers import GlobalAveragePooling2D
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator


# Define constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
SAMPLE_SIZE = 12198  # Changed to 1000 # Set to None for full dataset
TEST_SIZE = 0.2
RANDOM_STATE = 42
BATCH_SIZE = 32
EPOCHS = 15
CSV_FILE = 'full_df.csv'
IMAGE_DIR = 'ODIR-5K/Training Images'

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

    return combined_features

# --- ResNet Feature Extractor setup ---
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
resnet_feature_extractor = KerasModel(inputs=base_model.input, outputs=x)
resnet_feature_extractor.trainable = False # Freeze the layers

# --- Load and Preprocess Data ---
def load_and_preprocess_data(csv_path, image_dir, sample_size=None):
    """Load and preprocess dataset"""
    df = pd.read_csv(csv_path)

    # Sample if needed
    if sample_size:
        df = df.sample(sample_size, random_state=42)

    # Prepare data arrays
    X_lstm = []
    y = []
    skipped_count = 0  # Counter for skipped images

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Load and preprocess image
            img_path = os.path.join(image_dir, row['filename'])
            g_channel = extract_g_channel(img_path)
            if g_channel is None:
                skipped_count += 1
                continue  # Skip to the next image

            enhanced = top_bottom_hat_transform(g_channel)
            if enhanced is None:
                skipped_count += 1
                continue

            filtered = trilateral_filter(enhanced)
            if filtered is None:
                skipped_count += 1
                continue

            # Extract features
            lum_features = extract_luminance_features(filtered)
            if lum_features is None:
                skipped_count += 1
                continue

            resnet_input = preprocess_for_resnet(filtered)
            if resnet_input is None:
                skipped_count += 1
                continue
            
            # Expand dimensions only once
            resnet_input = np.expand_dims(resnet_input, axis=0)
            resnet_features = resnet_feature_extractor.predict(resnet_input)
            
            # Prepare for LSTM
            lstm_input = prepare_features_for_lstm(lum_features, resnet_features)
            if lstm_input is None:
                skipped_count += 1
                continue

            X_lstm.append(lstm_input)

            # Get label (assuming 'label' column exists with 0/1 values)
            y.append(row['label'])

        except Exception as e:
            print(f"Error processing {row['filename']}: {e}")
            skipped_count += 1
            continue

    # Convert to numpy arrays
    X_lstm = np.array(X_lstm)
    y = np.array(y)

    print(f"Skipped {skipped_count} images due to errors.")

    return X_lstm, y

# --- Create LSTM Model ---
def create_lstm_model(input_shape, num_classes=1):
    """Create LSTM model with attention mechanism"""
    inputs = Input(shape=input_shape)

    # LSTM layer
    lstm_out = LSTM(128, return_sequences=False)(inputs)  # return_sequences=False
    lstm_out = Dropout(0.3)(lstm_out)

    # Classification head
    dense = Dense(64, activation='relu')(lstm_out)
    dense = Dropout(0.3)(dense)
    outputs = Dense(num_classes, activation='sigmoid')(dense)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

if __name__ == "__main__":
    # Load and preprocess data
    X, y = load_and_preprocess_data(CSV_FILE, IMAGE_DIR, sample_size=SAMPLE_SIZE) #Used smaller dataset instead of larger ones

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # Reshape X_train and X_val to have the correct input shape for the LSTM layer.
    # The expected input shape for the LSTM layer is (batch_size, timesteps, features).
    #  'X_train.shape[0]'is the batch size, which is the number of samples in the training set.
    #  '1'is the number of timesteps. It's set to 1 because  process each combined feature vector as a single time step in the LSTM.
    #  'X_train.shape[2]'is the number of features. It represents the combined length of the luminance features and ResNet features.
    
    num_features = X_train.shape[1]  # Number of features
    # Reshape the input data to (samples, timesteps, features)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, num_features))
    X_val = np.reshape(X_val, (X_val.shape[0], 1, num_features))

    input_shape = (1, num_features)  # timesteps, features
    model = create_lstm_model(input_shape)

    # Callbacks
    callbacks = [
        EarlyStopping(patience=5, restore_best_weights=True),
        ModelCheckpoint('final_cataract_model.h5', save_best_only=True)
    ]

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )

    print("Model training complete. Model saved as 'final_cataract_model.h5'")