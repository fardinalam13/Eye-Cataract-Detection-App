from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import cv2
import os
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from skimage import io as skio, morphology, exposure
from tensorflow import keras
from keras._tf_keras.keras.models import load_model
import final_model_training # Import for resnet_feature_extractor

app = Flask(__name__)
CORS(app)

# Load the model
model = load_model('final_cataract_model.h5')

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224

# --- Preprocessing Functions ---
def extract_g_channel(img):
    """Extracts the G channel from an RGB image."""

    g_channel = img[:, :, 1]

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    g_channel = clahe.apply(g_channel)

    # Adaptive illumination correction
    blurred = cv2.GaussianBlur(g_channel, (0, 0), sigmaX=30)
    g_channel = cv2.addWeighted(g_channel, 2.5, blurred, -1.5, 0)
    g_channel = np.clip(g_channel, 0, 255).astype(np.uint8)
    return g_channel

def top_bottom_hat_transform(gray_image):
    """Enhanced contrast using multi-scale top/bottom hat transforms"""
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

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Handles image uploads, preprocesses the image, and returns the model's prediction.
    """
    try:
        imagefile = request.files['image']
        image = Image.open(io.BytesIO(imagefile.read()))
        image = image.resize((IMG_WIDTH, IMG_HEIGHT)) # Resize Image

        # Convert image to numpy array
        img = np.array(image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        g_channel = extract_g_channel(img) #Extract G-Channel

        # Apply Top-Bottom Hat Transformation
        enhanced = top_bottom_hat_transform(g_channel) #Apply Top Bottom Hat

        # Apply Trilateral Filter
        filtered = trilateral_filter(enhanced) #Apply Trilateral Filter

        # Extract Luminance Feature
        lum_features = extract_luminance_features(filtered) #Extract Luminance feature
        
        # Prepare Image for Resnet
        resnet_input = preprocess_for_resnet(filtered) #Apply Resnet-18(Feature Extraction)

        # Make prediction
        resnet_features = final_model_training.resnet_feature_extractor.predict(np.expand_dims(resnet_input, axis=0))
        lstm_input = prepare_features_for_lstm(lum_features, resnet_features) #Call LSTM

        prediction = model.predict(lstm_input)[0][0]

        #probability = float(prediction if prediction > 0.5 else prediction)  #probability = float(prediction if prediction > 0.5 else 1 - prediction)
        # Return probability as float (will be formatted as percentage in frontend)
        probability = float(prediction)

        prediction = "Cataract" if prediction > 0.5 else "Normal"


        return jsonify({
            'prediction': prediction,
            'probability': probability
        })
    except Exception as e:
        return jsonify({'error': str(e)})
if __name__ == '__main__':
    app.run(debug=True)