import tensorflow as tf
import numpy as np
import cv2
from skimage import io as skio, morphology, exposure
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops
from PIL import Image

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
IMAGE_PATH = 'ODIR-5K/Training Images/0_right.jpg'  # Replace with your actual image path

# --- Preprocessing Functions ---
def extract_g_channel(image):
    """Enhanced G-channel extraction with adaptive illumination correction"""
    g_channel = image[:, :, 1]

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

# Load the model
model = tf.keras.models.load_model('final_cataract_model.h5')

# Load the pre-trained ResNet50V2 model (without the top classification layer)
from tensorflow import keras
from keras._tf_keras.keras.applications import ResNet50V2
from keras._tf_keras.keras.models import Model as KerasModel
from keras._tf_keras.keras.layers import GlobalAveragePooling2D
base_model = ResNet50V2(weights='imagenet', include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
resnet_feature_extractor = KerasModel(inputs=base_model.input, outputs=x)
resnet_feature_extractor.trainable = False # Freeze the layers

# Load the image for testing
img = cv2.imread(IMAGE_PATH)
if img is None:
    print(f"Error loading image: {IMAGE_PATH}")
else:
    #Convert image to numpy array
    img = np.array(img)

    # Preprocessing
    resized_image = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))  # Resize the image
    g_channel = extract_g_channel(resized_image) #Extract G-Channel
    enhanced = top_bottom_hat_transform(g_channel) #Apply Top Bottom Hat
    filtered = trilateral_filter(enhanced) #Apply Trilateral Filter

    # Feature Extraction
    lum_features = extract_luminance_features(filtered) #Extract Luminance feature
    resnet_input = preprocess_for_resnet(filtered) #Prepare Image for Resnet

    # Reshape Image
    resnet_input = np.expand_dims(resnet_input, axis=0)

    #Call Resnet-18(Feature Extraction)
    resnet_features = resnet_feature_extractor.predict(resnet_input)
    lstm_input = prepare_features_for_lstm(lum_features, resnet_features) #Call LSTM

    # Perform the prediction
    prediction = model.predict(lstm_input)
    # Print the prediction
    print(f"Prediction: {prediction[0][0]}")