# Eye-Cataract-Detection-App
A deep learning-powered web application that detects cataracts from retinal fundus images using deep learning techniques. This project combines computer vision with web development to create an accessible tool for early cataract detection.

# Key Features
🖥️ User-friendly web interface built with Flask and HTML/CSS

🧠 Deep learning model trained on 12,198 retinal images (ODIR-5K dataset)

🔍 Image analysis that provides cataract probability with visual feedback

🚀 Easy deployment with clear documentation for local setup

📊 Model training scripts included for customization and improvement

# Technology Stack

# Machine Learning
Python 3.x

TensorFlow/Keras

OpenCV (for image processing)

NumPy, Pandas (data handling)


# Model's Description
Preprocessing techniques :
1. G-Channel Extraction
3. Top Bottom Hat Transformation
4. Trilateral Filtering

Feature Extraction Techniques:
1. Extract Luminance Features
2. Pre-trained CNN (ResNet50)

Classification Techniques:
1. LSTM
2. Attention Mechanism


# Web Application
Flask (backend framework)

HTML5, CSS3 (frontend)

JavaScript (for interactive elements)

# Development Tools
VS Code (IDE)

Git (version control)

Virtual Environment (for dependency management)

# Dataset
The model was trained on the ODIR-5K dataset containing 12,198 high-resolution fundus images, carefully labeled for cataract presence. The dataset includes diverse samples to ensure robust performance across different patient demographics.

# How It Works
User uploads a retinal fundus image through the web interface

Preprocessing prepares the image for analysis

Deep learning model evaluates the image for cataract indicators

Results display shows prediction (Normal/Cataract) with confidence percentage

# Potential Applications
Clinical decision support for ophthalmologists

Telemedicine platforms for remote diagnosis

Medical education tool for students

Community health screening programs
