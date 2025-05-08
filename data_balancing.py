import pandas as pd
import os
import ast
import cv2
import numpy as np
from PIL import Image
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator

# Constants
IMG_WIDTH = 224
IMG_HEIGHT = 224
CSV_FILE = 'full_df.csv'
IMAGE_DIR = 'ODIR-5K/Training Images'

def load_data(csv_file, image_dir):
    """Loads data from CSV, constructs image paths, and extracts labels."""
    df = pd.read_csv(csv_file)
    
    # Convert the 'target' column from string to list of integers
    df['target'] = df['target'].apply(ast.literal_eval)
    
    # Construct full image paths
    df['image_path'] = df['filename'].apply(lambda x: os.path.join(image_dir, x))
    
    # Extract the 'Cataract' label (index 3 in target list)
    df['label'] = df['target'].apply(lambda x: x[3])
    
    # Debug: Check the distribution of the 'label' column
    print("Label distribution BEFORE balancing:")
    print(df['label'].value_counts())
    
    return df

def balance_data(df, image_dir):
    """Balances the dataset by oversampling the minority class (Cataract)."""
    # Separate the Cataract and Normal images
    cataract_df = df[df['label'] == 1].copy()
    normal_df = df[df['label'] == 0].copy()

    # Number of cataract images to generate
    num_to_generate = len(normal_df) - len(cataract_df)

    if num_to_generate <= 0:
        print("Dataset is already balanced or has more cataract images than normal images.")
        return df

    # Image Data Generator for augmentation
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    new_rows = []
    for i in range(num_to_generate):
        # Select a random cataract image
        sample = cataract_df.sample(1).iloc[0]
        img_path = sample['image_path']
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image: {img_path}")
            continue
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        img = np.expand_dims(img, axis=0)

        # Generate augmented image
        aug_iter = datagen.flow(img, batch_size=1)
        aug_img = next(aug_iter)[0].astype('uint8')
        
        # Save augmented image
        new_filename = f"aug_cataract_{len(df) + i}.jpg"
        new_path = os.path.join(image_dir, new_filename)
        Image.fromarray(aug_img).save(new_path)

        # Create new row with same metadata but new image path
        new_row = sample.to_dict()
        new_row.update({
            'image_path': new_path,
            'filename': new_filename,
            'ID': f"aug_{sample['ID']}_{i}"  # Unique ID for augmented sample
        })
        new_rows.append(new_row)

    # Add new rows to dataframe
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        
        print("Label distribution AFTER balancing:")
        print(df['label'].value_counts())
        
        # Save balanced dataset
        df.to_csv(CSV_FILE, index=False)
    else:
        print("No new images were generated during augmentation.")

    return df

if __name__ == "__main__":
    df = load_data(CSV_FILE, IMAGE_DIR)
    df = balance_data(df, IMAGE_DIR)
    print("Data balancing complete. Updated CSV saved.")