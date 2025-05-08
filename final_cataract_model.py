import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.layers import Input, LSTM, Dense, Dropout, Flatten
from keras._tf_keras.keras.models import Model
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras._tf_keras.keras.layers import Attention
import numpy as np

def create_lstm_model(input_shape, num_classes=1):
    """Create LSTM model with attention mechanism"""
    inputs = Input(shape=input_shape)

    # LSTM layer
    lstm_out = LSTM(128, return_sequences=True)(inputs)
    lstm_out = Dropout(0.3)(lstm_out)

    # Attention mechanism
    attention = Attention()([lstm_out, lstm_out])
    attention = Flatten()(attention)

    # Classification head
    dense = Dense(64, activation='relu')(attention)
    dense = Dropout(0.3)(dense)
    outputs = Dense(num_classes, activation='sigmoid')(dense)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model