#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  7 14:26:51 2024

@author: momo
"""

# Basic Libraries
import pandas as pd
import numpy as np
import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Load Metadata
df = pd.read_csv('/Users/momo/Desktop/archive/UrbanSound8K.csv')

# Constants
FEATURE_LENGTH = 128  # Fixed feature length for consistency

# Feature Extraction Function
def extract_features_and_labels(df):
    features, labels = [], []
    for index, row in df.iterrows():
        file_path = os.path.join('/Users/momo/Desktop/archive', f"fold{row['fold']}", row['slice_file_name'])
        try:
            # Load audio file
            y, sr = librosa.load(file_path, sr=None)
            n_fft = min(2048, len(y))  # Dynamically adjust n_fft
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=n_fft, hop_length=512)
            mfccs = np.mean(mfccs.T, axis=0)  # Take mean over time axis
            
            # Ensure consistent feature length
            if len(mfccs) < FEATURE_LENGTH:
                mfccs = np.pad(mfccs, (0, FEATURE_LENGTH - len(mfccs)), mode='constant')
            elif len(mfccs) > FEATURE_LENGTH:
                mfccs = mfccs[:FEATURE_LENGTH]
            
            features.append(mfccs)
            labels.append(row['classID'])
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
    return np.array(features), np.array(labels)

# Extract features and labels
X, Y = extract_features_and_labels(df)

# One-hot encode labels
Y = to_categorical(Y, num_classes=10)

# Split data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=42)

# Reshape for CNN input
X_train = X_train.reshape(X_train.shape[0], 16, 8, 1)  # Assuming (128,) -> (16, 8)
X_test = X_test.reshape(X_test.shape[0], 16, 8, 1)

# CNN Model
input_dim = (16, 8, 1)

model = Sequential()
model.add(Conv2D(64, (3, 3), padding='same', activation='tanh', input_shape=input_dim))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), padding='same', activation='tanh'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(1024, activation='tanh'))
model.add(Dense(10, activation='softmax'))

# Compile Model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
model.fit(X_train, Y_train, epochs=90, batch_size=50, validation_data=(X_test, Y_test))

# Model Summary
model.summary()

# Evaluate Model
score = model.evaluate(X_test, Y_test)
print(f"Test Loss: {score[0]}")
print(f"Test Accuracy: {score[1]}")

# Predict and Save Results
predictions = model.predict(X_test)
preds = np.argmax(predictions, axis=1)

result = pd.DataFrame(preds, columns=['Predicted'])
result.to_csv("UrbanSound8kResults.csv", index=False)
