"""
Create a simple demo model for immediate testing
This is a minimal model that allows the app to run without full training
"""

import os
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models

def create_demo_model():
    """
    Create a simple CNN model with random weights for demo purposes
    This allows testing the app before training on real data
    """
    print("Creating demo emotion recognition model...")

    # Create directories if they don't exist
    os.makedirs('models', exist_ok=True)

    # Build a simple CNN model (same architecture as full model but smaller)
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(48, 48, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(7, activation='softmax')
    ])

    # Compile the model
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print(model.summary())

    # Save the model
    model_path = 'models/emotion_model.h5'
    model.save(model_path)

    print(f"\n✅ Demo model created successfully at: {model_path}")
    print("\n⚠️  NOTE: This is a demo model with random weights.")
    print("For accurate predictions, train the model using model_training.py")
    print("with the FER2013 dataset or your custom dataset.")

    return model_path

if __name__ == '__main__':
    create_demo_model()
