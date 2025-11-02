"""
Emotion Recognition Model Training Script
Trains a CNN model on emotion datasets (FER2013 or custom dataset)
Supports transfer learning with MobileNetV2 or ResNet50
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import cv2


class EmotionModelTrainer:
    """Class to handle emotion recognition model training"""

    def __init__(self, img_size=(48, 48), model_type='cnn'):
        """
        Initialize the trainer

        Args:
            img_size: Tuple of (height, width) for input images
            model_type: 'cnn' for custom CNN, 'mobilenet' for transfer learning
        """
        self.img_size = img_size
        self.model_type = model_type
        self.model = None
        self.history = None
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    def build_cnn_model(self, num_classes=7):
        """
        Build a custom CNN model for emotion recognition

        Args:
            num_classes: Number of emotion classes

        Returns:
            Compiled Keras model
        """
        model = models.Sequential([
            # First Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                         input_shape=(*self.img_size, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Second Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Third Convolutional Block
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fourth Convolutional Block
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),

            # Fully Connected Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def build_transfer_learning_model(self, num_classes=7):
        """
        Build a model using MobileNetV2 transfer learning

        Args:
            num_classes: Number of emotion classes

        Returns:
            Compiled Keras model
        """
        # Load pre-trained MobileNetV2
        base_model = MobileNetV2(
            input_shape=(*self.img_size, 3),
            include_top=False,
            weights='imagenet'
        )

        # Freeze base model layers
        base_model.trainable = False

        # Add custom classification head
        model = models.Sequential([
            layers.Input(shape=(*self.img_size, 3)),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dense(512, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(num_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def load_fer2013_dataset(self, csv_path):
        """
        Load FER2013 dataset from CSV file

        Args:
            csv_path: Path to FER2013 CSV file

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Loading FER2013 dataset...")
        df = pd.read_csv(csv_path)

        # Convert pixel strings to numpy arrays
        X = []
        y = []

        for index, row in df.iterrows():
            pixels = np.array([int(pixel) for pixel in row['pixels'].split()])
            pixels = pixels.reshape(48, 48)
            X.append(pixels)
            y.append(row['emotion'])

        X = np.array(X)
        y = np.array(y)

        # Reshape for CNN input
        X = X.reshape(-1, 48, 48, 1)
        X = X.astype('float32') / 255.0

        # One-hot encode labels
        y = keras.utils.to_categorical(y, num_classes=7)

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def load_custom_dataset(self, dataset_dir):
        """
        Load custom dataset from directory structure
        Expected structure: dataset_dir/emotion_name/*.jpg

        Args:
            dataset_dir: Path to dataset directory

        Returns:
            X_train, X_test, y_train, y_test
        """
        print("Loading custom dataset...")
        X = []
        y = []

        for emotion_idx, emotion in enumerate(self.emotion_labels):
            emotion_dir = os.path.join(dataset_dir, emotion)
            if not os.path.exists(emotion_dir):
                continue

            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        X.append(img)
                        y.append(emotion_idx)
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")

        if len(X) == 0:
            raise ValueError("No images found in dataset directory")

        X = np.array(X)
        y = np.array(y)

        # Reshape for CNN input
        X = X.reshape(-1, *self.img_size, 1)
        X = X.astype('float32') / 255.0

        # One-hot encode labels
        y = keras.utils.to_categorical(y, num_classes=len(self.emotion_labels))

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        print(f"Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64):
        """
        Train the emotion recognition model

        Args:
            X_train: Training images
            y_train: Training labels
            X_val: Validation images
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training

        Returns:
            Training history
        """
        # Build model
        if self.model_type == 'cnn':
            self.model = self.build_cnn_model(num_classes=y_train.shape[1])
        else:
            self.model = self.build_transfer_learning_model(num_classes=y_train.shape[1])

        print(self.model.summary())

        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            shear_range=0.1
        )
        datagen.fit(X_train)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                'models/emotion_model_best.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train model
        print("Starting training...")
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            validation_data=(X_val, y_val),
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )

        return self.history

    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set

        Args:
            X_test: Test images
            y_test: Test labels

        Returns:
            Test loss and accuracy
        """
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"\nTest Accuracy: {test_acc:.4f}")
        print(f"Test Loss: {test_loss:.4f}")
        return test_loss, test_acc

    def plot_training_history(self, save_path='models/training_history.png'):
        """
        Plot training history

        Args:
            save_path: Path to save the plot
        """
        if self.history is None:
            print("No training history available")
            return

        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Plot accuracy
        axes[0].plot(self.history.history['accuracy'], label='Train Accuracy')
        axes[0].plot(self.history.history['val_accuracy'], label='Val Accuracy')
        axes[0].set_title('Model Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        axes[0].grid(True)

        # Plot loss
        axes[1].plot(self.history.history['loss'], label='Train Loss')
        axes[1].plot(self.history.history['val_loss'], label='Val Loss')
        axes[1].set_title('Model Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
        plt.close()

    def save_model(self, model_path='models/emotion_model.h5'):
        """
        Save trained model

        Args:
            model_path: Path to save the model
        """
        if self.model is None:
            print("No model to save")
            return

        self.model.save(model_path)
        print(f"Model saved to {model_path}")


def main():
    """Main training function"""
    # Initialize trainer
    trainer = EmotionModelTrainer(img_size=(48, 48), model_type='cnn')

    # Option 1: Load FER2013 dataset (if available)
    # X_train, X_test, y_train, y_test = trainer.load_fer2013_dataset('datasets/fer2013.csv')

    # Option 2: Load custom dataset from directory
    try:
        X_train, X_test, y_train, y_test = trainer.load_custom_dataset('datasets')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("\nTo train the model, you need to either:")
        print("1. Download FER2013 dataset and use load_fer2013_dataset()")
        print("2. Add custom emotion images to datasets/emotion_name/ folders")
        return

    # Train model
    history = trainer.train(
        X_train, y_train,
        X_test, y_test,
        epochs=50,
        batch_size=64
    )

    # Evaluate model
    trainer.evaluate(X_test, y_test)

    # Plot training history
    trainer.plot_training_history()

    # Save model
    trainer.save_model('models/emotion_model.h5')

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
