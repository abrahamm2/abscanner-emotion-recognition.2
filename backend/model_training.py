"""
Abscanner - Model Training Script

This script trains a Convolutional Neural Network (CNN) to recognize emotions
from facial images. The model uses the FER-2013 dataset structure.

Emotions recognized: angry, disgust, fear, happy, sad, surprise, neutral
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt


class EmotionModelTrainer:
    """Class to handle emotion recognition model training"""
    
    def __init__(self, img_size=48, batch_size=32):
        """
        Initialize the trainer
        
        Args:
            img_size: Size of input images (default: 48x48 pixels)
            batch_size: Number of images per training batch
        """
        self.img_size = img_size
        self.batch_size = batch_size
        self.num_classes = 7  # 7 emotion categories
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
    
    
    def create_data_generators(self, train_dir='datasets/train', test_dir='datasets/test'):
        """
        Create data generators for training and validation
        
        Args:
            train_dir: Path to training data directory
            test_dir: Path to test data directory
            
        Returns:
            train_generator, test_generator
        """
        # Data augmentation for training (helps prevent overfitting)
        train_datagen = ImageDataGenerator(
            rescale=1./255,              # Normalize pixel values to [0, 1]
            rotation_range=15,            # Randomly rotate images
            width_shift_range=0.1,        # Randomly shift width
            height_shift_range=0.1,       # Randomly shift height
            shear_range=0.1,              # Shear transformation
            zoom_range=0.1,               # Random zoom
            horizontal_flip=True,         # Random horizontal flip
            fill_mode='nearest',          # Fill empty pixels
            validation_split=0.2          # Use 20% of training data for validation
        )
        
        # Only rescaling for test data (no augmentation)
        test_datagen = ImageDataGenerator(rescale=1./255)
        
        # Load training data
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            subset='training'
        )
        
        # Load validation data
        validation_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical',
            subset='validation'
        )
        
        # Load test data
        test_generator = test_datagen.flow_from_directory(
            test_dir,
            target_size=(self.img_size, self.img_size),
            batch_size=self.batch_size,
            color_mode='grayscale',
            class_mode='categorical'
        )
        
        return train_generator, validation_generator, test_generator
    
    
    def build_model(self):
        """
        Build the CNN architecture for emotion recognition
        
        Returns:
            Compiled Keras model
        """
        model = keras.Sequential([
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), padding='same', activation='relu', 
                         input_shape=(self.img_size, self.img_size, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Fourth Convolutional Block
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(256, (3, 3), padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.25),
            
            # Flatten and Dense Layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            
            # Output layer (7 emotions)
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        # Compile the model
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    
    def train(self, epochs=50):
        """
        Train the emotion recognition model
        
        Args:
            epochs: Number of training epochs
            
        Returns:
            Training history
        """
        print("=" * 60)
        print("🎭 Abscanner - Training Emotion Recognition Model")
        print("=" * 60)
        
        # Create data generators
        print("\n📂 Loading datasets...")
        train_gen, val_gen, test_gen = self.create_data_generators()
        
        print(f"✅ Training samples: {train_gen.samples}")
        print(f"✅ Validation samples: {val_gen.samples}")
        print(f"✅ Test samples: {test_gen.samples}")
        
        # Build model
        print("\n🏗️  Building model architecture...")
        model = self.build_model()
        model.summary()
        
        # Callbacks for training
        callbacks = [
            # Save best model
            ModelCheckpoint(
                'models/emotion_model.h5',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            ),
            
            # Stop training if no improvement
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
                verbose=1
            ),
            
            # Reduce learning rate when stuck
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=0.00001,
                verbose=1
            )
        ]
        
        # Train the model
        print("\n🚀 Starting training...")
        history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        test_loss, test_accuracy = model.evaluate(test_gen)
        print(f"\n✅ Test Accuracy: {test_accuracy * 100:.2f}%")
        print(f"✅ Test Loss: {test_loss:.4f}")
        
        # Save final model
        model.save('models/emotion_model_final.h5')
        print("\n💾 Model saved to 'models/emotion_model.h5'")
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    
    def plot_training_history(self, history):
        """
        Plot training and validation accuracy/loss
        
        Args:
            history: Training history object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('models/training_history.png')
        print("📈 Training plots saved to 'models/training_history.png'")


def main():
    """Main function to train the model"""
    # Initialize trainer
    trainer = EmotionModelTrainer(img_size=48, batch_size=32)
    
    # Train the model
    history = trainer.train(epochs=50)
    
    print("\n" + "=" * 60)
    print("✅ Training completed successfully!")
    print("=" * 60)


if __name__ == '__main__':
    main()