"""
Abscanner - Face Detection and Emotion Classification

This module handles:
1. Face detection using OpenCV's Haar Cascade
2. Emotion prediction using the trained CNN model
"""

import cv2
import numpy as np
from tensorflow import keras
import os


class FaceEmotionDetector:
    """Class to detect faces and predict emotions"""
    
    def __init__(self, model_path='models/emotion_model.h5', 
                 cascade_path='haarcascade_frontalface_default.xml'):
        """
        Initialize the face and emotion detector
        
        Args:
            model_path: Path to trained emotion recognition model
            cascade_path: Path to Haar Cascade XML file for face detection
        """
        # Emotion labels corresponding to model output
        self.emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        
        # Load the trained emotion model
        if os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"✅ Emotion model loaded from: {model_path}")
        else:
            raise FileNotFoundError(f"❌ Model file not found: {model_path}")
        
        # Load Haar Cascade for face detection
        if os.path.exists(cascade_path):
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            print(f"✅ Face cascade loaded from: {cascade_path}")
        else:
            # Try to load from OpenCV data directory
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
            print("✅ Face cascade loaded from OpenCV data")
        
        # Image size expected by the model
        self.img_size = 48
    
    
    def detect_faces(self, image):
        """
        Detect faces in an image
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            List of face coordinates (x, y, w, h)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,      # Scale down image for detection
            minNeighbors=5,        # Minimum neighbors for valid detection
            minSize=(30, 30),      # Minimum face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    
    def predict_emotion(self, face_roi):
        """
        Predict emotion from a face region of interest
        
        Args:
            face_roi: Face region (grayscale image)
            
        Returns:
            Dictionary with emotion label, confidence, and all probabilities
        """
        # Resize to model input size
        face_resized = cv2.resize(face_roi, (self.img_size, self.img_size))
        
        # Normalize pixel values to [0, 1]
        face_normalized = face_resized / 255.0
        
        # Reshape for model input: (1, 48, 48, 1)
        face_input = face_normalized.reshape(1, self.img_size, self.img_size, 1)
        
        # Predict emotion probabilities
        predictions = self.model.predict(face_input, verbose=0)
        
        # Get the emotion with highest probability
        emotion_idx = np.argmax(predictions[0])
        emotion_label = self.emotion_labels[emotion_idx]
        confidence = predictions[0][emotion_idx] * 100
        
        # Create dictionary of all emotions and their probabilities
        all_emotions = {
            self.emotion_labels[i]: float(predictions[0][i] * 100)
            for i in range(len(self.emotion_labels))
        }
        
        return {
            'emotion': emotion_label,
            'confidence': confidence,
            'all_emotions': all_emotions
        }
    
    
    def detect_and_predict(self, image, draw_results=True):
        """
        Detect faces and predict emotions in one go
        
        Args:
            image: Input image (BGR format)
            draw_results: Whether to draw boxes and labels on image
            
        Returns:
            Dictionary with results and processed image
        """
        # Make a copy to draw on
        output_image = image.copy()
        
        # Convert to grayscale for processing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.detect_faces(image)
        
        results = []
        
        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract face region
            face_roi = gray[y:y+h, x:x+w]
            
            # Predict emotion
            emotion_data = self.predict_emotion(face_roi)
            
            # Store results
            results.append({
                'box': (x, y, w, h),
                'emotion': emotion_data['emotion'],
                'confidence': emotion_data['confidence'],
                'all_emotions': emotion_data['all_emotions']
            })
            
            # Draw results on image if requested
            if draw_results:
                # Draw rectangle around face (brown color: BGR)
                cv2.rectangle(output_image, (x, y), (x+w, y+h), (42, 85, 139), 2)
                
                # Prepare text
                emotion_text = f"{emotion_data['emotion']}"
                confidence_text = f"{emotion_data['confidence']:.1f}%"
                
                # Draw background for text (black with transparency)
                text_y = y - 10 if y > 30 else y + h + 25
                cv2.rectangle(output_image, (x, text_y - 25), (x + w, text_y + 5), (0, 0, 0), -1)
                
                # Draw emotion label
                cv2.putText(output_image, emotion_text, (x + 5, text_y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Draw confidence
                cv2.putText(output_image, confidence_text, (x + 5, text_y + 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)
        
        return {
            'faces_detected': len(faces),
            'results': results,
            'processed_image': output_image
        }


def test_detector():
    """Test function for the emotion detector"""
    print("=" * 60)
    print("🧪 Testing Abscanner Emotion Detector")
    print("=" * 60)
    
    # Initialize detector
    detector = FaceEmotionDetector()
    
    # Test with webcam
    print("\n📹 Opening webcam... (Press 'q' to quit)")
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect and predict
        result = detector.detect_and_predict(frame)
        
        # Display results
        cv2.imshow('Abscanner - Emotion Detection', result['processed_image'])
        
        # Print results to console
        if result['faces_detected'] > 0:
            for i, res in enumerate(result['results']):
                print(f"Face {i+1}: {res['emotion']} ({res['confidence']:.1f}%)")
        
        # Break on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("\n✅ Test completed!")


if __name__ == '__main__':
    test_detector()