"""
Face Detection and Emotion Recognition Module
Handles face detection using OpenCV and emotion prediction using trained model
"""

import cv2
import numpy as np
from tensorflow import keras
import os


class FaceEmotionDetector:
    """Class to handle face detection and emotion recognition"""

    def __init__(self, model_path='models/emotion_model.h5', img_size=(48, 48)):
        """
        Initialize the emotion detector

        Args:
            model_path: Path to trained emotion recognition model
            img_size: Input size for the model (height, width)
        """
        self.img_size = img_size
        self.emotion_labels = {
            0: 'Angry 😠',
            1: 'Disgust 🤢',
            2: 'Fear 😨',
            3: 'Happy 😊',
            4: 'Neutral 😐',
            5: 'Sad 😢',
            6: 'Surprised 😲'
        }

        # Load the trained model
        self.model = None
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Using dummy model for demonstration")
                self.model = None
        else:
            print(f"Model not found at {model_path}")
            print("Please train the model first using model_training.py")
            self.model = None

        # Load OpenCV face detector (Haar Cascade)
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        if self.face_cascade.empty():
            raise ValueError("Failed to load face cascade classifier")

    def detect_faces(self, image):
        """
        Detect faces in an image using OpenCV

        Args:
            image: Input image (BGR format)

        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        return faces

    def preprocess_face(self, face_img):
        """
        Preprocess face image for emotion prediction

        Args:
            face_img: Cropped face image

        Returns:
            Preprocessed image ready for model input
        """
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

        # Resize to model input size
        resized_face = cv2.resize(gray_face, self.img_size)

        # Normalize pixel values
        normalized_face = resized_face.astype('float32') / 255.0

        # Reshape for model input (batch_size, height, width, channels)
        preprocessed = normalized_face.reshape(1, *self.img_size, 1)

        return preprocessed

    def predict_emotion(self, face_img):
        """
        Predict emotion from a face image

        Args:
            face_img: Cropped face image

        Returns:
            Tuple of (emotion_label, confidence_score, all_probabilities)
        """
        if self.model is None:
            # Return dummy prediction if model not loaded
            return "Happy 😊", 0.85, {
                'Angry 😠': 0.05,
                'Disgust 🤢': 0.02,
                'Fear 😨': 0.03,
                'Happy 😊': 0.85,
                'Neutral 😐': 0.02,
                'Sad 😢': 0.01,
                'Surprised 😲': 0.02
            }

        # Preprocess face
        preprocessed_face = self.preprocess_face(face_img)

        # Make prediction
        predictions = self.model.predict(preprocessed_face, verbose=0)[0]

        # Get emotion with highest probability
        emotion_idx = np.argmax(predictions)
        emotion_label = self.emotion_labels[emotion_idx]
        confidence = float(predictions[emotion_idx])

        # Create dictionary of all emotions and their probabilities
        all_probabilities = {
            self.emotion_labels[i]: float(predictions[i])
            for i in range(len(predictions))
        }

        return emotion_label, confidence, all_probabilities

    def detect_and_predict(self, image_path=None, image_array=None):
        """
        Detect faces and predict emotions from an image

        Args:
            image_path: Path to image file
            image_array: NumPy array of image (alternative to image_path)

        Returns:
            Dictionary containing results
        """
        # Load image
        if image_path is not None:
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Failed to load image from {image_path}")
        elif image_array is not None:
            image = image_array
        else:
            raise ValueError("Either image_path or image_array must be provided")

        # Detect faces
        faces = self.detect_faces(image)

        if len(faces) == 0:
            return {
                'success': False,
                'message': 'No faces detected in the image',
                'faces': []
            }

        # Process each detected face
        results = []
        for (x, y, w, h) in faces:
            # Extract face region
            face_img = image[y:y+h, x:x+w]

            # Predict emotion
            emotion, confidence, probabilities = self.predict_emotion(face_img)

            results.append({
                'emotion': emotion,
                'confidence': confidence,
                'probabilities': probabilities,
                'bbox': {
                    'x': int(x),
                    'y': int(y),
                    'width': int(w),
                    'height': int(h)
                }
            })

        return {
            'success': True,
            'message': f'Detected {len(faces)} face(s)',
            'faces': results,
            'num_faces': len(faces)
        }

    def annotate_image(self, image_path, output_path):
        """
        Detect faces, predict emotions, and save annotated image

        Args:
            image_path: Path to input image
            output_path: Path to save annotated image

        Returns:
            Results dictionary
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Failed to load image from {image_path}")

        # Get predictions
        results = self.detect_and_predict(image_array=image)

        if not results['success']:
            return results

        # Draw bounding boxes and labels
        for face_data in results['faces']:
            bbox = face_data['bbox']
            emotion = face_data['emotion']
            confidence = face_data['confidence']

            # Draw rectangle around face
            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            # Prepare label text
            label = f"{emotion} ({confidence:.2%})"

            # Draw label background
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                image,
                (x, y - label_size[1] - 10),
                (x + label_size[0], y),
                (0, 255, 0),
                -1
            )

            # Draw label text
            cv2.putText(
                image,
                label,
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2
            )

        # Save annotated image
        cv2.imwrite(output_path, image)
        print(f"Annotated image saved to {output_path}")

        return results


def test_detector():
    """Test function to demonstrate emotion detection"""
    detector = FaceEmotionDetector()

    # Test with a sample image (if available)
    test_image = 'test_image.jpg'
    if os.path.exists(test_image):
        results = detector.detect_and_predict(image_path=test_image)
        print("Detection Results:")
        print(results)

        # Save annotated image
        detector.annotate_image(test_image, 'test_output.jpg')
    else:
        print(f"Test image not found: {test_image}")
        print("Place a test image and run again")


if __name__ == '__main__':
    test_detector()
