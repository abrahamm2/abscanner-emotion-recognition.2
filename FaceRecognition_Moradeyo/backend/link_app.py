"""
Link Application Module
Provides a clean interface between Flask routes and ML/detection logic
Ensures modularity and easy model replacement
"""

import os
import cv2
import numpy as np
from datetime import datetime
from face_emotions import FaceEmotionDetector


class EmotionRecognitionService:
    """Service class to handle emotion recognition operations"""

    def __init__(self, model_path='models/emotion_model.h5', upload_dir='uploads', dataset_dir='datasets'):
        """
        Initialize the emotion recognition service

        Args:
            model_path: Path to trained model
            upload_dir: Directory to store uploaded images
            dataset_dir: Directory to store labeled images for retraining
        """
        self.model_path = model_path
        self.upload_dir = upload_dir
        self.dataset_dir = dataset_dir

        # Create directories if they don't exist
        os.makedirs(upload_dir, exist_ok=True)
        os.makedirs(dataset_dir, exist_ok=True)

        # Initialize emotion detector
        self.detector = FaceEmotionDetector(model_path=model_path)

    def process_uploaded_image(self, image_file):
        """
        Process an uploaded image file

        Args:
            image_file: Flask file object from request.files

        Returns:
            Dictionary containing prediction results and file info
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"upload_{timestamp}_{image_file.filename}"
            filepath = os.path.join(self.upload_dir, filename)

            # Save uploaded file
            image_file.save(filepath)

            # Perform emotion detection
            results = self.detector.detect_and_predict(image_path=filepath)

            if results['success']:
                # Save to dataset directory for future retraining
                primary_emotion = results['faces'][0]['emotion'].split()[0].lower()
                self._save_to_dataset(filepath, primary_emotion)

                # Add file info to results
                results['file_info'] = {
                    'filename': filename,
                    'filepath': filepath,
                    'timestamp': timestamp
                }

            return results

        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing image: {str(e)}',
                'error': str(e)
            }

    def process_webcam_frame(self, image_data):
        """
        Process a webcam frame sent as base64 or binary data

        Args:
            image_data: Image data (numpy array or bytes)

        Returns:
            Dictionary containing prediction results
        """
        try:
            # Convert to numpy array if needed
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data

            # Perform emotion detection
            results = self.detector.detect_and_predict(image_array=image)

            if results['success']:
                # Optionally save webcam capture
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"webcam_{timestamp}.jpg"
                filepath = os.path.join(self.upload_dir, filename)
                cv2.imwrite(filepath, image)

                # Save to dataset directory
                primary_emotion = results['faces'][0]['emotion'].split()[0].lower()
                self._save_to_dataset(filepath, primary_emotion)

            return results

        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing webcam frame: {str(e)}',
                'error': str(e)
            }

    def process_base64_image(self, base64_string):
        """
        Process a base64 encoded image

        Args:
            base64_string: Base64 encoded image string

        Returns:
            Dictionary containing prediction results
        """
        try:
            import base64

            # Remove data URL prefix if present
            if ',' in base64_string:
                base64_string = base64_string.split(',')[1]

            # Decode base64 to bytes
            image_bytes = base64.b64decode(base64_string)

            # Convert to numpy array
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if image is None:
                raise ValueError("Failed to decode image")

            # Process the image
            return self.process_webcam_frame(image)

        except Exception as e:
            return {
                'success': False,
                'message': f'Error processing base64 image: {str(e)}',
                'error': str(e)
            }

    def _save_to_dataset(self, image_path, emotion_label):
        """
        Save processed image to dataset directory for future retraining

        Args:
            image_path: Path to processed image
            emotion_label: Detected emotion label
        """
        try:
            # Create emotion subdirectory if it doesn't exist
            emotion_dir = os.path.join(self.dataset_dir, emotion_label)
            os.makedirs(emotion_dir, exist_ok=True)

            # Generate unique filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
            filename = f"{emotion_label}_{timestamp}.jpg"
            dest_path = os.path.join(emotion_dir, filename)

            # Copy image to dataset directory
            image = cv2.imread(image_path)
            if image is not None:
                cv2.imwrite(dest_path, image)

        except Exception as e:
            print(f"Warning: Could not save to dataset: {e}")

    def get_model_info(self):
        """
        Get information about the loaded model

        Returns:
            Dictionary containing model information
        """
        return {
            'model_path': self.model_path,
            'model_loaded': self.detector.model is not None,
            'emotion_labels': list(self.detector.emotion_labels.values()),
            'input_size': self.detector.img_size
        }

    def create_annotated_image(self, image_path, output_path=None):
        """
        Create an annotated version of the image with detected emotions

        Args:
            image_path: Path to input image
            output_path: Path to save annotated image (optional)

        Returns:
            Dictionary containing results and output path
        """
        try:
            if output_path is None:
                base_name = os.path.basename(image_path)
                output_path = os.path.join(self.upload_dir, f"annotated_{base_name}")

            results = self.detector.annotate_image(image_path, output_path)

            if results['success']:
                results['annotated_image_path'] = output_path

            return results

        except Exception as e:
            return {
                'success': False,
                'message': f'Error creating annotated image: {str(e)}',
                'error': str(e)
            }


# Singleton instance for the application
_service_instance = None


def get_emotion_service(model_path='models/emotion_model.h5'):
    """
    Get or create the emotion recognition service instance

    Args:
        model_path: Path to trained model

    Returns:
        EmotionRecognitionService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = EmotionRecognitionService(model_path=model_path)
    return _service_instance


def initialize_service(model_path='models/emotion_model.h5', upload_dir='uploads', dataset_dir='datasets'):
    """
    Initialize the emotion recognition service with custom configuration

    Args:
        model_path: Path to trained model
        upload_dir: Directory to store uploaded images
        dataset_dir: Directory to store labeled images

    Returns:
        EmotionRecognitionService instance
    """
    global _service_instance
    _service_instance = EmotionRecognitionService(
        model_path=model_path,
        upload_dir=upload_dir,
        dataset_dir=dataset_dir
    )
    return _service_instance


# Test function
def test_service():
    """Test the emotion recognition service"""
    service = get_emotion_service()
    info = service.get_model_info()
    print("Service initialized successfully!")
    print(f"Model Info: {info}")


if __name__ == '__main__':
    test_service()
