"""
Abscanner - Backend-Frontend Connection Utilities

This module provides utility functions to connect the backend ML logic
with the frontend web interface.
"""

import cv2
import numpy as np
import base64
from datetime import datetime


# Global statistics storage (in production, use a database)
emotion_stats = {
    'total_predictions': 0,
    'emotions_count': {
        'Angry': 0,
        'Disgust': 0,
        'Fear': 0,
        'Happy': 0,
        'Sad': 0,
        'Surprise': 0,
        'Neutral': 0
    },
    'last_prediction': None
}


def process_image(image, detector):
    """
    Process an image and return emotion prediction results
    
    Args:
        image: OpenCV image (BGR format)
        detector: FaceEmotionDetector instance
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Detect faces and predict emotions
        result = detector.detect_and_predict(image, draw_results=True)
        
        if result['faces_detected'] == 0:
            return {
                'success': False,
                'message': 'No face detected in the image. Please try again with a clear face photo.'
            }
        
        # Get the first face's emotion (main face)
        main_result = result['results'][0]
        emotion = main_result['emotion']
        confidence = main_result['confidence']
        all_emotions = main_result['all_emotions']
        
        # Update statistics
        update_stats(emotion)
        
        # Convert processed image to base64 for web display
        processed_image_base64 = image_to_base64(result['processed_image'])
        
        return {
            'success': True,
            'emotion': emotion,
            'confidence': round(confidence, 2),
            'all_emotions': {k: round(v, 2) for k, v in all_emotions.items()},
            'faces_detected': result['faces_detected'],
            'processed_image': processed_image_base64
        }
        
    except Exception as e:
        return {
            'success': False,
            'message': f'Error processing image: {str(e)}'
        }


def image_to_base64(image):
    """
    Convert OpenCV image to base64 string for web display
    
    Args:
        image: OpenCV image (BGR format)
        
    Returns:
        Base64 encoded string
    """
    # Encode image as JPEG
    _, buffer = cv2.imencode('.jpg', image)
    
    # Convert to base64
    image_base64 = base64.b64encode(buffer).decode('utf-8')
    
    # Return as data URL
    return f"data:image/jpeg;base64,{image_base64}"


def update_stats(emotion):
    """
    Update global emotion statistics
    
    Args:
        emotion: Detected emotion label
    """
    global emotion_stats
    
    emotion_stats['total_predictions'] += 1
    emotion_stats['emotions_count'][emotion] += 1
    emotion_stats['last_prediction'] = {
        'emotion': emotion,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }


def get_emotion_stats():
    """
    Get current emotion detection statistics
    
    Returns:
        Dictionary with statistics
    """
    global emotion_stats
    
    # Calculate percentages
    total = emotion_stats['total_predictions']
    percentages = {}
    
    if total > 0:
        for emotion, count in emotion_stats['emotions_count'].items():
            percentages[emotion] = round((count / total) * 100, 2)
    else:
        percentages = {emotion: 0 for emotion in emotion_stats['emotions_count'].keys()}
    
    return {
        'total_predictions': total,
        'emotions_count': emotion_stats['emotions_count'],
        'percentages': percentages,
        'last_prediction': emotion_stats['last_prediction']
    }


def validate_image(image):
    """
    Validate if an image is suitable for processing
    
    Args:
        image: OpenCV image
        
    Returns:
        Tuple (is_valid, message)
    """
    if image is None:
        return False, "Invalid image data"
    
    # Check image dimensions
    height, width = image.shape[:2]
    
    if height < 48 or width < 48:
        return False, "Image too small (minimum 48x48 pixels)"
    
    if height > 4000 or width > 4000:
        return False, "Image too large (maximum 4000x4000 pixels)"
    
    return True, "Valid image"


def get_emotion_description(emotion):
    """
    Get a friendly description for each emotion
    
    Args:
        emotion: Emotion label
        
    Returns:
        Description string
    """
    descriptions = {
        'Happy': 'You look joyful and content! 😊',
        'Sad': 'You seem a bit down. Hope things get better! 😢',
        'Angry': 'You appear upset or frustrated. 😠',
        'Surprise': 'Something unexpected caught your attention! 😲',
        'Fear': 'You look worried or anxious. 😨',
        'Disgust': 'Something doesn\'t seem right to you. 😖',
        'Neutral': 'You have a calm and composed expression. 😐'
    }
    
    return descriptions.get(emotion, 'Emotion detected!')


def get_emotion_color(emotion):
    """
    Get color code for each emotion (for UI visualization)
    
    Args:
        emotion: Emotion label
        
    Returns:
        Hex color code
    """
    colors = {
        'Happy': '#FFD700',      # Gold
        'Sad': '#4169E1',        # Royal Blue
        'Angry': '#DC143C',      # Crimson
        'Surprise': '#FF69B4',   # Hot Pink
        'Fear': '#8B008B',       # Dark Magenta
        'Disgust': '#228B22',    # Forest Green
        'Neutral': '#808080'     # Gray
    }
    
    return colors.get(emotion, '#000000')