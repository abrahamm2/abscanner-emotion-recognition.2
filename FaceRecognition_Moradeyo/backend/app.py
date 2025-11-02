

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import sys
import base64
from datetime import datetime

# Import our custom modules
from link_app import get_emotion_service, initialize_service


# Initialize Flask app
app = Flask(__name__)

# Enable CORS for frontend communication
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}

# Initialize emotion recognition service
emotion_service = initialize_service(
    model_path='models/emotion_model.h5',
    upload_dir='uploads',
    dataset_dir='datasets'
)


def allowed_file(filename):
    """
    Check if file extension is allowed

    Args:
        filename: Name of the file

    Returns:
        Boolean indicating if file is allowed
    """
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Root endpoint - API status"""
    return jsonify({
        'status': 'running',
        'message': 'Emotion Recognition API is active',
        'version': '1.0.0',
        'endpoints': {
            '/': 'API status',
            '/health': 'Health check',
            '/api/upload': 'Upload image for emotion detection',
            '/api/webcam': 'Process webcam frame',
            '/api/predict': 'Predict emotion from base64 image',
            '/api/model-info': 'Get model information',
            '/uploads/<filename>': 'Access uploaded files'
        }
    })


@app.route('/health')
def health_check():
    """Health check endpoint"""
    model_info = emotion_service.get_model_info()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_info['model_loaded']
    })


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """
    Upload and process an image file

    Expected: multipart/form-data with 'image' field
    Returns: JSON with emotion detection results
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided',
                'error': 'Missing image field'
            }), 400

        file = request.files['image']

        # Check if file is selected
        if file.filename == '':
            return jsonify({
                'success': False,
                'message': 'No file selected',
                'error': 'Empty filename'
            }), 400

        # Check if file type is allowed
        if not allowed_file(file.filename):
            return jsonify({
                'success': False,
                'message': f'File type not allowed. Allowed types: {", ".join(app.config["ALLOWED_EXTENSIONS"])}',
                'error': 'Invalid file type'
            }), 400

        # Process the image
        results = emotion_service.process_uploaded_image(file)

        if results['success']:
            return jsonify(results), 200
        else:
            return jsonify(results), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Server error while processing image',
            'error': str(e)
        }), 500


@app.route('/api/webcam', methods=['POST'])
def process_webcam():
    """
    Process webcam frame

    Expected: JSON with 'image' field (base64 encoded image)
    Returns: JSON with emotion detection results
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'No image data provided',
                'error': 'Missing image field'
            }), 400

        # Process base64 image
        results = emotion_service.process_base64_image(data['image'])

        if results['success']:
            return jsonify(results), 200
        else:
            return jsonify(results), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Server error while processing webcam frame',
            'error': str(e)
        }), 500


@app.route('/api/predict', methods=['POST'])
def predict_emotion():
    """
    Predict emotion from base64 encoded image

    Expected: JSON with 'image' field (base64 encoded image)
    Returns: JSON with emotion detection results
    """
    try:
        data = request.get_json()

        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': 'No image data provided',
                'error': 'Missing image field'
            }), 400

        # Process base64 image
        results = emotion_service.process_base64_image(data['image'])

        if results['success']:
            return jsonify(results), 200
        else:
            return jsonify(results), 400

    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Server error during prediction',
            'error': str(e)
        }), 500


@app.route('/api/model-info', methods=['GET'])
def model_info():
    """
    Get information about the loaded model

    Returns: JSON with model information
    """
    try:
        info = emotion_service.get_model_info()
        return jsonify({
            'success': True,
            'model_info': info
        }), 200

    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'Error getting model info',
            'error': str(e)
        }), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """
    Serve uploaded files

    Args:
        filename: Name of the file to serve

    Returns: File from uploads directory
    """
    try:
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': 'File not found',
            'error': str(e)
        }), 404


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({
        'success': False,
        'message': 'File too large. Maximum size is 16MB',
        'error': 'File size exceeded'
    }), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({
        'success': False,
        'message': 'Endpoint not found',
        'error': 'Not found'
    }), 404


@app.errorhandler(500)
def internal_error(e):
    """Handle internal server errors"""
    return jsonify({
        'success': False,
        'message': 'Internal server error',
        'error': str(e)
    }), 500


# Development server
if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('uploads', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('datasets', exist_ok=True)

    # Print startup info
    print("=" * 60)
    print("🎭 Emotion Recognition API Server")
    print("=" * 60)
    print(f"Server starting on http://127.0.0.1:5000")
    print(f"Model loaded: {emotion_service.get_model_info()['model_loaded']}")
    print("=" * 60)

    # Run Flask app
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )
