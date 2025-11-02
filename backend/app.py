c"""
Abscanner - Facial Emotion Recognition System
Main Flask Application

This file handles the web server, routes, and integrates all components.
"""

from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
import base64
from face_emotions import FaceEmotionDetector
from link_app import process_image, get_emotion_stats

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize the emotion detector
try:
    detector = FaceEmotionDetector()
    print("✅ Abscanner emotion detector initialized successfully!")
except Exception as e:
    print(f"❌ Error initializing detector: {e}")
    detector = None


def allowed_file(filename):
    """Check if uploaded file has an allowed extension"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']


@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Handle image upload or webcam capture and return emotion prediction
    
    Accepts:
    - File upload via 'file' parameter
    - Base64 image via 'image' parameter (from webcam)
    
    Returns:
    - JSON with emotion, confidence, and processed image
    """
    try:
        if detector is None:
            return jsonify({'error': 'Emotion detector not initialized'}), 500

        image = None
        
        # Check if image comes from file upload
        if 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Read the image
                image = cv2.imread(filepath)
                
        # Check if image comes from webcam (base64)
        elif 'image' in request.form:
            image_data = request.form['image']
            # Remove the data URL prefix if present
            if 'base64,' in image_data:
                image_data = image_data.split('base64,')[1]
            
            # Decode base64 to image
            image_bytes = base64.b64decode(image_data)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({'error': 'No valid image provided'}), 400
        
        # Process the image and get emotion prediction
        result = process_image(image, detector)
        
        if result['success']:
            return jsonify({
                'emotion': result['emotion'],
                'confidence': result['confidence'],
                'all_emotions': result['all_emotions'],
                'processed_image': result['processed_image'],
                'faces_detected': result['faces_detected']
            })
        else:
            return jsonify({'error': result['message']}), 400
            
    except Exception as e:
        print(f"Error in prediction: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/stats', methods=['GET'])
def stats():
    """Return statistics about emotion detection"""
    stats = get_emotion_stats()
    return jsonify(stats)


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'detector_loaded': detector is not None,
        'model_path': 'models/emotion_model.h5'
    })


if __name__ == '__main__':
    print("=" * 60)
    print("🎭 Starting Abscanner - Facial Emotion Recognition System")
    print("=" * 60)
    print("\n📍 Access the application at: http://localhost:5000")
    print("Press CTRL+C to stop the server\n")
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
     # For local development
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)