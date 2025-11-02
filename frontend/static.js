// Abscanner - Frontend JavaScript

let stream = null;
let isWebcamActive = false;

// DOM Elements
const webcamBtn = document.getElementById('webcamBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const webcamContainer = document.getElementById('webcamContainer');
const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const captureBtn = document.getElementById('captureBtn');
const stopWebcamBtn = document.getElementById('stopWebcamBtn');
const previewContainer = document.getElementById('previewContainer');
const previewImage = document.getElementById('previewImage');
const resultsSection = document.getElementById('resultsSection');
const loadingSpinner = document.getElementById('loadingSpinner');
const errorMessage = document.getElementById('errorMessage');
const errorText = document.getElementById('errorText');
const closeErrorBtn = document.getElementById('closeErrorBtn');
const tryAgainBtn = document.getElementById('tryAgainBtn');

// Emotion icons mapping
const emotionIcons = {
    'Happy': '😊',
    'Sad': '😢',
    'Angry': '😠',
    'Surprise': '😲',
    'Fear': '😨',
    'Disgust': '😖',
    'Neutral': '😐'
};

// Event Listeners
webcamBtn.addEventListener('click', startWebcam);
uploadBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileUpload);
captureBtn.addEventListener('click', captureImage);
stopWebcamBtn.addEventListener('click', stopWebcam);
closeErrorBtn.addEventListener('click', hideError);
tryAgainBtn.addEventListener('click', resetApp);

// Start Webcam
async function startWebcam() {
    try {
        hideError();
        hideResults();
        previewContainer.style.display = 'none';
        
        stream = await navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: 640, 
                height: 480 
            } 
        });
        
        webcam.srcObject = stream;
        webcamContainer.style.display = 'block';
        isWebcamActive = true;
        
    } catch (error) {
        showError('Could not access webcam. Please check permissions or try uploading an image.');
        console.error('Webcam error:', error);
    }
}

// Stop Webcam
function stopWebcam() {
    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        webcam.srcObject = null;
        webcamContainer.style.display = 'none';
        isWebcamActive = false;
    }
}

// Capture Image from Webcam
function captureImage() {
    const context = canvas.getContext('2d');
    canvas.width = webcam.videoWidth;
    canvas.height = webcam.videoHeight;
    context.drawImage(webcam, 0, 0);
    
    // Convert canvas to blob and send for prediction
    canvas.toBlob(blob => {
        const reader = new FileReader();
        reader.onloadend = () => {
            const base64data = reader.result.split(',')[1];
            predictEmotion(base64data, 'webcam');
        };
        reader.readAsDataURL(blob);
    }, 'image/jpeg');
}

// Handle File Upload
function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;
    
    // Validate file type
    const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif'];
    if (!validTypes.includes(file.type)) {
        showError('Please upload a valid image file (JPEG, PNG, or GIF)');
        return;
    }
    
    // Validate file size (max 16MB)
    if (file.size > 16 * 1024 * 1024) {
        showError('File size too large. Maximum 16MB allowed.');
        return;
    }
    
    hideError();
    hideResults();
    stopWebcam();
    
    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        previewImage.src = e.target.result;
        previewContainer.style.display = 'block';
    };
    reader.readAsDataURL(file);
    
    // Send for prediction
    const formData = new FormData();
    formData.append('file', file);
    
    showLoading();
    
    fetch('/predict', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error processing image. Please try again.');
        console.error('Prediction error:', error);
    });
}

// Predict Emotion
function predictEmotion(base64Image, source) {
    hideError();
    showLoading();
    
    if (source === 'webcam') {
        stopWebcam();
    }
    
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded',
        },
        body: `image=${encodeURIComponent(base64Image)}`
    })
    .then(response => response.json())
    .then(data => {
        hideLoading();
        if (data.error) {
            showError(data.error);
        } else {
            displayResults(data);
        }
    })
    .catch(error => {
        hideLoading();
        showError('Error processing image. Please try again.');
        console.error('Prediction error:', error);
    });
}

// Display Results
function displayResults(data) {
    // Update emotion label and icon
    const emotionLabel = document.getElementById('emotionLabel');
    const emotionIcon = document.getElementById('emotionIcon');
    const confidenceValue = document.getElementById('confidenceValue');
    const resultImage = document.getElementById('resultImage');
    
    emotionLabel.textContent = data.emotion;
    emotionIcon.textContent = emotionIcons[data.emotion] || '😊';
    confidenceValue.textContent = `${data.confidence}%`;
    
    // Display processed image
    resultImage.src = data.processed_image;
    
    // Display all emotions as bar chart
    displayEmotionsChart(data.all_emotions);
    
    // Show results section
    resultsSection.style.display = 'block';
    
    // Scroll to results
    resultsSection.scrollIntoView({ behavior: 'smooth' });
}

// Display Emotions Bar Chart
function displayEmotionsChart(emotions) {
    const emotionsBar = document.getElementById('emotionsBar');
    emotionsBar.innerHTML = '';
    
    // Sort emotions by confidence (highest first)
    const sortedEmotions = Object.entries(emotions).sort((a, b) => b[1] - a[1]);
    
    sortedEmotions.forEach(([emotion, confidence]) => {
        const barDiv = document.createElement('div');
        barDiv.className = 'emotion-bar';
        
        barDiv.innerHTML = `
            <div class="emotion-bar-label">
                <span>${emotionIcons[emotion]} ${emotion}</span>
                <span>${confidence.toFixed(1)}%</span>
            </div>
            <div class="emotion-bar-fill-container">
                <div class="emotion-bar-fill" style="width: ${confidence}%">
                </div>
            </div>
        `;
        
        emotionsBar.appendChild(barDiv);
    });
}

// Show Loading Spinner
function showLoading() {
    loadingSpinner.style.display = 'block';
    previewContainer.style.display = 'none';
}

// Hide Loading Spinner
function hideLoading() {
    loadingSpinner.style.display = 'none';
}

// Show Error Message
function showError(message) {
    errorText.textContent = message;
    errorMessage.style.display = 'block';
    errorMessage.scrollIntoView({ behavior: 'smooth' });
}

// Hide Error Message
function hideError() {
    errorMessage.style.display = 'none';
}

// Hide Results
function hideResults() {
    resultsSection.style.display = 'none';
}

// Reset App
function resetApp() {
    hideResults();
    hideError();
    previewContainer.style.display = 'none';
    fileInput.value = '';
    
    // Scroll to top
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

// Initialize
console.log('🎭 Abscanner initialized!');
``