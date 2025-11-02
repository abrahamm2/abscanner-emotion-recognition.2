# Quick Start Guide

Get the Emotion Recognition app running in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- Node.js 16 or higher
- npm or yarn

## Step 1: Backend Setup

```bash
# Navigate to backend
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create demo model (for testing without training)
python create_demo_model.py

# Start the Flask server
python app.py
```

The backend should now be running on `http://localhost:5000`

## Step 2: Frontend Setup

Open a **new terminal** window:

```bash
# Navigate to frontend
cd frontend

# Install dependencies
npm install

# Start the development server
npm run dev
```

The frontend should now be running on `http://localhost:3000`

## Step 3: Use the App

1. Open your browser and go to `http://localhost:3000`
2. Try uploading an image or using your webcam
3. View the emotion detection results!

## Important Notes

### Demo Model vs Trained Model

The `create_demo_model.py` script creates a model with **random weights** for immediate testing. For accurate predictions:

1. **Download FER2013 dataset** from [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
2. Place `fer2013.csv` in `backend/datasets/`
3. Update `model_training.py` to use FER2013
4. Run: `python model_training.py`

Training takes 30-60 minutes but provides much better accuracy!

### Testing with Sample Images

For best results with the demo model, try:
- Clear, front-facing photos
- Good lighting
- Single face in frame
- Neutral background

## Troubleshooting

### Backend won't start
- Check if port 5000 is available
- Ensure all dependencies installed: `pip list`
- Check Python version: `python --version` (should be 3.8+)

### Frontend won't start
- Delete `node_modules` and reinstall: `rm -rf node_modules && npm install`
- Check Node version: `node --version` (should be 16+)
- Try a different port if 3000 is occupied

### Model errors
- Make sure `emotion_model.h5` exists in `backend/models/`
- Recreate demo model: `python create_demo_model.py`

### CORS errors
- Ensure backend is running on port 5000
- Check CORS settings in `backend/app.py`

## What's Next?

Once you have the app running:

1. **Train a real model** for better accuracy
2. **Customize the UI** to match your style
3. **Deploy to production** (see README.md)
4. **Add new features** (video analysis, emotion tracking, etc.)

## Need Help?

- Read the full [README.md](README.md)
- Check the [API documentation](README.md#api-endpoints)
- Open an issue on GitHub

Happy emotion detecting! 😊
