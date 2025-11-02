/**
 * API utility functions for communicating with Flask backend
 */

import axios from 'axios';

// Base API URL - adjust for production deployment
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

// Create axios instance with default config
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 second timeout
});

/**
 * Upload an image file for emotion detection
 * @param {File} imageFile - The image file to upload
 * @returns {Promise} Response with emotion detection results
 */
export const uploadImage = async (imageFile) => {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);

    const response = await api.post('/api/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });

    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Process a webcam frame
 * @param {string} base64Image - Base64 encoded image
 * @returns {Promise} Response with emotion detection results
 */
export const processWebcamFrame = async (base64Image) => {
  try {
    const response = await api.post('/api/webcam', {
      image: base64Image,
    });

    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Predict emotion from base64 image
 * @param {string} base64Image - Base64 encoded image
 * @returns {Promise} Response with emotion detection results
 */
export const predictEmotion = async (base64Image) => {
  try {
    const response = await api.post('/api/predict', {
      image: base64Image,
    });

    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Get model information
 * @returns {Promise} Response with model info
 */
export const getModelInfo = async () => {
  try {
    const response = await api.get('/api/model-info');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Check API health
 * @returns {Promise} Response with health status
 */
export const checkHealth = async () => {
  try {
    const response = await api.get('/health');
    return response.data;
  } catch (error) {
    throw handleApiError(error);
  }
};

/**
 * Handle API errors consistently
 * @param {Error} error - The error object
 * @returns {Error} Formatted error object
 */
const handleApiError = (error) => {
  if (error.response) {
    // Server responded with error status
    const message = error.response.data?.message || 'Server error occurred';
    return new Error(message);
  } else if (error.request) {
    // Request made but no response received
    return new Error('Cannot connect to server. Please check if the backend is running.');
  } else {
    // Something else happened
    return new Error(error.message || 'An unexpected error occurred');
  }
};

/**
 * Convert file to base64
 * @param {File} file - The file to convert
 * @returns {Promise<string>} Base64 encoded string
 */
export const fileToBase64 = (file) => {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => resolve(reader.result);
    reader.onerror = (error) => reject(error);
  });
};

/**
 * Get emotion color class based on emotion label
 * @param {string} emotion - The emotion label
 * @returns {string} Tailwind color class
 */
export const getEmotionColor = (emotion) => {
  const emotionLower = emotion.toLowerCase();

  if (emotionLower.includes('happy')) return 'emotion-happy';
  if (emotionLower.includes('sad')) return 'emotion-sad';
  if (emotionLower.includes('angry')) return 'emotion-angry';
  if (emotionLower.includes('surprised')) return 'emotion-surprised';
  if (emotionLower.includes('neutral')) return 'emotion-neutral';
  if (emotionLower.includes('fear')) return 'emotion-fear';
  if (emotionLower.includes('disgust')) return 'emotion-disgust';

  return 'emotion-neutral';
};

export default api;
