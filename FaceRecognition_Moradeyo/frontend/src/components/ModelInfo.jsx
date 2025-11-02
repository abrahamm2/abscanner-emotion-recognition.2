import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { FaBrain, FaCheckCircle, FaTimesCircle, FaCog } from 'react-icons/fa';
import { getModelInfo } from '../utils/api';
import LoadingSpinner from './LoadingSpinner';

const ModelInfo = () => {
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo();
  }, []);

  const fetchModelInfo = async () => {
    try {
      setLoading(true);
      const response = await getModelInfo();
      setModelInfo(response.model_info);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto" id="info">
        <div className="card">
          <LoadingSpinner message="Loading model information..." />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto" id="info">
        <div className="card bg-red-50 border-red-200">
          <p className="text-red-600">{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto" id="info">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Model Information
        </h2>

        {/* Model Status Card */}
        <div className="card mb-6">
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-3">
              <FaBrain className="text-4xl text-blue-600" />
              <div>
                <h3 className="text-2xl font-bold text-gray-800">
                  Emotion Recognition Model
                </h3>
                <p className="text-gray-600">Deep Learning CNN Architecture</p>
              </div>
            </div>
            <div>
              {modelInfo?.model_loaded ? (
                <div className="flex items-center space-x-2 bg-green-100 text-green-800 px-4 py-2 rounded-full">
                  <FaCheckCircle />
                  <span className="font-semibold">Active</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2 bg-red-100 text-red-800 px-4 py-2 rounded-full">
                  <FaTimesCircle />
                  <span className="font-semibold">Not Loaded</span>
                </div>
              )}
            </div>
          </div>

          {/* Model Details */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Model Path</p>
              <p className="font-mono text-sm text-gray-800 break-all">
                {modelInfo?.model_path || 'N/A'}
              </p>
            </div>

            <div className="bg-gray-50 p-4 rounded-lg">
              <p className="text-sm text-gray-600 mb-1">Input Size</p>
              <p className="font-mono text-sm text-gray-800">
                {modelInfo?.input_size?.[0]} x {modelInfo?.input_size?.[1]} pixels
              </p>
            </div>
          </div>
        </div>

        {/* Supported Emotions */}
        <div className="card mb-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <FaCog className="mr-2 text-blue-600" />
            Supported Emotions
          </h3>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
            {modelInfo?.emotion_labels?.map((emotion, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: index * 0.05 }}
                className="bg-gradient-to-br from-blue-50 to-cyan-50 p-3 rounded-lg text-center border border-blue-200"
              >
                <span className="text-2xl mb-1 block">{emotion}</span>
              </motion.div>
            ))}
          </div>
        </div>

        {/* Technical Details */}
        <div className="card">
          <h3 className="text-xl font-bold text-gray-800 mb-4">Technical Details</h3>
          <div className="space-y-3 text-gray-700">
            <div className="flex justify-between border-b border-gray-200 pb-2">
              <span className="font-medium">Framework:</span>
              <span>TensorFlow / Keras</span>
            </div>
            <div className="flex justify-between border-b border-gray-200 pb-2">
              <span className="font-medium">Architecture:</span>
              <span>Convolutional Neural Network (CNN)</span>
            </div>
            <div className="flex justify-between border-b border-gray-200 pb-2">
              <span className="font-medium">Face Detection:</span>
              <span>OpenCV Haar Cascade</span>
            </div>
            <div className="flex justify-between border-b border-gray-200 pb-2">
              <span className="font-medium">Number of Classes:</span>
              <span>{modelInfo?.emotion_labels?.length || 7}</span>
            </div>
            <div className="flex justify-between">
              <span className="font-medium">Image Preprocessing:</span>
              <span>Grayscale + Normalization</span>
            </div>
          </div>
        </div>

        {/* How It Works */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mt-6 card bg-gradient-to-br from-blue-50 to-cyan-50 border-blue-200"
        >
          <h3 className="text-xl font-bold text-gray-800 mb-4">How It Works</h3>
          <ol className="space-y-3 text-gray-700">
            <li className="flex items-start">
              <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0 font-bold">
                1
              </span>
              <span>
                <strong>Face Detection:</strong> OpenCV Haar Cascade detects faces in the image
              </span>
            </li>
            <li className="flex items-start">
              <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0 font-bold">
                2
              </span>
              <span>
                <strong>Preprocessing:</strong> Detected faces are converted to grayscale and resized to 48x48 pixels
              </span>
            </li>
            <li className="flex items-start">
              <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0 font-bold">
                3
              </span>
              <span>
                <strong>Prediction:</strong> The CNN model analyzes facial features and predicts emotion probabilities
              </span>
            </li>
            <li className="flex items-start">
              <span className="bg-blue-600 text-white rounded-full w-6 h-6 flex items-center justify-center mr-3 flex-shrink-0 font-bold">
                4
              </span>
              <span>
                <strong>Results:</strong> The emotion with the highest probability is returned along with confidence scores
              </span>
            </li>
          </ol>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default ModelInfo;
