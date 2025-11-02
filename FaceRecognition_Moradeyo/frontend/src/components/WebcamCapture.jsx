import React, { useState, useRef, useCallback } from 'react';
import { motion } from 'framer-motion';
import Webcam from 'react-webcam';
import { FaCamera, FaRedo, FaVideo, FaVideoSlash } from 'react-icons/fa';
import { processWebcamFrame } from '../utils/api';
import LoadingSpinner from './LoadingSpinner';
import ResultDisplay from './ResultDisplay';

const WebcamCapture = () => {
  const [isWebcamActive, setIsWebcamActive] = useState(false);
  const [capturedImage, setCapturedImage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const webcamRef = useRef(null);

  const videoConstraints = {
    width: 1280,
    height: 720,
    facingMode: 'user',
  };

  const handleStartWebcam = () => {
    setIsWebcamActive(true);
    setCapturedImage(null);
    setResult(null);
  };

  const handleStopWebcam = () => {
    setIsWebcamActive(false);
  };

  const captureImage = useCallback(() => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      setCapturedImage(imageSrc);
      setIsWebcamActive(false);
    }
  }, [webcamRef]);

  const handleAnalyze = async () => {
    if (!capturedImage) return;

    setLoading(true);
    setResult(null);

    try {
      const response = await processWebcamFrame(capturedImage);
      setResult(response);
    } catch (error) {
      setResult({
        success: false,
        message: error.message || 'Failed to process image',
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRetake = () => {
    setCapturedImage(null);
    setResult(null);
    setIsWebcamActive(true);
  };

  const handleReset = () => {
    setCapturedImage(null);
    setResult(null);
    setIsWebcamActive(false);
  };

  return (
    <div className="max-w-4xl mx-auto" id="webcam">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <h2 className="text-3xl font-bold text-gray-800 mb-6 text-center">
          Webcam Capture
        </h2>

        {/* Webcam Area */}
        <div className="card">
          {!isWebcamActive && !capturedImage && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="text-center py-12"
            >
              <FaVideo className="text-6xl text-gray-400 mx-auto mb-4" />
              <h3 className="text-xl font-semibold text-gray-700 mb-2">
                Start Your Webcam
              </h3>
              <p className="text-gray-500 mb-6">
                Click the button below to enable your camera and capture your emotion
              </p>
              <button
                onClick={handleStartWebcam}
                className="btn-primary inline-flex items-center space-x-2"
              >
                <FaVideo />
                <span>Enable Webcam</span>
              </button>
            </motion.div>
          )}

          {isWebcamActive && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-4"
            >
              <div className="relative rounded-lg overflow-hidden bg-black">
                <Webcam
                  ref={webcamRef}
                  audio={false}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="w-full"
                />
              </div>

              <div className="flex justify-center space-x-4">
                <button
                  onClick={captureImage}
                  className="btn-primary inline-flex items-center space-x-2"
                >
                  <FaCamera />
                  <span>Capture Photo</span>
                </button>
                <button
                  onClick={handleStopWebcam}
                  className="btn-secondary inline-flex items-center space-x-2"
                >
                  <FaVideoSlash />
                  <span>Stop Webcam</span>
                </button>
              </div>
            </motion.div>
          )}

          {capturedImage && !isWebcamActive && (
            <motion.div
              initial={{ opacity: 0, scale: 0.95 }}
              animate={{ opacity: 1, scale: 1 }}
              className="space-y-4"
            >
              <div className="relative">
                <img
                  src={capturedImage}
                  alt="Captured"
                  className="w-full rounded-lg"
                />
              </div>

              <div className="flex justify-center space-x-4">
                <button
                  onClick={handleAnalyze}
                  disabled={loading}
                  className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {loading ? 'Analyzing...' : 'Analyze Emotion'}
                </button>
                <button
                  onClick={handleRetake}
                  className="btn-secondary inline-flex items-center space-x-2"
                >
                  <FaRedo />
                  <span>Retake Photo</span>
                </button>
                <button
                  onClick={handleReset}
                  className="btn-secondary"
                >
                  Reset
                </button>
              </div>
            </motion.div>
          )}
        </div>

        {/* Loading State */}
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-6"
          >
            <div className="card">
              <LoadingSpinner message="Analyzing emotions from webcam..." />
            </div>
          </motion.div>
        )}

        {/* Results */}
        {result && !loading && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="mt-6"
          >
            <ResultDisplay result={result} />
          </motion.div>
        )}

        {/* Info Note */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
          className="mt-6 bg-blue-50 border border-blue-200 rounded-lg p-4"
        >
          <p className="text-sm text-blue-800">
            <strong>Note:</strong> Your images are processed locally on the server and
            are used to improve the model. No data is shared with third parties.
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
};

export default WebcamCapture;
