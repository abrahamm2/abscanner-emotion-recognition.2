import React from 'react';
import { motion } from 'framer-motion';
import { FaCheckCircle, FaTimesCircle } from 'react-icons/fa';
import { getEmotionColor } from '../utils/api';

const ResultDisplay = ({ result }) => {
  if (!result) return null;

  const { success, message, faces, num_faces } = result;

  if (!success) {
    return (
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card bg-red-50 border-red-200"
      >
        <div className="flex items-center space-x-3">
          <FaTimesCircle className="text-red-500 text-2xl" />
          <div>
            <h3 className="text-lg font-semibold text-red-800">Error</h3>
            <p className="text-red-600">{message}</p>
          </div>
        </div>
      </motion.div>
    );
  }

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.3 }}
      className="space-y-6"
    >
      {/* Success Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card bg-green-50 border-green-200"
      >
        <div className="flex items-center space-x-3">
          <FaCheckCircle className="text-green-500 text-2xl" />
          <div>
            <h3 className="text-lg font-semibold text-green-800">Detection Complete</h3>
            <p className="text-green-600">{message}</p>
          </div>
        </div>
      </motion.div>

      {/* Results for Each Face */}
      {faces && faces.map((face, index) => (
        <motion.div
          key={index}
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: index * 0.1 }}
          className="card"
        >
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-xl font-bold text-gray-800">
              {num_faces > 1 ? `Face ${index + 1}` : 'Detected Emotion'}
            </h3>
          </div>

          {/* Primary Emotion */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 font-medium">Primary Emotion:</span>
              <motion.span
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ type: 'spring', stiffness: 200, delay: 0.2 }}
                className={`emotion-badge ${getEmotionColor(face.emotion)} text-2xl`}
              >
                {face.emotion}
              </motion.span>
            </div>
            <div className="text-right text-sm text-gray-500">
              Confidence: {(face.confidence * 100).toFixed(1)}%
            </div>
          </div>

          {/* Confidence Bar */}
          <div className="mb-6">
            <div className="flex items-center justify-between mb-2">
              <span className="text-gray-600 font-medium">Confidence Level</span>
              <span className="text-gray-800 font-bold">{(face.confidence * 100).toFixed(1)}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-3 overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${face.confidence * 100}%` }}
                transition={{ duration: 0.8, ease: 'easeOut' }}
                className="bg-gradient-to-r from-blue-500 to-cyan-500 h-full rounded-full"
              />
            </div>
          </div>

          {/* All Probabilities */}
          <div>
            <h4 className="text-gray-600 font-medium mb-3">All Emotion Probabilities:</h4>
            <div className="space-y-3">
              {Object.entries(face.probabilities)
                .sort((a, b) => b[1] - a[1])
                .map(([emotion, probability]) => (
                  <div key={emotion} className="space-y-1">
                    <div className="flex items-center justify-between text-sm">
                      <span className="font-medium text-gray-700">{emotion}</span>
                      <span className="text-gray-600">{(probability * 100).toFixed(2)}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2 overflow-hidden">
                      <motion.div
                        initial={{ width: 0 }}
                        animate={{ width: `${probability * 100}%` }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                        className={`h-full rounded-full ${
                          probability === face.confidence
                            ? 'bg-gradient-to-r from-blue-500 to-cyan-500'
                            : 'bg-gray-400'
                        }`}
                      />
                    </div>
                  </div>
                ))}
            </div>
          </div>

          {/* Bounding Box Info (Optional) */}
          {face.bbox && (
            <div className="mt-6 pt-6 border-t border-gray-200">
              <h4 className="text-gray-600 font-medium mb-2">Detection Details:</h4>
              <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                <div>X: {face.bbox.x}px</div>
                <div>Y: {face.bbox.y}px</div>
                <div>Width: {face.bbox.width}px</div>
                <div>Height: {face.bbox.height}px</div>
              </div>
            </div>
          )}
        </motion.div>
      ))}
    </motion.div>
  );
};

export default ResultDisplay;
