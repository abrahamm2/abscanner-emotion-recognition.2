import React from 'react';
import { motion } from 'framer-motion';
import { FaBrain } from 'react-icons/fa';

const Navbar = () => {
  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className="bg-white shadow-lg sticky top-0 z-50"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex items-center space-x-3">
            <motion.div
              whileHover={{ rotate: 360 }}
              transition={{ duration: 0.5 }}
              className="bg-gradient-to-r from-blue-600 to-cyan-600 p-2 rounded-lg"
            >
              <FaBrain className="text-white text-2xl" />
            </motion.div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent">
                Emotion Recognition
              </h1>
              <p className="text-xs text-gray-500">AI-Powered Face Analysis</p>
            </div>
          </div>

          <div className="hidden md:flex items-center space-x-6">
            <a
              href="#home"
              className="text-gray-700 hover:text-blue-600 transition-colors font-medium"
            >
              Home
            </a>
            <a
              href="#upload"
              className="text-gray-700 hover:text-blue-600 transition-colors font-medium"
            >
              Upload
            </a>
            <a
              href="#webcam"
              className="text-gray-700 hover:text-blue-600 transition-colors font-medium"
            >
              Webcam
            </a>
            <a
              href="#info"
              className="text-gray-700 hover:text-blue-600 transition-colors font-medium"
            >
              About
            </a>
          </div>
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar;
