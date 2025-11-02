import React from 'react';
import { FaGithub, FaHeart } from 'react-icons/fa';

const Footer = () => {
  return (
    <footer className="bg-gray-900 text-white mt-20">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* About Section */}
          <div>
            <h3 className="text-lg font-bold mb-4">Emotion Recognition</h3>
            <p className="text-gray-400 text-sm">
              AI-powered facial emotion detection using deep learning.
              Analyze emotions in real-time from images or webcam.
            </p>
          </div>

          {/* Quick Links */}
          <div>
            <h3 className="text-lg font-bold mb-4">Quick Links</h3>
            <ul className="space-y-2 text-sm">
              <li>
                <a href="#home" className="text-gray-400 hover:text-white transition-colors">
                  Home
                </a>
              </li>
              <li>
                <a href="#upload" className="text-gray-400 hover:text-white transition-colors">
                  Upload Image
                </a>
              </li>
              <li>
                <a href="#webcam" className="text-gray-400 hover:text-white transition-colors">
                  Webcam Capture
                </a>
              </li>
              <li>
                <a href="#info" className="text-gray-400 hover:text-white transition-colors">
                  Model Info
                </a>
              </li>
            </ul>
          </div>

          {/* Tech Stack */}
          <div>
            <h3 className="text-lg font-bold mb-4">Built With</h3>
            <ul className="space-y-2 text-sm text-gray-400">
              <li>React + Tailwind CSS</li>
              <li>Flask + TensorFlow</li>
              <li>OpenCV + Keras</li>
              <li>Framer Motion</li>
            </ul>
          </div>
        </div>

        <div className="border-t border-gray-800 mt-8 pt-8 flex flex-col md:flex-row justify-between items-center">
          <p className="text-gray-400 text-sm flex items-center">
            Made with <FaHeart className="text-red-500 mx-1" /> using AI & ML
          </p>

          <div className="flex items-center space-x-4 mt-4 md:mt-0">
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="text-gray-400 hover:text-white transition-colors"
            >
              <FaGithub className="text-xl" />
            </a>
          </div>
        </div>

        <div className="text-center mt-4">
          <p className="text-gray-500 text-xs">
            &copy; {new Date().getFullYear()} Emotion Recognition. All rights reserved.
          </p>
        </div>
      </div>
    </footer>
  );
};

export default Footer;
