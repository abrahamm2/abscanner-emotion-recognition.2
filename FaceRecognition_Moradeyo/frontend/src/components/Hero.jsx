import React from 'react';
import { motion } from 'framer-motion';
import { FaRocket, FaCamera, FaUpload, FaChartBar } from 'react-icons/fa';

const Hero = () => {
  const features = [
    {
      icon: <FaUpload />,
      title: 'Upload Images',
      description: 'Upload any image to detect emotions from faces',
    },
    {
      icon: <FaCamera />,
      title: 'Webcam Capture',
      description: 'Use your webcam for real-time emotion detection',
    },
    {
      icon: <FaChartBar />,
      title: 'Detailed Analysis',
      description: 'Get probability scores for all detected emotions',
    },
  ];

  return (
    <div className="relative overflow-hidden bg-gradient-to-br from-blue-600 via-cyan-600 to-blue-800 text-white" id="home">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20">
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="text-center"
        >
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ type: 'spring', stiffness: 200, delay: 0.2 }}
            className="inline-block mb-6"
          >
            <FaRocket className="text-6xl mx-auto" />
          </motion.div>

          <h1 className="text-5xl md:text-6xl font-extrabold mb-6">
            Emotion Recognition
          </h1>

          <p className="text-xl md:text-2xl mb-8 text-blue-100 max-w-3xl mx-auto">
            AI-powered facial emotion detection using deep learning.
            Analyze emotions in real-time from images or webcam with state-of-the-art accuracy.
          </p>

          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.4 }}
            className="flex flex-wrap justify-center gap-4 mb-12"
          >
            <a
              href="#upload"
              className="bg-white text-blue-600 font-bold py-3 px-8 rounded-lg shadow-lg hover:shadow-xl transform hover:-translate-y-1 transition-all duration-200"
            >
              Get Started
            </a>
            <a
              href="#info"
              className="bg-blue-700 text-white font-bold py-3 px-8 rounded-lg shadow-lg hover:shadow-xl border-2 border-white transform hover:-translate-y-1 transition-all duration-200"
            >
              Learn More
            </a>
          </motion.div>
        </motion.div>

        {/* Features */}
        <motion.div
          initial={{ opacity: 0, y: 40 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6, duration: 0.6 }}
          className="grid grid-cols-1 md:grid-cols-3 gap-8 mt-16"
        >
          {features.map((feature, index) => (
            <motion.div
              key={index}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.7 + index * 0.1 }}
              whileHover={{ scale: 1.05, y: -5 }}
              className="bg-white/10 backdrop-blur-sm rounded-xl p-6 border border-white/20 hover:bg-white/20 transition-all"
            >
              <div className="text-4xl mb-4">{feature.icon}</div>
              <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
              <p className="text-blue-100">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>
      </div>

      {/* Decorative Elements */}
      <div className="absolute top-0 left-0 w-full h-full overflow-hidden pointer-events-none">
        <motion.div
          animate={{
            scale: [1, 1.2, 1],
            rotate: [0, 90, 0],
          }}
          transition={{
            duration: 20,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="absolute -top-1/2 -left-1/4 w-96 h-96 bg-blue-500/20 rounded-full blur-3xl"
        />
        <motion.div
          animate={{
            scale: [1, 1.3, 1],
            rotate: [0, -90, 0],
          }}
          transition={{
            duration: 25,
            repeat: Infinity,
            ease: 'linear',
          }}
          className="absolute -bottom-1/2 -right-1/4 w-96 h-96 bg-cyan-500/20 rounded-full blur-3xl"
        />
      </div>
    </div>
  );
};

export default Hero;
