import React from 'react';
import { motion } from 'framer-motion';
import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ImageUpload from './components/ImageUpload';
import WebcamCapture from './components/WebcamCapture';
import ModelInfo from './components/ModelInfo';
import Footer from './components/Footer';

function App() {
  return (
    <div className="min-h-screen">
      <Navbar />

      <main>
        {/* Hero Section */}
        <Hero />

        {/* Upload Section */}
        <section className="py-20 px-4">
          <ImageUpload />
        </section>

        {/* Divider */}
        <motion.div
          initial={{ opacity: 0, scaleX: 0 }}
          whileInView={{ opacity: 1, scaleX: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto"
        >
          <div className="h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent" />
        </motion.div>

        {/* Webcam Section */}
        <section className="py-20 px-4">
          <WebcamCapture />
        </section>

        {/* Divider */}
        <motion.div
          initial={{ opacity: 0, scaleX: 0 }}
          whileInView={{ opacity: 1, scaleX: 1 }}
          transition={{ duration: 0.8 }}
          viewport={{ once: true }}
          className="max-w-4xl mx-auto"
        >
          <div className="h-px bg-gradient-to-r from-transparent via-gray-300 to-transparent" />
        </motion.div>

        {/* Model Info Section */}
        <section className="py-20 px-4">
          <ModelInfo />
        </section>
      </main>

      <Footer />
    </div>
  );
}

export default App;
