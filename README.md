🚗 Lane Detection Web App using Deep Learning
📌 Overview

This project is a deep learning–based lane detection web application built using Streamlit, TensorFlow, and OpenCV.
It allows users to upload a road video and automatically detects lane markings using a trained neural network model, overlaying the detected lanes in green on the original video.

The application focuses on:

High-quality video processing

Real-time progress feedback

Clean and interactive UI

Resume-friendly deployment readiness

🎯 Key Features

✅ Upload and process MP4 videos

✅ AI-based lane detection using a trained CNN model

✅ Adjustable lane overlay opacity

✅ Maintains aspect ratio and HD quality

✅ Real-time processing progress bar

✅ One-click download of processed video

🧠 Model Details

Framework: TensorFlow / Keras

Input Resolution: 512 × 256

Output: Binary lane segmentation mask

Post-processing: Thresholding + green overlay

The model predicts a pixel-wise lane mask, which is resized back to the original frame resolution and blended with the input video.

## 🎥 Demo
![Lane Detection Demo](output_colored.gif)

