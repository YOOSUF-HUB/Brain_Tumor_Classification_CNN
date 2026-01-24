# 🧠 NeuroScan AI: Brain Tumor Classification

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-url-here.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)](https://www.tensorflow.org/)

## 📋 Overview
NeuroScan AI is a medical imaging application designed to assist in the early detection of brain tumors. Using a **Custom Convolutional Neural Network (CNN)**, the model analyzes MRI scans to classify them into four categories with **97% accuracy**:
* **Glioma**
* **Meningioma**
* **Pituitary Tumor**
* **No Tumor**

The project includes a fully functional web interface built with **Streamlit**, providing a simple "drag-and-drop" user experience for real-time predictions.

**Live Demo: [NeuroScan AI: Brain Tumor Classification](https://braintumorclassificationcnn-btcnn.streamlit.app/)**

## 🚀 Features
* **High Accuracy:** Custom CNN architecture optimized for medical imaging (97% Test Accuracy).
* **Real-time Analysis:** Instant classification of uploaded MRI images.
* **Interactive UI:** User-friendly dashboard with confidence scores and probability breakdowns.
* **Medical Context:** Provides automated educational descriptions for detected conditions.
* **Visual feedback:** "Scanning" animations and dynamic result cards.

## 🛠️ Tech Stack
* **Deep Learning:** TensorFlow, Keras
* **Web Framework:** Streamlit
* **Image Processing:** OpenCV, Pillow (PIL)
* **Visualization:** Matplotlib, NumPy

## 📂 Dataset
The model was trained on the [Brain Tumor MRI Dataset](https://www.kaggle.com/masoudnickparvar/brain-tumor-mri-dataset) containing over 7,000 images across 4 classes.
* **Preprocessing:** Images resized to 256x256, normalized (rescaling layer), and augmented.

## 📸 Screenshots

### 1. Dashboard Interface
<img width="1680" height="1050" alt="Screenshot 2026-01-24 at 10 11 44" src="https://github.com/user-attachments/assets/8b151134-23ad-4210-bf66-acf2544da649" />


### 2. Prediction Result
<img width="1680" height="1050" alt="Screenshot 2026-01-24 at 10 11 58" src="https://github.com/user-attachments/assets/9b71c5f6-9e6a-45c0-a1d1-0a1bf2670582" />

## 🔧 Installation & Usage

### 1. Clone the Repository
```bash
git clone [https://github.com/your-username/brain-tumor-detection.git](https://github.com/your-username/brain-tumor-detection.git)
cd brain-tumor-detection

🧠 Model Architecture
The custom CNN consists of:
  Rescaling Layer: Normalizes pixel values (0-255 → 0-1).
  Convolutional Blocks: 3 blocks of Conv2D (ReLu) + MaxPooling2D.
  Dense Layers: Flatten layer followed by a Dense layer (64 units).
  Output Layer: Softmax activation for 4-class classification.

⚠️ Disclaimer
This tool is for educational and research purposes only. It is not a substitute for professional medical diagnosis. Always consult a certified neurologist or radiologist.
