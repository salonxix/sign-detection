# 🤟 AI-Powered Real-Time Sign Language Recognition System

## 📌 Project Overview

The **AI-Powered Real-Time Sign Language Recognition System** is a computer vision and machine learning application that recognizes hand gestures in real time using a webcam. It leverages **MediaPipe's Hand Landmark Detection** to extract 21 key hand landmarks and applies machine learning models to classify gestures accurately.

The system performs real-time detection, landmark preprocessing, feature normalization, confidence estimation, and gesture prediction, making it suitable as a foundation for assistive communication technologies.

---

# 🎯 Problem Statement

Millions of people with hearing or speech impairments rely on sign language for communication. However, many people are unable to understand sign language, creating communication barriers.

This project aims to bridge that gap by converting sign language gestures into machine-recognizable outputs that can later be extended into text and speech.

---

# 🎯 Objectives

* Detect hands in real time using a webcam.
* Extract hand landmarks accurately.
* Train machine learning models for gesture recognition.
* Compare multiple ML algorithms.
* Provide smooth and stable predictions.
* Display prediction confidence.
* Build a scalable foundation for a complete sign language translator.

---

# ✨ Key Features

## 👋 Real-Time Hand Detection

* Detects hands using a webcam.
* Uses MediaPipe Tasks API.
* Supports continuous live recognition.

---

## 🧠 Hand Landmark Extraction

The system extracts **21 hand landmarks**, including fingertips, joints, and wrist coordinates.

Each detected hand is converted into numerical features for machine learning.

---

## 📐 Landmark Preprocessing

To improve prediction accuracy, the landmarks undergo preprocessing:

* Translation (wrist-centered coordinates)
* Scaling
* Normalization
* Feature vector generation

This makes the model more robust against:

* Different hand positions
* Camera distance
* Minor rotations

---

## 🤖 Machine Learning Models

The application compares multiple machine learning algorithms:

* Random Forest
* Support Vector Machine (RBF Kernel)
* XGBoost

The best-performing model is automatically selected after training.

---

## 📊 Automatic Model Evaluation

During training, the system evaluates each model using:

* Cross Validation
* Test Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix

The best-performing model is saved for inference.

---

## 🎥 Real-Time Inference

During live prediction, the application provides:

* Bounding box around the hand
* Predicted gesture
* Confidence percentage
* FPS (Frames Per Second)
* Smoothed prediction using previous frames

---

## 📈 Prediction Smoothing

Instead of relying on a single frame, predictions are averaged across multiple recent frames to reduce flickering and improve stability.

---

## 📂 Dataset Generation

The project includes tools to:

* Capture gesture images
* Automatically organize them by class
* Generate training datasets
* Create serialized feature files for model training

---

# 🛠 Technologies Used

## Programming Language

* Python 3.12

## Computer Vision

* OpenCV

## Hand Tracking

* MediaPipe Tasks API

## Machine Learning

* Random Forest
* Support Vector Machine (SVM)
* XGBoost
* Scikit-learn

## Data Processing

* NumPy

## Visualization

* Matplotlib

---

# 📂 Project Workflow

```
Webcam
   │
   ▼
Capture Video Frame
   │
   ▼
MediaPipe Hand Detection
   │
   ▼
Extract 21 Landmarks
   │
   ▼
Preprocessing
   │
   ├── Translation
   ├── Scaling
   └── Normalization
   │
   ▼
Feature Vector
   │
   ▼
Machine Learning Model
(Random Forest / SVM / XGBoost)
   │
   ▼
Prediction
   │
   ▼
Confidence Score
   │
   ▼
Display Result
```

---

# 📁 Project Files

### `collect_imgs.py`

Captures images for each gesture class and stores them in the dataset directory.

---

### `create_dataset.py`

Processes all collected images, extracts MediaPipe hand landmarks, preprocesses them, and generates `data.pickle`.

---

### `train_classifier.py`

* Loads the dataset.
* Trains multiple machine learning models.
* Compares performance.
* Saves the best model as `model.p`.
* Generates evaluation metrics.

---

### `inference_classifier.py`

Runs real-time prediction using the trained model.

Displays:

* Predicted label
* Confidence score
* FPS
* Bounding box
* Hand landmarks

---

### `model.p`

Serialized machine learning model.

---

### `label_map.json`

Maps model outputs to readable class labels.

---

### `metrics/`

Contains:

* Confusion Matrix
* Training Report

---

# 📊 Current Model Performance

Current Dataset:

* Classes: **0, 1, 2**
* Images: **300**
* Features: **42 per sample**

Training Results:

* Random Forest: **100%**
* SVM: **100%**
* XGBoost: **100%**

(Current performance is based on the existing 3-class dataset.)

---

# 🚀 Future Scope

The project is designed to be extended with:

* Recognition of digits (0–9)
* Recognition of alphabet (A–Z)
* Word formation
* Sentence formation
* Text-to-Speech
* Speech-to-Sign
* Deep learning models (CNN/LSTM/Transformers)
* AWS cloud deployment
* Mobile application
* Multi-language support
* User authentication
* Real-time translation

---

# 👩‍💻 Author

**Saloni Kumari**

B.Tech – Computer Science & Engineering

Amity University Bengaluru

Passionate about Artificial Intelligence, Machine Learning, Computer Vision, Cloud Computing, and Accessible Technology.

---

# 🌟 Conclusion

This project demonstrates the integration of **Computer Vision**, **Machine Learning**, and **Human–Computer Interaction** to build an intelligent real-time sign language recognition system. It serves as a scalable foundation for future assistive technologies capable of enabling more inclusive communication between sign language users and the broader community.
