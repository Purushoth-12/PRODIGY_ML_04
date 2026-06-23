# PRODIGY_ML_04

# ✋ Hand Gesture Recognition using CNN

## 📌 Project Overview

This project demonstrates a Hand Gesture Recognition system built using Convolutional Neural Networks (CNNs). The model is trained on the LeapGestRecog dataset to identify and classify various hand gestures from image inputs. By leveraging Deep Learning techniques, the system automatically learns meaningful visual patterns and achieves accurate gesture classification.

---

## 🎯 Objective

The main objective of this project is to develop a Deep Learning model capable of:

- Recognizing different hand gestures from images
- Learning spatial features automatically using CNNs
- Classifying gestures into multiple categories
- Evaluating model performance using visualization techniques

---

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Matplotlib
- Seaborn
- Scikit-Learn

---

## 📂 Dataset

Dataset Used: LeapGestRecog

The dataset contains thousands of hand gesture images belonging to multiple gesture classes, including:

- Palm
- Fist
- Thumbs Up
- OK Sign
- Index Finger
- Victory Sign
- And several other gestures

---

## ⚙️ Methodology

### Data Preprocessing

- Loaded gesture images from the dataset
- Converted images to grayscale format
- Resized images to 64 × 64 pixels
- Normalized pixel values between 0 and 1
- Encoded gesture labels for model training

### CNN Architecture

The model consists of:

- Conv2D Layers for feature extraction
- MaxPooling Layers for dimensionality reduction
- Flatten Layer
- Dense Fully Connected Layers
- Dropout Layer for regularization
- Softmax Output Layer for classification

### Model Training

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Validation Split used during training
- Performance monitored across multiple epochs

---

## 📊 Performance Evaluation

The model was evaluated using:

### Accuracy Curve

Tracks the training and validation accuracy over epochs.

### Loss Curve

Shows how the model's error decreases during training.

### Confusion Matrix

Provides a detailed visualization of prediction performance across all gesture classes.

### Sample Predictions

Displays actual and predicted gesture labels on test images for verification.

---

## 🚀 Results

- Successfully trained a CNN-based gesture recognition model
- Achieved high classification performance on test data
- Generated Accuracy and Loss visualizations
- Evaluated predictions using a Confusion Matrix
- Demonstrated effective hand gesture classification on unseen images

---

## 📁 Project Structure

PRODIGY_ML_04

├── main.py
├── README.md
├── accuracy_loss_graph.png
├── confusion_matrix.png
└── predictions.png

---

## 🎓 Internship Task

This project was completed as Task 04 of the Machine Learning Internship Program at Prodigy InfoTech.

The project provided hands-on experience in Deep Learning, Computer Vision, Image Processing, and CNN-based classification systems.

⭐ Thank you for visiting this repository! Feel free to explore the code and results. 🚀
