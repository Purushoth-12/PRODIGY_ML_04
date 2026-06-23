# ✋ Hand Gesture Recognition using CNN

## 📌 Project Overview

This project implements a Hand Gesture Recognition System using Convolutional Neural Networks (CNNs). The model is trained on hand gesture image data to automatically recognize and classify different hand gestures. By learning visual patterns from images, the CNN can accurately identify gesture categories and make predictions on unseen data.

---

## 🎯 Objective

The objective of this project is to develop a Deep Learning model that can:

✅ Recognize different hand gestures from images

✅ Automatically learn image features using CNNs

✅ Classify gestures into multiple categories

✅ Evaluate model performance through visualizations and predictions

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

The dataset contains thousands of hand gesture images belonging to multiple gesture classes. Images are preprocessed and resized before being fed into the CNN model.

---

## ⚙️ Methodology

### Data Preprocessing

- Loaded gesture images from dataset
- Converted images into grayscale
- Resized images to a fixed size
- Normalized pixel values
- Encoded gesture labels

### CNN Model Development

- Convolution Layers for feature extraction
- MaxPooling Layers for dimensionality reduction
- Dense Layers for classification
- Dropout Layer to reduce overfitting
- Softmax Output Layer for multi-class prediction

### Model Training

- Optimizer: Adam
- Loss Function: Categorical Crossentropy
- Validation Split used during training
- Performance monitored across multiple epochs

---

## 📊 Output Results

### 📈 Accuracy and Loss Graph

Shows the training and validation accuracy and loss across epochs, helping visualize the learning progress of the CNN model.

Output File:
accuracy_&_loss.png

### 📊 Confusion Matrix

Displays the classification performance by comparing actual labels with predicted labels for all gesture classes.

Output File:
confussion_matrix.png

### ✋ Sample Predictions

Shows test gesture images along with their actual and predicted labels to demonstrate the model's recognition capability.

Output File:
pridiction_plot.png

---

## 📁 Project Structure

PRODIGY_ML_04

├── hand_gesture_recognition.py

├── README.md

├── accuracy_loss_graph.png

├── confusion_matrix.png

└── sample_predictions.png

---

## 🚀 Results

✅ Successfully trained a CNN-based Hand Gesture Recognition Model

✅ Generated Accuracy and Loss Visualization Graphs

✅ Created a Confusion Matrix for performance evaluation

✅ Predicted hand gestures on unseen images

✅ Demonstrated effective gesture classification using Deep Learning

---

## 🎓 Internship Task

This project was completed as Task-04 of the Machine Learning Internship Program at Prodigy InfoTech.

The task provided practical experience in:

- Deep Learning
- Computer Vision
- Image Processing
- Convolutional Neural Networks (CNNs)
- Multi-Class Image Classification

---

⭐ Thank you for visiting this repository!

If you found this project interesting, feel free to explore the code and results. 🚀

#MachineLearning #DeepLearning #ComputerVision #CNN #TensorFlow #Python #ProdigyInfoTech
