# 📊 CIFAR-10 Image Classification using PyTorch 🚀

This repository contains my implementation of an image classification model for the **CIFAR-10 dataset** using **PyTorch**. The goal was to classify images into 10 different categories, and after applying several optimization techniques, I achieved an accuracy of **81.23%** on test data.

---

## 📌 Overview

The **CIFAR-10 dataset** consists of **60,000 color images** (32x32 pixels) divided into **10 classes** (e.g., airplane, automobile, bird, cat, etc.).  

Through this project, I explored:
- Building a **Convolutional Neural Network (CNN)** from scratch.
- Applying **Data Augmentation** for better generalization.
- Using **Batch Normalization** and **Dropout** to stabilize and regularize training.
- Plotting **Learning Curves** to visualize training dynamics.

---

## ✅ Achieved Results

- **Final Test Accuracy**: 81.23%
- Smooth training and validation curves.
- Improved generalization using data augmentation.

---

## 🚀 Key Techniques Used

### 1. **Data Augmentation**
- `RandomHorizontalFlip`
- `RandomRotation`

Helps in preventing overfitting and makes the model robust to different image orientations.

### 2. **Model Architecture**
- **Convolutional Layers** for feature extraction.
- **Batch Normalization** for faster convergence.
- **Max Pooling** to reduce spatial dimensions.
- **Fully Connected Layers** for final classification.
- **Dropout** to prevent overfitting.

### 3. **Training Details**
- **Optimizer**: Adam
- **Loss Function**: Cross Entropy Loss
- **Epochs**: 50
- **Batch Size**: 100

---

## ⚙️ Training and Evaluation

- The model was trained for **50 epochs**, showing steady improvement in both accuracy and loss.
- Applied **train/test split** to evaluate generalization.
- Plotted **training loss and accuracy curves** to analyze performance over epochs.

---

## 🧠 Challenges and Solutions

### ✅ Overfitting:
- **Problem**: High training accuracy but low test accuracy.
- **Solution**: Added **Dropout layers** and used **Data Augmentation** to generate diverse training samples.

### ✅ Slow accuracy improvement:
- **Problem**: Accuracy plateaued at ~70%.
- **Solution**: Introduced **Batch Normalization** and tuned model complexity.

### ✅ Generalization issues:
- **Problem**: Poor performance on unseen data.
- **Solution**: Data augmentation and regularization techniques helped to improve model robustness.

---

## 📉 Learning Curve (Training Loss & Accuracy)

- Tracked and plotted training loss and accuracy over epochs to ensure stable training and convergence.
- Observed smooth curves indicating proper model optimization.

---

## 💻 Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib (for plotting)
- NumPy

---

## 📂 Project Structure

