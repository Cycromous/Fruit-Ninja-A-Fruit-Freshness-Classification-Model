# Deep Learning Fruit Classifier (CNN)

A desktop application that uses a Convolutional Neural Network (CNN) to classify fruit images as **Fresh** or **Rotten**. Built with Python, TensorFlow/Keras, and Tkinter.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Dataset Setup](#dataset-setup)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [How It Works](#how-it-works)

---

## Overview

This project is a school machine learning application that trains a CNN model on a labeled fruit dataset and provides a GUI for real-time batch inference. Users can train the model from scratch or load a pre-trained one, then test it on individual images or entire folders.

---

## Features

- **CNN Training** — Train a binary image classifier directly from the GUI
- **Batch Inference** — Upload multiple images or a whole folder for classification
- **Ground Truth Comparison** — Automatically compares predictions against known labels and reports accuracy
- **Pre-trained Model Support** — Loads an existing `.keras` model on startup if available
- **Progress Tracking** — Live progress bar and log output during training and inference
- **Image Navigation** — Browse through batch results one image at a time with Prev/Next controls

---

## Project Structure

```
project-root/
│
├── main.py                        # Main application script
├── fruit_cnn_classifier.keras     # Saved model (generated after training)
│
└── Project_Dataset/
    ├── FreshFruits/               # Training images of fresh fruits
    │   ├── apple_fresh_001.jpg
    │   └── ...
    └── RottenFruits/              # Training images of rotten fruits
        ├── apple_rotten_001.jpg
        └── ...
```

---

## Requirements

- Python 3.8+
- TensorFlow 2.x
- OpenCV (`cv2`)
- Pillow
- scikit-learn
- Tkinter (usually included with Python)

---

## Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
   ```

2. **Install dependencies**
   ```bash
   pip install tensorflow opencv-python pillow scikit-learn
   ```

3. **Run the application**
   ```bash
   python main.py
   ```

---

## Dataset Setup

The application expects the dataset to be organized as follows:

```
Project_Dataset/
├── FreshFruits/    ← Place all fresh fruit images here
└── RottenFruits/   ← Place all rotten fruit images here
```

Images must be in `.jpg`, `.jpeg`, or `.png` format.

> **Recommended Dataset:** [Fruits Fresh and Rotten for Classification on Kaggle](https://www.kaggle.com/datasets/sriramr/fruits-fresh-and-rotten-for-classification)

---

## Usage

### Training the Model

1. Make sure your dataset is set up in the `Project_Dataset/` folder.
2. Launch the app with `python main.py`.
3. Click **"Start CNN Training"** — the log window will show progress.
4. Once complete, the model is saved as `fruit_cnn_classifier.keras`.

### Running Inference

After the model is ready (trained or pre-loaded):

- Click **"Upload Images"** to select one or more image files.
- Click **"Upload Folder"** to run inference on an entire directory recursively.
- Use **Next / Prev** to browse results.
- The label shows the prediction, confidence score, and whether it matched the ground truth.

### Ground Truth Detection

The app automatically infers ground truth labels from:
1. **Parent folder name** — if the folder contains "fresh" or "rotten" in its name.
2. **Filename prefix** — filenames starting with `f` = Fresh, `r` = Rotten.

---

## Model Architecture

```
Input (128 × 128 × 3)
    ↓
Conv2D(32 filters, 3×3, ReLU) → MaxPooling(2×2)
    ↓
Conv2D(64 filters, 3×3, ReLU) → MaxPooling(2×2)
    ↓
Conv2D(128 filters, 3×3, ReLU) → MaxPooling(2×2)
    ↓
Flatten
    ↓
Dense(128, ReLU) → Dropout(0.5)
    ↓
Dense(1, Sigmoid)  ← Binary output: 0 = Fresh, 1 = Rotten
```

- **Loss:** Binary Cross-Entropy  
- **Optimizer:** Adam  
- **Epochs:** 10  
- **Batch Size:** 32  
- **Train/Validation Split:** 80% / 20%

---

## How It Works

1. Images are loaded, resized to **128×128 pixels**, and normalized to `[0, 1]`.
2. The dataset is split into training (80%) and validation (20%) sets using stratified sampling.
3. The CNN is trained for 10 epochs and saved to disk.
4. During inference, each image is preprocessed the same way and passed through the model.
5. A sigmoid output above **0.5** is classified as **Rotten**, otherwise **Fresh**.
