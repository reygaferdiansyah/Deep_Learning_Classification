# Deep Learning Implementation for Image Classification using Convolutional Neural Network (CNN) And  VGG16 on Wayang Kulit Images

[![PyPI version](https://badge.fury.io/py/colabcode.svg)](https://badge.fury.io/py/colabcode)
![python version](https://img.shields.io/badge/python-3.6%2C3.7%2C3.8-blue?logo=python)
![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)
![Kaggle](https://img.shields.io/badge/Kaggle-Dataset-blue.svg)
[![TensorFlow Version](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)

This project demonstrates the implementation of deep learning techniques, specifically Convolutional Neural Network (CNN), for image classification on traditional Indonesian shadow puppet (wayang kulit) images.

## Table of Contents
- [Introduction](#introduction)
- [Prerequisites](#prerequisites)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)

## Introduction

Traditional Indonesian shadow puppetry, known as wayang kulit, is a rich cultural heritage. This project focuses on leveraging deep learning, specifically CNN, to classify wayang kulit images.

## Prerequisites

Before you begin, make sure you have:
- Python installed
- TensorFlow or PyTorch installed
- Access to a dataset of wayang kulit images

## Dataset

The dataset consists of a collection of annotated wayang kulit images. Ensure that the dataset is appropriately prepared for training.

## Model Architecture

The CNN model architecture is designed for image classification tasks. Details about the architecture can be found in the `model_architecture.py` file.

## Training

1. **Prepare Dataset:**
    - Organize the dataset into training and testing sets.

2. **Configure Parameters:**
    - Adjust hyperparameters and configurations in the `config.py` file.

3. **Train the Model:**
    - Run the training script:
      ```bash
      python train.py
      ```

## Evaluation

1. **Model Evaluation:**
    - Assess the performance of the trained model using testing data.

2. **Metrics Calculation:**
    - Calculate relevant metrics such as accuracy, precision, and recall.

## Inference

1. **Inference on New Images:**
    - Utilize the trained model CNN to classify new wayang kulit images.

![image](https://github.com/reygaferdiansyah/Deep_Learning_Classification_CNN/assets/54634029/cdf2db19-ff23-4749-8baf-fc6bee14e890)

2. **Visualization:**
    - Visualize model predictions on new images.

## Results

Include visualizations and a summary of the model's performance, showcasing its ability to classify wayang kulit images accurately.

## CNN
![image](https://github.com/reygaferdiansyah/Deep_Learning_Classification_CNN/assets/54634029/cac7ec56-abbe-426a-9586-030cbe2a3465)

Accuracy : 97.88%
Loss     : 0.112

## VGG16
![image](https://github.com/reygaferdiansyah/Deep_Learning_Classification_CNN/assets/54634029/f7f16c4b-c7c0-443e-8716-050d9e89c788)

Accuracy : 99.39%
Loss     : 0.036


