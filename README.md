# Drone Classification with CNN

## Overview

This project focuses on developing two approaches for the classification of drone types from images:

1. A custom Convolutional Neural Network.
2. A fine-tuned pre-trained model, specifically **MobileNetV2**, adapted for the drone classification task.

The models are trained to classify 4 classes:

- **no_drone** — images without any drones  
- **dji_mavic**  
- **dji_phantom**  
- **dji_inspire**

## Dataset

The dataset used for training is from Kaggle:  
[Drone Type Classification Dataset](https://www.kaggle.com/datasets/balajikartheek/drone-type-classification)

The dataset is pre-structured for training and validation, and has all PNG images with different size formats.

## Project Structure

This repository contains two main notebooks:

- **`drone-classification-model.ipynb`** – a custom CNN for drone classification, including data exploration, image preprocessing, model building, and training with Optuna for hyperparameter tuning.
- **`mobilenetv2_fine-tuning.ipynb`** – transfer learning with MobileNetV2.

## Key Features

- Data preprocessing with augmentation
- Built custom CNN model and adapted MobileNetV2
- Hyperparameter optimization using Optuna
- Evaluation with accuracy, F1-score, confusion matrix, and classification report
- Visualization with matplotlib, seaborn, and plotly

## Libraries Used

- Python  
- PyTorch, torchvision  
- Optuna  
- matplotlib, seaborn, plotly  
- scikit-learn  
- PIL (Pillow)  
- pandas, numpy

## To run this project:

To run this project:

1. Clone this repository and install the dependencies listed in the `requirements.txt`.
2. Download the dataset from Kaggle and place it in the appropriate directory.
3. Open the notebook and run the cells step-by-step.

## Author

**Mariia Cherkasova**

