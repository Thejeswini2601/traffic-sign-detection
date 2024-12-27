# traffic-sign-detection
This project is aimed at building a machine learning model to classify Indian traffic signs using deep learning. The model is trained on a dataset containing 59 classes of traffic signs and predicts the category of a traffic sign from an image input.


## Overview
This project utilizes Convolutional Neural Networks (CNNs) to detect and classify Indian traffic signs. The model is trained on images of traffic signs, then tested on new images to predict the category. The model achieves an accuracy of 81.8% on the test dataset.

## Features
- **59 Classes**: The model can classify 59 different types of traffic signs.
- **Real-Time Prediction**: The model can predict traffic sign categories from new images in real-time.
- **Image Preprocessing**: Images are resized, normalized, and prepared for input into the model.
- **Early Stopping**: To prevent overfitting, early stopping is used during training to monitor validation loss.

## Dataset
The dataset used in this project is the **Indian Traffic Sign Dataset** containing images of various traffic signs. The images are organized in subfolders where each subfolder corresponds to a traffic sign class. You can download the dataset from the following link:
- [Indian Traffic Sign Dataset](https://www.kaggle.com/datasets/neelpratiksha/indian-traffic-sign-dataset)

## Model Architecture
The model is a CNN that consists of the following layers:
- **Convolutional Layers**: Two convolutional layers with ReLU activation and max-pooling.
- **Fully Connected Layer**: A dense layer with 128 neurons and ReLU activation.
- **Dropout**: A dropout layer to prevent overfitting.
- **Output Layer**: A softmax output layer for multi-class classification (59 classes).

The model is compiled with the Adam optimizer and categorical cross-entropy loss function.

## Installation
To get started with this project, clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/your-username/traffic-sign-classifier.git
cd traffic-sign-classifier
pip install -r requirements.txt
