# Image-Classification
An interactive web application for classifying images using a machine learning model trained on the CIFAR-100 dataset. 

## Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Demo](#demo)
- [License](#license)

## Overview
This project demonstrates an end-to-end workflow for image classification using the CIFAR-100 dataset. It includes:

- Preprocessing the CIFAR-100 dataset.
- Training a Convolutional Neural Network (CNN) model using TensorFlow/Keras.
- Deploying the trained model with Streamlit to create an interactive web application.

## Dataset
### CIFAR-100
- Contains 60,000 32x32 images divided into 100 classes.
- Each class belongs to 20 superclasses, e.g.:
- Split into 50,000 training and 10,000 test samples.
- [Download the dataset](https://www.cs.toronto.edu/~kriz/cifar.html).

## Model Architecture
- **Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Layers**:
  - Convolutional layers with ReLU activation
  - MaxPooling for down-sampling
  - Dropout layers to prevent overfitting
  - Fully connected (Dense) layers for classification into 100 classes
- **Training**:
  - Optimizer: Adam
  - Loss Function: Categorical Crossentropy
  - Metrics: Accuracy

## Features
- Upload an image for classification.
- View the model’s prediction and confidence score in real time.
- Supports all CIFAR-100 categories.
- Lightweight and user-friendly interface.

## Setup Instructions
Follow these steps to set up the project locally:

### Clone the Repository:
```bash
git clone https://github.com/your-username/Image-Classification.git
cd Image-Classification
```

### Install Dependencies:
Ensure Python 3.8+ is installed. Then run:
```bash
pip install -r requirements.txt
```

### Download Model Weights:
Download the trained model from the repository files above (optimized_model.h5) and place them in the `models` directory.

### Run the App:
```bash
streamlit run app.py
```

### Access the App:
Open your browser and go to [https://3signet-internship-fqv4rnpnqqdbhuqbtkgyur.streamlit.app/]

## Usage
1. Launch the app.
2. Upload a CIFAR-100-like image (32x32 size recommended).
3. View the prediction and confidence score instantly.

## Demo
Watch the live demo [https://www.loom.com/share/7dd0fa14f31c496ba883db1d0faef582](#).

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


















 























