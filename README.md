# Multi-Layer Perceptron (MLP) for CIFAR Dataset

This repository contains a Python implementation of a multilayer perceptron (MLP) designed to classify images in the CIFAR-10 dataset. The MLP is built using TensorFlow and Keras, and includes steps to load data, preprocess it, build the model, train it, and evaluate its performance. 


## Introduction

The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes, with 6,000 images per class. The goal of this implementation is to train an MLP to classify these images.

## Requirements

To run this code, you will need:

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib


## Usage

### 1. Clone this repository

```
git clone https://github.com/yourusername/mlp-cifar.git
cd mlp-cifar
```

### 2. Run the Jupyter Notebook:
```
jupyter notebook
```

### 3. Open the notebook mlp_cifar.ipynb and execute the cells step by step.

## Model Architecture

The MLP architecture consists of:
* Input layer: Accepts 32x32 RGB images.
* Flatten layer: Converts the 2D image into a 1D vector.
* Two hidden layers: Each with ReLU activation.
* Output layer: Softmax activation to classify into 10 categories.

```
input_layer = layers.Input((32, 32, 3))
x = layers.Flatten()(input_layer)
x = layers.Dense(200, activation="relu")(x)
x = layers.Dense(150, activation="relu")(x)
output_layer = layers.Dense(NUM_CLASSES, activation="softmax")(x)
```

## Training and Evaluation

The model is trained using the Adam optimizer with a learning rate of 0.0005. The training process consists of 10 epochs, with a batch size of 32. 

To evaluate the model, we use categorical cross-entropy loss and accuracy metrics.


```
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=10, shuffle=True)
```

## Results

The model's categorical cross-entropy loss decreased from 1.95 to 1.36, while the accuracy improved from 29.15% to 51.39% after 10 epochs. The model predictions and actual classes are visualized for a sample of images.



