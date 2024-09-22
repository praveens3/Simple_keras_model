# Simple Keras Model Project

## Project Overview
This project implements a simple neural network model using Keras to predict values based on a mathematical relationship between input features. The model is trained on synthetic data generated using specified conditions, demonstrating basic regression techniques.

## Features
- Uses Keras to build a sequential model for regression tasks.
- Trains the model with a custom mathematical relationship.
- Evaluates model performance using predicted vs. actual values.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Data Generation](#data-generation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Contributing](#contributing)
- [License](#license)

## Installation
To run this project, ensure you have Python and the necessary packages installed. You can install the required packages using pip:
```bash
pip install numpy tensorflow keras
```

Alternatively, you can create a requirements.txt file with the following content:
```bash
numpy
tensorflow
keras
```

Then install using: ```bash pip install -r requirements.txt```

## Usage
1. Clone the repository:
```bash
git clone <repository-url>
cd simple-keras-model
```
2. Run the model training script:
```bash
python train_model.py
```

## Data Generation
The model uses synthetic data generated from the following mathematical relationship:
 For even value b = a*2 +1
 For odd value b = a*2 +5
This data is normalized before training.

## Model Architecture
The model consists of:

Input layer: Accepts normalized input features.
Hidden layer: Uses a Dense layer with ReLU activation.
Output layer: Outputs a single predicted value.

## Training
The model is trained using a set number of epochs, with the ability to configure batch size. The training process is designed to minimize the mean squared error loss.

## Evaluation
The model evaluates its performance by comparing predicted values to actual values, displaying individual accuracy and average accuracy for a set of test inputs.

## Contributing
Contributions are welcome! If you have suggestions for improvements or would like to report a bug, please open an issue or submit a pull request.

## License
This project is for learning purpose but it can get scaled and used, licensed under the MIT License - see the LICENSE file for details.
