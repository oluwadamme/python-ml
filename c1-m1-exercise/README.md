# Delivery Time Prediction with PyTorch

This project implements a neural network using PyTorch to predict delivery times based on various features such as distance, time of day, and whether the delivery is on a weekend.

## Project Structure

- `C1M1_Assignment.ipynb`: The main Jupyter Notebook containing the assignment, model implementation, and training loop.
- `data_with_features.csv`: The dataset used for training the model.
- `helper_utils.py`: Helper functions for data visualization and processing.
- `unittests.py`: Unit tests to validate the implementation of the exercises.
- `unittests_utils.py`: Utility functions for the unit tests.

## Prerequisites

Ensure you have Python installed (preferably Python 3.8 or higher).

## Setup and Installation

1.  **Clone the repository** (if you haven't already) and navigate to the project directory:
    ```bash
    cd c1-m1-exercise
    ```

2.  **Install the required dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## How to Run

1.  **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2.  **Open the Assignment**:
    In the Jupyter interface, open `C1M1_Assignment.ipynb`.

3.  **Run the Cells**:
    Execute the cells sequentially to load the data, build the model, train it, and view the predictions.

## Features Implemented

- **Rush Hour Feature**: A custom feature engineering step to identify rush hour periods.
- **Neural Network Architecture**: A sequential model with Linear and ReLU layers.
- **Training Loop**: A custom training loop using MSE Loss and SGD Optimizer.
