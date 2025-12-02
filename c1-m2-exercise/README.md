# Handwritten Letter Recognition with PyTorch

This project implements a neural network using PyTorch to recognize handwritten letters (A-Z) from the EMNIST dataset. The model achieves 60%+ accuracy across all 26 letter classes and includes a bonus challenge to decode hidden handwritten messages.

## Project Structure

- `C1M2_Assignment.ipynb`: The main Jupyter Notebook containing the assignment, model implementation, training loop, and evaluation.
- `EMNIST_data/`: Directory containing the EMNIST Letters dataset.
- `helper_utils.py`: Helper functions for data visualization, model evaluation, and decoding hidden messages.
- `unittests.py`: Unit tests to validate the implementation of the exercises.
- `unittests_utils.py`: Utility functions and test batteries for the unit tests.
- `hidden_message_images.pkl`: Pickled file containing handwritten images for the bonus decoding challenge.
- `trained_student_model.pth`: Saved trained model weights.

## Prerequisites

Ensure you have Python installed (preferably Python 3.8 or higher).

## Setup and Installation

1.  **Navigate to the project directory**:
    ```bash
    cd c1-m2-exercise
    ```

2.  **Install the required dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download the EMNIST Dataset**:
    The EMNIST dataset will be automatically downloaded when you run the notebook for the first time.

## How to Run

1.  **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2.  **Open the Assignment**:
    In the Jupyter interface, open `C1M2_Assignment.ipynb`.

3.  **Run the Cells**:
    Execute the cells sequentially to:
    - Load and explore the EMNIST Letters dataset
    - Create DataLoaders for efficient batch processing
    - Build and initialize the neural network model
    - Train the model for multiple epochs
    - Evaluate the model's accuracy on the test set
    - Decode the hidden handwritten message (bonus challenge!)

## Features Implemented

### Exercise 1: Create DataLoaders
- Implemented `create_dataloaders()` function to create training and test DataLoaders
- Configured proper batch size, shuffling, and dataset splitting

### Exercise 2: Initialize Model
- Built a Sequential neural network with:
  - `nn.Flatten()` layer to convert 28x28 images to 1D vectors
  - Multiple `nn.Linear()` and `nn.ReLU()` layers for feature extraction
  - Final `nn.Linear()` layer for 26-class classification
- Configured CrossEntropyLoss and SGD optimizer with learning rate 0.01

### Exercise 3: Training Loop
- Implemented `train_epoch()` function with:
  - Forward pass through the model
  - Loss calculation
  - Backpropagation (`loss.backward()`)
  - Weight updates (`optimizer.step()`)
  - Gradient zeroing (`optimizer.zero_grad()`)

### Exercise 4: Model Evaluation
- Implemented `evaluate()` function to:
  - Calculate accuracy on the test dataset
  - Use `torch.no_grad()` for efficient inference
  - Track correct predictions across all batches

### Exercise 5: Per-Class Accuracy
- Achieved 60%+ accuracy for all 26 letter classes (A-Z)
- Visualized per-class performance to identify strengths and weaknesses

### Bonus Challenge: Decode Hidden Message
- Used the trained model to decode a hidden handwritten message
- Demonstrated real-world application of the classifier

## Model Performance

The trained model achieves:
- **Overall Test Accuracy**: 60%+ across all classes
- **Per-Class Accuracy**: Each letter (A-Z) has at least 60% accuracy
- **Hidden Message Decoding**: Successfully decodes messy handwritten text

## Key Learnings

- Understanding of image classification pipelines
- Hands-on experience with PyTorch DataLoaders and batch processing
- Implementation of complete training and evaluation loops
- Importance of per-class metrics in multi-class classification
- Practical application of neural networks to real-world OCR tasks
