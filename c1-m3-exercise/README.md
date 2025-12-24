# Plant Seedling Classification with PyTorch Custom Dataset

This project implements a complete image classification pipeline using PyTorch to classify plant seedlings into 12 different species. The assignment focuses on building a **custom Dataset class**, implementing **data transformations and augmentation**, and creating **efficient DataLoaders** for training, validation, and testing.

## Project Overview

This assignment demonstrates how to:
- Build a custom PyTorch `Dataset` class from scratch
- Implement data transformations and augmentation strategies
- Create efficient DataLoaders with proper train/val/test splits
- Handle image data stored in a directory structure with CSV labels
- Apply normalization and preprocessing for optimal model training

## Project Structure

```
c1-m3-exercise/
├── C1M3_Assignment.ipynb          # Main Jupyter Notebook with all exercises
├── data/                           # Plant seedling dataset
│   ├── train/                      # Training images
│   ├── train.csv                   # Training labels
│   ├── val/                        # Validation images
│   ├── val.csv                     # Validation labels
│   ├── test/                       # Test images
│   └── test.csv                    # Test labels
├── helper_utils.py                 # Visualization and utility functions
├── unittests.py                    # Unit tests for validating implementations
├── unittests_utils.py              # Test battery utilities
├── requirements.txt                # Python dependencies
├── output.png                      # Sample visualization outputs
└── output-1.png                    # Additional visualization outputs
```

## Dataset Information

The **Plant Seedling Classification** dataset contains images of plant seedlings from 12 different species:
- Black-grass
- Charlock
- Cleavers
- Common Chickweed
- Common wheat
- Fat Hen
- Loose Silky-bent
- Maize
- Scentless Mayweed
- Shepherds Purse
- Small-flowered Cranesbill
- Sugar beet

Each image is stored in a directory structure, with corresponding labels in CSV files.

## Prerequisites

- Python 3.8 or higher
- PyTorch 2.6.0+
- torchvision 0.21.0+

## Setup and Installation

1.  **Navigate to the project directory**:
    ```bash
    cd c1-m3-exercise
    ```

2.  **Install the required dependencies**:
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Verify the dataset**:
    The dataset should already be present in the `data/` directory with the following structure:
    - `data/train/` - Training images
    - `data/train.csv` - Training labels
    - `data/val/` - Validation images
    - `data/val.csv` - Validation labels
    - `data/test/` - Test images
    - `data/test.csv` - Test labels

## How to Run

1.  **Start Jupyter Notebook**:
    ```bash
    jupyter notebook
    ```

2.  **Open the Assignment**:
    In the Jupyter interface, open `C1M3_Assignment.ipynb`.

3.  **Run the Cells Sequentially**:
    Execute the cells in order to:
    - Explore the dataset structure and file organization
    - Implement the custom `PlantsDataset` class
    - Create and test data transformations
    - Build DataLoaders with proper splits
    - Visualize transformed and augmented images
    - Validate your implementation with unit tests

## Exercises Implemented

### Exercise 1: Custom Dataset Class - `PlantsDataset`

Implemented a complete custom PyTorch `Dataset` class with the following components:

#### `__init__` Method
- Accepts `dataset_path`, `csv_file`, and optional `transform`
- Calls `load_labels()` to populate `.labels` attribute
- Calls `read_classname()` to populate `.class_names` attribute

#### `load_labels()` Method
- Reads the CSV file containing image filenames and labels
- Returns a list of tuples: `[(filename, label), ...]`

#### `read_classname()` Method
- Reads the class names from a text file
- Returns a list of class names (12 plant species)

#### `retrieve_image()` Method
- Loads an image from disk given its filename
- Returns a PIL Image object

#### `__len__()` Method
- Returns the total number of samples in the dataset

#### `__getitem__()` Method
- Retrieves the image and label for a given index
- Applies transformations if provided
- Returns `(image, label)` tuple

#### `get_label_description()` Method
- Converts a numeric label to its human-readable class name

### Exercise 2: Data Transformations - `get_transformations()`

Implemented two transformation pipelines:

#### Main Transform (for validation/test)
1. **Resize**: Resize images to 224x224 pixels
2. **ToTensor**: Convert PIL Image to PyTorch Tensor
3. **Normalize**: Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

#### Augmentation Transform (for training)
1. **RandomHorizontalFlip**: Flip images horizontally with p=0.5
2. **RandomRotation**: Rotate images by ±10 degrees
3. **Resize**: Resize images to 224x224 pixels
4. **ToTensor**: Convert PIL Image to PyTorch Tensor
5. **Normalize**: Normalize with mean=[0.485, 0.456, 0.406] and std=[0.229, 0.224, 0.225]

### Exercise 3: DataLoaders - `get_dataloaders()`

Implemented efficient DataLoaders with the following features:

#### Dataset Splitting
- Used `random_split()` to split the dataset into train/val/test sets
- Proper split sizes based on the provided dataset structure

#### Transform Assignment
- Training set: Uses `augmentation_transform` (with RandomHorizontalFlip and RandomRotation)
- Validation set: Uses `main_transform` (no augmentation)
- Test set: Uses `main_transform` (no augmentation)

#### DataLoader Configuration
- **Training DataLoader**: `shuffle=True` for randomization
- **Validation DataLoader**: `shuffle=False` for consistent evaluation
- **Test DataLoader**: `shuffle=False` for consistent evaluation
- Configurable `batch_size` parameter

#### SubsetWithTransform Wrapper
- Custom wrapper class to apply different transforms to different dataset splits
- Ensures training data gets augmented while val/test data remains unchanged

## Key Concepts Covered

### 1. Custom Dataset Implementation
- Understanding PyTorch's `Dataset` abstract class
- Implementing required methods: `__len__()` and `__getitem__()`
- Loading images from disk on-demand (lazy loading)
- Handling CSV-based label files

### 2. Data Transformations
- Image preprocessing (resizing, normalization)
- Data augmentation techniques (flipping, rotation)
- Using `torchvision.transforms.Compose()` for transformation pipelines
- Understanding when to apply augmentation (training only)

### 3. DataLoaders and Batching
- Creating DataLoaders for efficient batch processing
- Proper train/val/test splitting with `random_split()`
- Shuffling strategies for different dataset splits
- Applying different transforms to different splits

### 4. Best Practices
- Normalizing images using ImageNet statistics
- Augmenting training data to improve generalization
- Not augmenting validation/test data for fair evaluation
- Using PIL Images as the base format before transformations

## Helper Utilities

The `helper_utils.py` file provides several useful functions:

- **`plot_img()`**: Visualize individual images with labels
- **`get_grid()`**: Create subplot grids for multiple images
- **`print_data_folder_structure()`**: Display dataset directory structure
- **`explore_extensions()`**: Analyze file types in the dataset
- **`visual_exploration()`**: Display random samples from the dataset
- **`plot_training_metrics()`**: Visualize training/validation loss and accuracy
- **`Denormalize`**: Class to reverse normalization for visualization

## Unit Tests

The assignment includes comprehensive unit tests in `unittests.py`:

- **Exercise 1 Tests**: Validate `PlantsDataset` implementation
  - Check class inheritance from `Dataset`
  - Verify `load_labels()` and `read_classname()` are called in `__init__`
  - Test `__len__()` returns correct dataset size
  - Validate `__getitem__()` returns correct image and label
  - Ensure transforms are applied correctly

- **Exercise 2 Tests**: Validate `get_transformations()` implementation
  - Check main transform has 3 transformations (Resize, ToTensor, Normalize)
  - Check augmentation transform has 5 transformations
  - Verify correct transformation types and parameters
  - Validate normalization mean and std values

- **Exercise 3 Tests**: Validate `get_dataloaders()` implementation
  - Check DataLoader types
  - Verify dataset split sizes
  - Ensure correct transforms are applied to each split
  - Validate shuffling configuration

## Real-World Applications

This assignment's techniques are used in:

### 1. **Agricultural AI & Precision Farming**
   - **Application**: Automated weed detection and crop monitoring
   - **Connection**: Farmers use computer vision to identify weeds vs. crops, similar to your plant seedling classifier

### 2. **Plant Disease Detection**
   - **Application**: Identifying diseased plants from leaf images
   - **Connection**: Custom datasets and data augmentation are crucial for training robust plant disease classifiers

### 3. **Biodiversity Monitoring**
   - **Application**: Automated species identification in ecological surveys
   - **Connection**: Custom datasets with CSV labels are common in biodiversity research

### 4. **Smart Greenhouses**
   - **Application**: Automated plant growth monitoring and optimization
   - **Connection**: Real-time image classification helps optimize growing conditions

### 5. **Food Quality Control**
   - **Application**: Sorting and grading produce based on visual inspection
   - **Connection**: Custom datasets and efficient DataLoaders enable real-time processing

## Tips for Success

1. **Understanding the Dataset**: Explore the data structure before implementing the Dataset class
2. **Testing Incrementally**: Test each method of `PlantsDataset` individually before moving on
3. **Visualizing Transformations**: Use the helper functions to visualize how transformations affect images
4. **Debugging DataLoaders**: Print batch shapes and sample images to verify correct implementation
5. **Running Unit Tests**: Use the provided unit tests to validate your implementation at each step

## Common Issues and Solutions

### Issue: Images not loading correctly
- **Solution**: Check that `dataset_path` and `csv_file` paths are correct
- Verify that `retrieve_image()` constructs the full image path correctly

### Issue: Transforms not being applied
- **Solution**: Ensure `__getitem__()` checks if `self.transform` is not None before applying
- Verify that transforms are passed correctly to the Dataset constructor

### Issue: DataLoader errors
- **Solution**: Check that `SubsetWithTransform` is correctly wrapping the dataset splits
- Ensure `random_split()` is used with the correct split sizes

### Issue: Normalization values incorrect
- **Solution**: Use ImageNet statistics: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]

## Next Steps

After completing this assignment, you'll be ready to:
- Train a CNN model on this dataset
- Implement custom training loops
- Experiment with different augmentation strategies
- Build end-to-end image classification pipelines

## Key Learnings

- **Custom Dataset Design**: How to build flexible, reusable Dataset classes
- **Data Pipeline Optimization**: Efficient data loading and preprocessing strategies
- **Augmentation Strategies**: When and how to apply data augmentation
- **Code Organization**: Separating data handling from model training logic
- **Testing and Validation**: Using unit tests to ensure correct implementation

---

**Happy Learning! 🌱🚀**
