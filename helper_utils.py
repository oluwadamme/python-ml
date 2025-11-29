import torch
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time
import os
import sys
from directory_tree import DisplayTree
from fastai.vision.all import show_image, show_titled_image
from tqdm.auto import tqdm
import copy
import random
from collections import defaultdict
import matplotlib.ticker as mticker
import numpy as np
import torchvision
from torch.utils.data import DataLoader


def plot_results(model, distances, times):
    """
    Plots the actual data points and the model's predicted line for a given dataset.

    Args:
        model: The trained machine learning model to use for predictions.
        distances: The input data points (features) for the model.
        times: The target data points (labels) for the plot.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculation for efficient inference
    with torch.no_grad():
        # Make predictions using the trained model
        predicted_times = model(distances)

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the actual data points
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='None', label='Actual Delivery Times')
    
    # Plot the predicted line from the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', marker='None', label='Predicted Line')
    
    # Set the title of the plot
    plt.title('Actual vs. Predicted Delivery Times')
    # Set the x-axis label
    plt.xlabel('Distance (miles)')
    # Set the y-axis label
    plt.ylabel('Time (minutes)')
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()


def plot_nonlinear_comparison(model, new_distances, new_times):
    """
    Compares and plots the predictions of a model against new, non-linear data.

    Args:
        model: The trained model to be evaluated.
        new_distances: The new input data for generating predictions.
        new_times: The actual target values for comparison.
    """
    # Set the model to evaluation mode
    model.eval()
    
    # Disable gradient computation for inference
    with torch.no_grad():
        # Generate predictions using the model
        predictions = model(new_distances)

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))
    
    # Plot the actual data points
    plt.plot(new_distances.numpy(), new_times.numpy(), color='orange', marker='o', linestyle='None', label='Actual Data (Bikes & Cars)')
    
    # Plot the predictions from the model
    plt.plot(new_distances.numpy(), predictions.numpy(), color='green', marker='None', label='Linear Model Predictions')
    
    # Set the title of the plot
    plt.title('Linear Model vs. Non-Linear Reality')
    # Set the label for the x-axis
    plt.xlabel('Distance (miles)')
    # Set the label for the y-axis
    plt.ylabel('Time (minutes)')
    # Add a legend to the plot
    plt.legend()
    # Add a grid to the plot for better readability
    plt.grid(True)
    # Display the plot
    plt.show()


def plot_data(distances, times, normalize=False):
    """
    Creates a scatter plot of the data points.

    Args:
        distances: The input data points for the x-axis.
        times: The target data points for the y-axis.
        normalize: A boolean flag indicating whether the data is normalized.
    """
    # Create a new figure with a specified size
    plt.figure(figsize=(8, 6))

    # Plot the data points as a scatter plot
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Actual Delivery Times')

    # Check if the data is normalized to set appropriate labels and title
    if normalize:
        # Set the plot title for normalized data
        plt.title('Normalized Delivery Data (Bikes & Cars)')
        # Set the x-axis label for normalized data
        plt.xlabel('Normalized Distance')
        # Set the y-axis label for normalized data
        plt.ylabel('Normalized Time')

    # Handle the case for un-normalized data
    else:
        # Set the plot title for un-normalized data
        plt.title('Delivery Data (Bikes & Cars)')
        # Set the x-axis label for un-normalized data
        plt.xlabel('Distance (miles)')
        # Set the y-axis label for un-normalized data
        plt.ylabel('Time (minutes)')
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()


def plot_final_fit(model, distances, times, distances_norm, times_std, times_mean):
    """
    Plots the predictions of a trained model against the original data,
    after de-normalizing the predictions.

    Args:
        model: The trained model used for prediction.
        distances: The original, un-normalized input data.
        times: The original, un-normalized target data.
        distances_norm: The normalized input data for the model.
        times_std: The standard deviation used for de-normalization.
        times_mean: The mean value used for de-normalization.
    """
    # Set the model to evaluation mode
    model.eval()

    # Disable gradient calculations for prediction
    with torch.no_grad():
        # Get predictions from the model using normalized data
        predicted_norm = model(distances_norm)

    # De-normalize the predictions to their original scale
    predicted_times = (predicted_norm * times_std) + times_mean

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original data points
    plt.plot(distances.numpy(), times.numpy(), color='orange', marker='o', linestyle='none', label='Actual Data (Bikes & Cars)')

    # Plot the de-normalized predictions from the model
    plt.plot(distances.numpy(), predicted_times.numpy(), color='green', label='Non-Linear Model Predictions')

    # Set the title of the plot
    plt.title('Non-Linear Model Fit vs. Actual Data')
    # Set the x-axis label
    plt.xlabel('Distance (miles)')
    # Set the y-axis label
    plt.ylabel('Time (minutes)')
    # Add a legend to the plot
    plt.legend()
    # Enable the grid
    plt.grid(True)
    # Display the plot
    plt.show()


def plot_training_progress(epoch, loss, model, distances_norm, times_norm):
    """
    Plots the training progress of a model on normalized data,
    showing the current fit at each epoch.

    Args:
        epoch: The current training epoch number.
        loss: The loss value at the current epoch.
        model: The model being trained.
        distances_norm: The normalized input data.
        times_norm: The normalized target data.
    """
    # Clear the previous plot from the output cell
    clear_output(wait=True)

    # Make predictions using the current state of the model
    predicted_norm = model(distances_norm)

    # Convert tensors to NumPy arrays for plotting
    x_plot = distances_norm.numpy()
    y_plot = times_norm.numpy()

    # Detach predictions from the computation graph and convert to NumPy
    y_pred_plot = predicted_norm.detach().numpy()

    # Sort the data based on distance to ensure a smooth line plot
    sorted_indices = x_plot.argsort(axis=0).flatten()

    # Create a new figure for the plot
    plt.figure(figsize=(8, 6))

    # Plot the original normalized data points
    plt.plot(x_plot, y_plot, color='orange', marker='o', linestyle='none', label='Actual Normalized Data')

    # Plot the model's predictions as a line
    plt.plot(x_plot[sorted_indices], y_pred_plot[sorted_indices], color='green', label='Model Predictions')

    # Set the title of the plot, including the current epoch
    plt.title(f'Epoch: {epoch + 1} | Normalized Training Progress')
    # Set the x-axis label
    plt.xlabel('Normalized Distance')
    # Set the y-axis label
    plt.ylabel('Normalized Time')
    # Display the legend
    plt.legend()
    # Add a grid to the plot
    plt.grid(True)
    # Show the plot
    plt.show()

    # Pause briefly to allow the plot to be rendered
    time.sleep(0.05)


def get_dataloader_bar(dataloader, color="green"):
    """
    Generates and returns a tqdm progress bar for a dataloader.

    Args:
        dataloader: The data loader for which to create the progress bar.
        color (str): The color of the progress bar.
    """
    # Get the total number of samples from the dataloader's dataset.
    num_samples = len(dataloader.dataset)

    # Initialize a tqdm progress bar with specified settings.
    pbar = tqdm(
        # Set the total number of iterations for the bar.
        total=num_samples,
        # Dynamically calculate the width of the progress bar.
        ncols=int(num_samples / 10) + 300,
        # Define the format string for the progress bar's appearance.
        bar_format="{desc} {bar} {postfix}",
        # Direct the progress bar output to the standard output stream.
        file=sys.stdout,
        # Set the color of the progress bar.
        colour=color,
    )

    # Return the configured progress bar object.
    return pbar


def update_dataloader_bar(p_bar, batch, current_bs, n_samples):
    """
    Updates a given tqdm progress bar with the current batch processing status.

    Args:
        p_bar: The tqdm progress bar object to update.
        batch (int): The current batch index (zero-based).
        current_bs (int): The size of the current batch.
        n_samples (int): The total number of samples in the dataset.
    """
    # Advance the progress bar by the number of items in the current batch.
    p_bar.update(current_bs)
    # Set the description to show the current batch number.
    p_bar.set_description(f"Batch {batch+1}")

    # Check if the current batch is the last one.
    if (batch + 1) * current_bs > n_samples:
        # Update the postfix to show the total number of samples processed.
        p_bar.set_postfix_str(f"{n_samples} of a total of  {n_samples} samples")
    else:
        # Update the postfix to show the cumulative number of samples processed.
        p_bar.set_postfix_str(
            f"{current_bs*(batch+1)} of a total of  {n_samples} samples"
        )


def plot_img(img, label=None, info=None, ax=None):
    """
    Displays an image with an optional label and supplementary information.

    Args:
        img: The image to be displayed.
        label: An optional label to be used as the image title.
        info: Optional text to display below the image.
        ax: An optional matplotlib axes object to plot on. If not provided,
            a new figure and axes will be created.
    """

    def add_info_text(ax, info):
        """
        Adds supplementary text below the plot on a given axes.

        Args:
            ax: The matplotlib axes object.
            info (str): The text to be added.
        """
        # Add text to the axes at a specified position.
        ax.text(
            0.5, -0.1, info, transform=ax.transAxes, ha="center", va="top", fontsize=10
        )
        # Set the x-axis label position to the top.
        ax.xaxis.set_label_position("top")

    # Create a new figure and axes if none are provided.
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Check if a label is provided to determine how to display the image.
    if label:
        # Create a title string with the provided label.
        title = f"Label: {label}"
        # Display the image with the generated title.
        show_titled_image((img, title), ax=ax)
    else:
        # Display the image without a title.
        show_image(img, ax=ax)

    # Check if supplementary information is provided.
    if info:
        # Add the information as text below the image.
        add_info_text(ax, info)

    # If no axes were passed in, display the newly created plot.
    if ax is None:
        plt.show()


def get_grid(num_rows, num_cols, figsize=(16, 8)):
    """
    Creates a grid of subplots within a Matplotlib figure.

    This function handles cases where the grid has only one row or one
    column to ensure the returned axes object is always a 2D iterable.

    Args:
        num_rows (int): The number of rows in the subplot grid.
        num_cols (int): The number of columns in the subplot grid.
        figsize (tuple): The width and height of the figure in inches.

    Returns:
        tuple: A tuple containing the Matplotlib figure and a 2D list of axes objects.
    """
    # Create a figure and a set of subplots.
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)

    # Handle the case where there is only one row.
    if num_rows == 1:
        # Ensure the axes object is iterable for consistency.
        axes = [axes]
    # Handle the case where there is only one column.
    elif num_cols == 1:
        # Ensure the axes object is a 2D list for consistent indexing.
        axes = [[ax] for ax in axes]

    # Return the figure and the formatted axes grid.
    return fig, axes


def print_data_folder_structure(root_dir, max_depth=1):
    """
    Displays the folder and file structure of a given directory.

    Args:
        root_dir (str): The path to the root directory to be displayed.
        max_depth (int): The maximum depth to traverse the directory tree.
    """
    # Define the configuration settings for displaying the directory tree.
    config_tree = {
        # Specify the starting path for the directory tree.
        "dirPath": root_dir,
        # Set to False to include both files and directories.
        "onlyDirs": False,
        # Set the maximum depth for the tree traversal.
        "maxDepth": max_depth,
        # Specify a sorting option (100 typically means no specific sort).
        "sortBy": 100,
    }
    # Create and display the tree structure using the unpacked configuration.
    DisplayTree(**config_tree)


def explore_extensions(root_dir):
    """
    Scans a directory and its subdirectories to catalog files by their extension.

    Args:
        root_dir (str): The path to the root directory to scan.

    Returns:
        dict: A dictionary where keys are the unique file extensions found (in lowercase)
              and values are lists of full file paths for each extension.
    """
    # Initialize a dictionary to store file paths, grouped by extension.
    extensions = {}
    # Walk through the directory tree starting from the root directory.
    for dirpath, _, filenames in os.walk(root_dir):
        # Iterate over each file in the current directory.
        for filename in filenames:
            # Extract the file extension and convert it to lowercase.
            ext = os.path.splitext(filename)[1].lower()
            # If the extension has not been seen before, add it to the dictionary.
            if ext not in extensions:
                # Initialize a new list for this extension.
                extensions[ext] = []
            # Append the full path of the file to the list for its extension.
            extensions[ext].append(os.path.join(dirpath, filename))
    # Return the dictionary of extensions and their corresponding file paths.
    return extensions


def quick_debug(img):
    """
    Prints key debugging information about an image tensor.

    This function displays the shape, data type, and value range of a given
    image tensor to help with quick diagnostics.

    Args:
        img: The image tensor to inspect.
    """
    # Print the shape of the image tensor.
    print(f"Shape: {img.shape}")  # Should be [3, 224, 224]
    # Print the data type of the tensor.
    print(f"Type: {img.dtype}")  # Should be torch.float32
    # Print the minimum and maximum pixel values in the tensor.
    print(
        f"Range of pixel values: [{img.min():.1f}, {img.max():.1f}]"
    )  # Should be around [-2, 2]# Should be around [-2, 2]


# def get_mean_std():
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     return mean, std



def load_cifar100_subset(target_classes, train_transform, val_transform, root='./cifar_100'):
    """
    Loads and filters the CIFAR-100 dataset to include only specified target classes.

    This function first checks for a local copy of the CIFAR-100 dataset and
    downloads it if not found. It then filters both the training and test sets
    to retain only the images and labels corresponding to the classes specified
    in `target_classes`. The labels are remapped to be contiguous from 0.

    Args:
        target_classes: A list of class name strings to be included in the dataset subset.
        train_transform: A torchvision transform to be applied to the training dataset images.
        val_transform: A torchvision transform to be applied to the test/val dataset images.
        root: The root directory where the dataset is stored or will be downloaded.

    Returns:
        A tuple containing the filtered training dataset and the filtered test dataset.
        Returns (None, None) if a specified target class is not found.
    """
    # Construct the path to the CIFAR-100 dataset directory.
    cifar100_path = os.path.join(root, 'cifar-100-python')
    # Check if the dataset directory exists locally.
    if os.path.isdir(cifar100_path):
        print(f"Dataset found in '{root}'. Loading from local files.")
    # If not found, inform the user that it will be downloaded.
    else:
        print(f"Dataset not found in '{root}'. Downloading...")

    # Load the full CIFAR-100 training dataset.
    train_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=True, 
        download=True, 
        transform=train_transform
    )

    # Load the full CIFAR-100 test dataset.
    test_dataset_full = torchvision.datasets.CIFAR100(
        root=root, 
        train=False, 
        download=True, 
        transform=val_transform
    )
    print("Dataset loaded successfully.")

    # Get the list of all class names from the dataset.
    all_classes = train_dataset_full.classes
    try:
        # Get the original integer indices for the target class names.
        target_indices = [all_classes.index(cls) for cls in target_classes]
    # Handle the case where a specified class name is not in the dataset.
    except ValueError as e:
        print(f"Error: One of the target classes not found in CIFAR-100. {e}")
        return None, None
        
    # Create a mapping from the original class indices to new, contiguous indices (0, 1, 2, ...).
    label_map = {old_label: new_label for new_label, old_label in enumerate(target_indices)}

    # Define a helper function to filter a dataset based on the target classes.
    def _filter_dataset(dataset):
        # Convert the list of targets to a NumPy array for efficient boolean indexing.
        targets_np = np.array(dataset.targets)
        # Create a boolean mask to identify which samples belong to the target classes.
        indices_to_keep = np.isin(targets_np, target_indices)
        
        # Filter the dataset's image data using the boolean mask.
        dataset.data = dataset.data[indices_to_keep]
        
        # Get the original labels of the samples that are being kept.
        original_targets_to_keep = targets_np[indices_to_keep]
        # Remap the original labels to the new contiguous labels.
        dataset.targets = [label_map[target] for target in original_targets_to_keep]
        
        # Update the dataset's class list to only include the target classes.
        dataset.classes = target_classes
        return dataset

    print(f"\nFiltering for {len(target_classes)} classes...")
    # Apply the filtering logic to the full training dataset.
    train_dataset_subset = _filter_dataset(train_dataset_full)
    # Apply the filtering logic to the full test dataset.
    test_dataset_subset = _filter_dataset(test_dataset_full)
    print("Filtering complete. Returning training and validation datasets.")
    
    # Return the filtered training and test subsets.
    return train_dataset_subset, test_dataset_subset



def visualise_images(dataset, grid):
    """
    Displays a grid of images from a dataset, with one random image per class.

    Args:
        dataset: The dataset object containing the images and labels.
        grid (tuple): A tuple specifying the number of rows and columns for the image grid.
    """

    # Create a shallow copy of the dataset to avoid modifying the original
    dataset_copy = copy.copy(dataset)
    # Set the transform on the copied dataset to convert images to tensors
    dataset_copy.transform = torchvision.transforms.ToTensor()

    # Create a DataLoader to handle batching and shuffling of the data
    loader = DataLoader(dataset_copy, batch_size=64, shuffle=True)

    # Unpack the grid dimensions from the input tuple
    rows, cols = grid
    # Calculate the total number of images to display in the grid
    num_images_to_show = rows * cols

    # Get the dataset object from the DataLoader
    dataset_to_show = loader.dataset

    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset_to_show.targets):
        class_indices[target].append(idx)
        
    # Get the list of class names from the dataset
    class_names = dataset_to_show.classes

    # Create a figure and a set of subplots for the grid layout
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        
        # Retrieve the image tensor and its corresponding label from the dataset
        image_tensor, _ = dataset_to_show[random_image_index]
        
        # Convert the tensor to a NumPy array and transpose dimensions for display
        img_to_display = image_tensor.numpy().transpose((1, 2, 0))
        
        # Get the name of the class corresponding to the class label
        class_name = class_names[class_label]
        
        # Display the image on the current subplot
        ax.imshow(img_to_display)
        
        # Set the title of the subplot to the capitalized class name
        ax.set_title(class_name.capitalize(), fontsize=16)
        # Turn off the axis for a cleaner look
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Display the plot
    plt.show()

    # Clean up the copied dataset to free up memory
    del dataset_copy
    
    

def plot_training_metrics(metrics):
    """
    Plots the training and validation metrics from a model training process.

    This function generates two side-by-side plots:
    1. Training Loss vs. Validation Loss.
    2. Validation Accuracy.

    Args:
        metrics (list): A list or tuple containing three lists:
                        [train_losses, val_losses, val_accuracies].
    """
    # Unpack the metrics into their respective lists
    train_losses, val_losses, val_accuracies = metrics
    
    # Determine the number of epochs from the length of the training losses list
    num_epochs = len(train_losses)
    # Create a 1-indexed range of epoch numbers for the x-axis
    epochs = range(1, num_epochs + 1)

    # Create a figure and a set of subplots with 1 row and 2 columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Configure the first subplot for training and validation loss ---
    # Select the first subplot
    ax1 = axes[0]
    # Plot training loss data
    ax1.plot(epochs, train_losses, color='#085c75', linewidth=2.5, marker='o', markersize=5, label='Training Loss')
    # Plot validation loss data
    ax1.plot(epochs, val_losses, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Loss')
    # Set the title and axis labels for the loss plot
    ax1.set_title('Training & Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    # Display the legend
    ax1.legend()
    # Add a grid for better readability
    ax1.grid(True, linestyle='--', alpha=0.6)

    # --- Configure the second subplot for validation accuracy ---
    # Select the second subplot
    ax2 = axes[1]
    # Plot validation accuracy data
    ax2.plot(epochs, val_accuracies, color='#fa5f64', linewidth=2.5, marker='o', markersize=5, label='Validation Accuracy')
    # Set the title and axis labels for the accuracy plot
    ax2.set_title('Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    # Display the legend
    ax2.legend()
    # Add a grid for better readability
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # --- Apply dynamic and consistent styling to both subplots ---
    # Calculate a suitable interval for the x-axis ticks to avoid clutter
    x_interval = (num_epochs - 1) // 10 + 1

    # Loop through each subplot to apply common axis settings
    for ax in axes:
        # Set the y-axis to start at 0 and the x-axis to span the epochs
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=1, right=num_epochs)
        
        # Set the major tick locator for the x-axis using the dynamic interval
        ax.xaxis.set_major_locator(mticker.MultipleLocator(x_interval))
        # Set the font size for the tick labels on both axes
        ax.tick_params(axis='both', which='major', labelsize=10)

    # Adjust subplot parameters for a tight layout
    plt.tight_layout()
    # Display the plots
    plt.show()
    
    
    
def visualise_predictions(model, data_loader, device, grid):
    """
    Visualizes model predictions on a grid of images from a dataset.

    Args:
        model: The trained PyTorch model to use for predictions.
        data_loader: The PyTorch DataLoader for the dataset.
        device: The device (e.g., 'cpu' or 'cuda') to run the model on.
        grid (tuple): A tuple specifying the number of rows and columns for the image grid.
    """
    # Set the model to evaluation mode
    model.eval()

    # Get the dataset and class names from the data loader
    dataset = data_loader.dataset
    class_names = dataset.classes
    
    # Define mean and standard deviation values for de-normalizing the images
    cifar100_mean = np.array([0.5071, 0.4867, 0.4408])
    cifar100_std = np.array([0.2675, 0.2565, 0.2761])
    
    # Create a dictionary to store lists of indices for each class
    class_indices = defaultdict(list)
    # Iterate through the dataset to populate the class_indices dictionary
    for idx, target in enumerate(dataset.targets):
        class_indices[target].append(idx)
        
    # Unpack the grid dimensions
    rows, cols = grid
    # Calculate the total number of images to display
    num_images_to_show = rows * cols
    
    # Create a figure and a set of subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2)) 
    # Adjust the spacing between subplots
    plt.subplots_adjust(wspace=0.3, hspace=0.8)

    # Iterate over each subplot in the grid
    for i, ax in enumerate(axes.flat):
        # If the current index is out of bounds, turn off the subplot axis
        if i >= num_images_to_show or i >= len(class_names):
            ax.axis('off')
            continue
            
        # Set the class label based on the current iteration index
        class_label = i
        
        # Get the list of image indices for the current class
        indices_for_class = class_indices[class_label]
        # If there are no images for this class, turn off the subplot axis
        if not indices_for_class:
            ax.axis('off')
            continue

        # Choose a random image index from the list for the current class
        random_image_index = random.choice(indices_for_class)
        # Retrieve the image tensor and its true label
        image_tensor, true_label = dataset[random_image_index]
        
        # Add a batch dimension and move the tensor to the specified device
        image_batch = image_tensor.unsqueeze(0).to(device)
        
        # Disable gradient calculations for inference
        with torch.no_grad():
            # Get model predictions
            output = model(image_batch)
            # Find the index of the highest score, which is the predicted class
            _, predicted_index = torch.max(output, 1)
        
        # Extract the predicted label as a Python number
        predicted_label = predicted_index.item()
        
        # Convert tensor to a NumPy array and transpose dimensions for display
        img_np = image_tensor.cpu().numpy().transpose((1, 2, 0))
        # De-normalize the image using the predefined mean and std
        denormalized_img = cifar100_std * img_np + cifar100_mean
        # Clip the pixel values to the valid range [0, 1]
        clipped_img = np.clip(denormalized_img, 0, 1)
        
        # Get the string names for the true and predicted labels
        true_name = class_names[true_label]
        predicted_name = class_names[predicted_label]
        
        # Set the title color to green for correct predictions and red for incorrect ones
        title_color = 'green' if true_label == predicted_label else 'red'
        
        # Display the image
        ax.imshow(clipped_img)
        # Set the title with true and predicted labels
        ax.set_title(f"True: {true_name.capitalize()}\nPred: {predicted_name.capitalize()}", 
                     color=title_color, fontsize=10, pad=5)
        # Turn off the axis
        ax.axis('off')

    # Adjust subplot parameters for a tight layout
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # Show the final plot
    plt.show()

class Denormalize:
    """
    A callable class to reverse the normalization of a tensor image.

    This class calculates the inverse transformation of a standard normalization
    and can be used as a transform step, for instance, to visualize images
    after they have been normalized for a model.
    """
    def __init__(self, mean, std):
        """
        Initializes the denormalization transform.

        Args:
            mean (list or tuple): The mean values used for the original normalization.
            std (list or tuple): The standard deviation values used for the original
                                 normalization.
        """
        # Calculate the adjusted mean for the denormalization process.
        new_mean = [-m / s for m, s in zip(mean, std)]
        # Calculate the adjusted standard deviation for the denormalization process.
        new_std = [1 / s for s in std]
        # Create a Normalize transform object with the inverse parameters.
        self.denormalize = transforms.Normalize(mean=new_mean, std=new_std)

    def __call__(self, tensor):
        """
        Applies the denormalization transform to a tensor.

        Args:
            tensor: The normalized tensor to be denormalized.

        Returns:
            The denormalized tensor.
        """
        # Apply the denormalization transform to the input tensor.
        return self.denormalize(tensor)
        
# Combined dataset: bikes for short distances, cars for longer ones
new_distances = torch.tensor(
    [
        [1.0],
        [1.5],
        [2.0],
        [2.5],
        [3.0],
        [3.5],
        [4.0],
        [4.5],
        [5.0],
        [5.5],
        [6.0],
        [6.5],
        [7.0],
        [7.5],
        [8.0],
        [8.5],
        [9.0],
        [9.5],
        [10.0],
        [10.5],
        [11.0],
        [11.5],
        [12.0],
        [12.5],
        [13.0],
        [13.5],
        [14.0],
        [14.5],
        [15.0],
        [15.5],
        [16.0],
        [16.5],
        [17.0],
        [17.5],
        [18.0],
        [18.5],
        [19.0],
        [19.5],
        [20.0],
    ],
    dtype=torch.float32,
)

# Corresponding delivery times in minutes
new_times = torch.tensor(
    [
        [6.96],
        [9.67],
        [12.11],
        [14.56],
        [16.77],
        [21.7],
        [26.52],
        [32.47],
        [37.15],
        [42.35],
        [46.1],
        [52.98],
        [57.76],
        [61.29],
        [66.15],
        [67.63],
        [69.45],
        [71.57],
        [72.8],
        [73.88],
        [76.34],
        [76.38],
        [78.34],
        [80.07],
        [81.86],
        [84.45],
        [83.98],
        [86.55],
        [88.33],
        [86.83],
        [89.24],
        [88.11],
        [88.16],
        [91.77],
        [92.27],
        [92.13],
        [90.73],
        [90.39],
        [92.98],
    ],
    dtype=torch.float32,
)
