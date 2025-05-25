"""
Authors: A. Pérez, S. Rodríguez, N. Ayobi, N. Aparicio, E. Dessevres, and P. Arbeláez
Paper: MuST: Multi-Scale Transformers for Surgical Phase Recognition

Description:
This script processes JSON files containing annotations and predictions for a specified video, generating visualizations of the ground truth (annotations) and predictions. The output is saved as an image file.

Usage:
- Provide paths to the annotation and prediction JSON files.
- Specify the video name to visualize as a parameter.
- The script will generate and save RGB visualizations of the annotations and predictions for the specified video.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---------------------------------------------------------------------------- #
# Utility Functions
# ---------------------------------------------------------------------------- #

def load_json_file(file_path):
    """Load and return the content of a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in file: {file_path}")

def hex_to_rgb(hex_color):
    """Convert a hex color to an RGB tuple."""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def rgb_annotations(color_tuple, annotations):
    """Map annotation values to their corresponding RGB colors."""
    annotations_array = np.array(annotations)
    return np.array([color_tuple[ann] for ann in annotations_array])

def sort_prediction_dict(original_dict):
    """Sort the prediction dictionary."""
    sorted_dict = dict(
    sorted(
        original_dict.items(),
        key=lambda item: (
            int(item[0].split('/')[0][-2:]),  # Extract video number
            int(item[0].split('/')[1].split('.')[0].split('_')[-1])  # Extract frame number
        )
    )
    )
    return sorted_dict


# ---------------------------------------------------------------------------- #
# Visualization Function
# ---------------------------------------------------------------------------- #

def plot_annotations_predictions(annotations, predictions, phases_colors, output_filename, video, output_folder):
    """
    Plot and save ground truth (annotations) and predictions for a video.

    Args:
        annotations (list): Ground truth annotations.
        predictions (list): Predictions for the video.
        phases_colors (list): List of hex color codes for phases.
        output_filename (str): Name of the output file.
        video (str): Name of the video being visualized.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Convert annotations and predictions to RGB
    annotations_rgb = rgb_annotations([hex_to_rgb(color) for color in phases_colors], annotations)
    predictions_rgb = rgb_annotations([hex_to_rgb(color) for color in phases_colors], predictions)

    # Create matrices for visualizations
    matrix_annotations = np.ones((int(len(annotations_rgb) * 0.2), len(annotations_rgb), 3))
    matrix_predictions = np.ones((int(len(predictions_rgb) * 0.2), len(predictions_rgb), 3))

    matrix_annotations *= annotations_rgb.reshape(1, -1, 3)
    matrix_predictions *= predictions_rgb.reshape(1, -1, 3)

    # Plot ground truth
    fig_ann, ax_ann = plt.subplots(figsize=(10, 2))
    ax_ann.imshow(matrix_annotations)
    ax_ann.axis('off')
    ax_ann.set_title(f"Ground Truth (GT) - {video}")
    os.makedirs(video, exist_ok=True)
    plt.savefig(f"{video}/{video}_ground_truth_{output_filename}", dpi=800, bbox_inches='tight')
    plt.close(fig_ann)

    # Plot predictions
    fig_pred, ax_pred = plt.subplots(figsize=(10, 2))
    ax_pred.imshow(matrix_predictions)
    ax_pred.axis('off')
    ax_pred.set_title(f"Predictions - {video}")
    plt.savefig(f"{video}/{video}_predictions_{output_filename}", dpi=800, bbox_inches='tight')
    plt.close(fig_pred)

# ---------------------------------------------------------------------------- #
# Main Function
# ---------------------------------------------------------------------------- #

def main(annotations_file, predictions_file, video_name, output_folder):
    """Main function to visualize annotations and predictions for a specified video."""
    # Load JSON data
    annotations = load_json_file(annotations_file)["annotations"]
    predictions = sort_prediction_dict(load_json_file(predictions_file))
    
    # Validate inputs
    if not annotations:
        raise ValueError("Annotations data is empty or not properly loaded.")
    if not predictions:
        raise ValueError("Predictions data is empty or not properly loaded.")
    if not any(video_name in ann["image_name"] for ann in annotations):
        raise ValueError(f"Video name '{video_name}' not found in annotations.")

    # Define phase colors - Can be defined based on the number of phases in the task
    phases_colors = [
        "#001219", "#005F73", "#0A9396", "#76C893", "#94D2BD",
        "#E9D8A6", "#EE9B00", "#CA6702", "#BB3E03", "#AE2012", "#9B2226"
    ]

    # Filter annotations and predictions for the specified video
    gt = []
    preds = []

    for key, value in predictions.items():
        if video_name in key:
            matching_ann = next((ann for ann in annotations if ann["image_name"] == key), None)
            if matching_ann:
                gt.append(matching_ann["phases"])
                preds.append(np.argmax(value["phases_score_dist"]))
    
    # Plot and save visualizations
    plot_annotations_predictions(gt, preds, phases_colors, f"{video_name}_visual.png", video_name, output_folder)

if __name__ == "__main__":
    # Replace with the actual file paths and video name you want to visualize
    annotations_file = "path/to/annotations.json"
    predictions_file = "path/to/predictions.json"
    output_folder = ""
    video_name = "" # Case or video for visualization (e.g., CASEXX, video_XX, videoXX)

    main(annotations_file, predictions_file, video_name, output_folder)

    print("Visualizations saved successfully.")