import argparse
import os

import cv2
import numpy as np
from PIL import Image


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def ensure_directory(path):
    """Ensure a directory exists."""
    os.makedirs(path, exist_ok=True)

def save_image(image, path, file_name):
    full_dir_path = os.path.join(ROOT_DIR, path)
    ensure_directory(full_dir_path)
    full_file_path = os.path.join(full_dir_path, file_name)
    image.save(full_file_path)

def load_image_as_bytes(path, file_name):
    full_path = os.path.join(ROOT_DIR, path, file_name)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"The file does not exist: {full_path}")
    return cv2.imread(full_path)


def apply_clean_full(image):
    """Applies Gaussian blur, adaptive thresholding, removes grid lines, and removes caps from previous lines."""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours of the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Remove small caps above the text
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)

        # Define a heuristic to detect caps (adjust thresholds as needed)
        if h < 10 and w < 30 and y < gray.shape[0] // 2:
            # Fill the detected cap region with white
            cv2.rectangle(binary, (x, y), (x + w, y + h), 0, -1)

    # Restore handwriting onto a white background
    handwriting_mask = binary > 0
    output = np.ones_like(image) * 255  # Create a white background
    output[handwriting_mask] = image[handwriting_mask]

    return output


def apply_clean(image):
    """Applies Gaussian blur, adaptive thresholding, and removes grid lines from an image."""

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding to binarize the image
    binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)

    # # Step 6: Detect vertical lines
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)

    # Step 7: Combine horizontal and vertical lines into one mask
    grid_mask = cv2.add(horizontal_lines, vertical_lines)

    # Step 8: Invert the grid mask
    grid_removed = cv2.bitwise_not(grid_mask)

    # Step 9: Remove the grid from the original binary image
    cleaned_binary = cv2.bitwise_and(binary, binary, mask=grid_removed)

    # Step 10: Remove small noise using contour filtering
    contours, _ = cv2.findContours(cleaned_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 10:  # Threshold for small noise (adjust based on your image)
            cv2.drawContours(cleaned_binary, [contour], -1, 0, -1)  # Remove the contour by filling it with black

    # Step 11: Restore handwriting onto a white background
    handwriting_mask = cleaned_binary > 0
    output = np.ones_like(image) * 255  # Create a white background
    output[handwriting_mask] = image[handwriting_mask]

    return output


def remove_background(input_path, input_name, output_path, output_name):
    image = load_image_as_bytes(input_path, input_name)
    processed_image = apply_clean(image)
    result_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
    save_image(result_image, output_path, output_name)
    return processed_image


def crop_horizontal_and_vertical(image):
    """
    Crop the image by removing white space on all sides (top, bottom, left, right).

    Args:
        image (numpy.ndarray): The input image with handwriting on a white background.

    Returns:
        numpy.ndarray: The cropped image.
    """
    # Convert to grayscale and threshold
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image

    _, binary = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Invert binary image
    binary_inv = cv2.bitwise_not(binary)

    # Find non-white region using contours
    coords = cv2.findNonZero(binary_inv)  # Get all non-zero points
    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)  # Get bounding box
        cropped = image[y:y+h, x:x+w]  # Crop the region
        return cropped
    return image  # If no handwriting is found, return the original image


def crop_lines(image, output_path, padding=5, margin=10, projection_value=0.1):
    """
    Crop the text lines from the given image, considering letter tails and caps,
    and add a white margin around each line, with additional processing to ignore
    and merge small lines.

    Args:
        image (numpy.ndarray): The input image with text on a white background.
        output_path (str): The directory to save the cropped line images.
        padding (int): Number of pixels to extend above and below detected lines.
        margin (int): White space to add above and below each cropped line.
    """
    # Convert to grayscale
    image = crop_horizontal_and_vertical(image)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply horizontal projection to detect text lines
    inverted = cv2.bitwise_not(gray)

    # Apply horizontal projection to detect text lines
    projection = np.sum(inverted, axis=1)

    threshold = np.max(projection) * projection_value  # 10% of the maximum projection value
    line_regions = projection > threshold

    # Detect line start and end indices
    line_indices = []
    start = None
    for i, val in enumerate(line_regions):
        if val and start is None:
            start = i
        elif not val and start is not None:
            line_indices.append((start, i))
            start = None

    # Handle case where the last line reaches the image bottom
    if start is not None:
        line_indices.append((start, len(line_regions)))

    # Calculate the average vertical size of lines
    line_heights = [end - start for start, end in line_indices]
    avg_height = np.mean(line_heights)

    # Filter out lines that are less than half the average height
    filtered_indices = []
    merge_previous_indices = None

    # Merge lines which are too small (wrong crop)
    for i, (start, end) in enumerate(line_indices):
        line_height = end - start
        if line_height >= avg_height / 2:  # Keep lines that are not too small
            if merge_previous_indices and start - merge_previous_indices[1] < padding:
                merged_start = merge_previous_indices[0]
                merged_end = end
                filtered_indices.append((merged_start, merged_end))
                merge_previous_indices = None
            else:
                filtered_indices.append((start, end))

        elif i > 0 and i < len(line_indices) - 1 and (len(filtered_indices) > 0 and start - filtered_indices[len(filtered_indices) - 1][0] < padding):  # Merge with neighbors
            merged_start = filtered_indices[len(filtered_indices) - 1][0]
            merged_end = end
            filtered_indices[len(filtered_indices) - 1] = (merged_start, merged_end)
            merge_previous_indices = (start, end)

    # Expand lines to include letter caps/tails and add white margin
    expanded_indices = []
    for start, end in filtered_indices:
        # Expand by padding
        start = max(0, start - padding)
        end = min(image.shape[0], end + padding)

        # Detect the maximum vertical extent within the line
        line_slice = inverted[start:end, :]
        vertical_projection = np.sum(line_slice, axis=0)
        top_tail = np.where(vertical_projection > 0)[0]
        if len(top_tail) > 0:
            line_min = max(0, np.min(top_tail))
            line_max = min(image.shape[1], np.max(top_tail))
        else:
            line_min, line_max = 0, image.shape[1]

        expanded_indices.append((start, end, line_min, line_max))

    ensure_directory(output_path)

    # Crop and save each line
    for i, (start, end, min_x, max_x) in enumerate(expanded_indices):
        # Add white margin
        start = max(0, start - margin)
        end = min(image.shape[0], end + margin)
        cropped_line = image[start:end, :]  # Crop the entire width

        # ADD BACKGROUND REMOVAL
        cropped_line = apply_clean_full(cropped_line)
        cropped_line = crop_horizontal_and_vertical(cropped_line)

        # Add horizontal margin
        cropped_line = cv2.copyMakeBorder(
            cropped_line,
            top=margin,
            bottom=margin,
            left=padding,
            right=padding,
            borderType=cv2.BORDER_CONSTANT,
            value=[255, 255, 255],
        )

        # Save the cropped line
        line_image_path = os.path.join(output_path, f"line_{i + 1}.png")

        cv2.imwrite(line_image_path, cropped_line)
        print(f"Saved: {line_image_path}")

def main():
    parser = argparse.ArgumentParser(description="Remove background and crop lines from an image.")

    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder containing input files.")
    parser.add_argument("--input_image", type=str, required=True, help="Filename of the input image.")
    parser.add_argument("--output_crop_folder", type=str, required=True, help="Path to save the cropped images.")
    parser.add_argument("--output_image", type=str, required=True, help="Path to save the cleaned output image.")
    parser.add_argument("--output_lines_folder", type=str, required=True, help="Path to save cropped lines.")

    args = parser.parse_args()

    # Run remove_background
    cropped_output = remove_background(args.input_folder, args.input_image, args.output_crop_folder, args.output_image)

    # Run crop_lines
    crop_lines(cropped_output, args.output_lines_folder)

if __name__ == "__main__":
    main()

# Example of run: python process_image.py --input_folder tests --input_image test_many_lines_with_blank_black_white.jpg --output_crop_folder output_crop --output_image output_many_lines_with_blank_black_white.png --output_lines_folder output_crop/cleaned6
