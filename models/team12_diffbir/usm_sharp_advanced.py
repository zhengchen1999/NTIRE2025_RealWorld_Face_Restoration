import cv2
import os
from pathlib import Path
import numpy as np

def usm_sharp_advanced(img, weight=0.7, contrast=1.2, radius=50, threshold=25, edges_only=True, edge_radius=5, scale=0.4):
    """
    Optimized USM sharpening with enhanced edge detection, frequency separation, and overshoot control.

    Args:
        img (np.array): Input image, HWC, BGR; float32, [0, 1].
        weight (float): Control for sharpening strength.
        contrast (float): High-frequency mask contrast adjustment.
        radius (float): Kernel size of Gaussian blur.
        threshold (int): Threshold for sharpening mask.
        edges_only (bool): If True, only sharpen edges.
        edge_radius (float): Radius for edge preservation filter if edges_only is True.
        scale (float): Overshoot scaling factor for values exceeding local range.
    """

    # Step 1: Gaussian blur to get the low-frequency image
    if radius % 2 == 0:
        radius += 1
    blur = cv2.GaussianBlur(img, (radius, radius), 0)
    residual = img - blur

    # Step 2: Create a blend mask using high-frequency areas based on contrast
    mask = np.abs(residual) * 255 > threshold
    mask = mask.astype(np.float32)
    soft_mask = cv2.GaussianBlur(mask, (radius, radius), 0)
    high_freq_mask = np.clip(contrast * soft_mask, 0, 1)

    # Step 3: Edge-aware filtering (optional)
    if edges_only:
        img_bilateral = cv2.bilateralFilter(img, d=0, sigmaColor=edge_radius, sigmaSpace=edge_radius)
        edge_blur = cv2.GaussianBlur(img_bilateral, (radius, radius), 0)
    else:
        edge_blur = blur

    # Step 4: Compute delta for sharpening enhancement with weight adjustment
    sharpened = img + weight * residual
    sharpened = np.clip(sharpened, 0, 1)  # clip to valid range

    # Step 5: Overshoot control - calculate local min and max values
    local_max = cv2.dilate(img, np.ones((3, 3), np.float32))  # compute local max
    local_min = cv2.erode(img, np.ones((3, 3), np.float32))   # compute local min

    # Apply overshoot control
    overshoot_high = sharpened > local_max
    overshoot_low = sharpened < local_min
    # Scale overshoot areas back within bounds
    sharpened[overshoot_high] = local_max[overshoot_high] + (sharpened[overshoot_high] - local_max[overshoot_high]) * scale
    sharpened[overshoot_low] = local_min[overshoot_low] - (local_min[overshoot_low] - sharpened[overshoot_low]) * scale

    # Step 6: Combine the sharpened image and original using the high-frequency mask
    output = high_freq_mask * sharpened + (1 - high_freq_mask) * img

    return output

def process_images(input_folder, output_folder):
    # Create the output folder if it does not exist
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    
    # Loop through all files in the input folder
    for filename in os.listdir(input_folder):
        file_path = os.path.join(input_folder, filename)
        # Check if the file is an image
        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            # Read the image
            img = cv2.imread(file_path)
            if img is not None:
                # Convert image to float32 and normalize to [0, 1]
                img = img.astype(np.float32) / 255.0
                # Apply USM sharpening
                sharpened_img = usm_sharp_advanced(img)
                # Convert back to uint8
                sharpened_img = (sharpened_img * 255).astype(np.uint8)
                # Build the output file path
                output_path = os.path.join(output_folder, filename)
                # Save the sharpened image
                cv2.imwrite(output_path, sharpened_img)
                print(f'Processed and saved: {output_path}')
            else:
                print(f'Failed to read image: {file_path}')
        else:
            print(f'Skipped non-image file: {file_path}')

# Example usage
# input_folder = '/data/disk1/wsq/Datasets/honor/DIV2K_valid_HR/'
# output_folder = '/data/disk1/wsq/Datasets/honor/DIV2K_valid_HR_USM_0.7_advanced/'
# process_images(input_folder, output_folder)
