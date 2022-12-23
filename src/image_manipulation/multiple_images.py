import imghdr
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import os


def read_images(dir_path):
    """
    Read all images in a directory and return them as a list of NumPy arrays.

    Parameters:
        dir_path (str): The path to the directory containing the images.

    Returns:
        images: A list of NumPy arrays representing the images.
    """
    images = []
    for filename in os.listdir(dir_path):
        # Check if file is an image
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".PNG") or filename.endswith(
                ".JPG"):
            # Read image and store as NumPy array
            filepath = os.path.join(dir_path, filename)
            image = cv2.imread(filepath)
            images.append(image)
    return images


def cut_images(image, patch_height=300, patch_width=300):
    """
    Cut an image into smaller images of a given size.

    Args:
        image (ndarray): The image to be cut, represented as a NumPy array.
        patch_height (int, optional): The height of the smaller images. Default is 300.
        patch_width (int, optional): The width of the smaller images. Default is 300.

    Returns:
        tuple: A tuple containing two lists. The first list, `patches`, contains the smaller images as NumPy arrays.
        The second list, `patches_with_coords`, contains tuples, each containing a smaller image and its starting coordinates (i and j) in the original image.
    """
    patches = []
    patches_with_cords = []
    width, height, _ = image.shape
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = image[j:j + patch_width, i:i + patch_height]
            patches.append(patch)
            patches_with_cords.append((patch, i, j))
    return np.array(patches), np.array(patches_with_cords)


def save_patches(image_name, patches, dir_path):
    """
  Saves the given patches with the original image name as the prefix in the specified directory.

  Parameters:
      image_name (str): the original image name to use as the prefix for the file names
      patches (list): a list of patches (images) to save
      dir_path (str): the directory to save the patches in

  Returns:
    None
  """
    for i, patch in enumerate(patches):
        cv2.imwrite(f"{dir_path}/{image_name}_p{i}.jpg", patch)


def apply_function_to_images(images, func):
    """Apply a function to each image in a list of images.

    Parameters:
        images: A list of images, represented as numpy.ndarray objects.
        func: The function to apply to each image. This function should take a single
            image as an argument and return the modified image.

    Returns:
        A list of modified images.
    """
    return [func(image) for image in images]


def filter_blurred(images, threshold=50):
    """
    Filter out blurred images from a list of images.

    Args:
        images (list): A list of images, each represented as a NumPy array.
        threshold (int, optional): The minimum variance of the Laplacian of the image to consider it not blurred. Default is 50.

    Returns:
        list: A list of images that are not blurred, represented as NumPy arrays.
    """
    not_blurred_images = []
    for image in images:
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Compute the variance of the Laplacian of the image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # If the variance is above the threshold, consider the image not blurred
        if variance > threshold:
            not_blurred_images.append(image)

    return not_blurred_images


def filter_monochromatic(images, tolerance=30):
    """
    Filter out monochromatic images from a list of images.

    Args:
        images (list): A list of images, each represented as a NumPy array.
        tolerance (int, optional): The maximum tolerance for color variations in the image. Default is 30.

    Returns:
        list: A list of images that are not monochromatic, represented as NumPy arrays.
    """
    not_monochromatic_images = []
    for image in images:
        # Convert the image to the HSV color space
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # Split the image into its channels
        h, s, v = cv2.split(hsv)

        # Calculate the standard deviation of each channel
        h_std = np.std(h)
        s_std = np.std(s)
        v_std = np.std(v)

        # If the standard deviation of any channel is above the tolerance, consider the image not monochromatic
        if h_std > tolerance or s_std > tolerance or v_std > tolerance:
            not_monochromatic_images.append(image)

    return not_monochromatic_images
