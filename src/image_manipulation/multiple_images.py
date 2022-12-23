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
    Cut an image into smaller images of a given height and width.

    Parameters:
        image (ndarray): The image to be cut, represented as a NumPy array.
        patch_height (int, optional): The height of the smaller images. Default is 300.
        patch_width (int, optional): The width of the smaller images. Default is 300.

    Returns:
        list: A list of tuples, each containing a smaller image (as a NumPy array) and the starting coordinates (i and j) of the image in the original image.
    """
    patches = []
    width, height, _ = image.shape
    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = image[j:j + patch_width, i:i + patch_height]
            patches.append((patch, i, j))
    return np.array(patches)


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