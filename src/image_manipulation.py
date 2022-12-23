import imghdr
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import os


# ----- Single Image Manipulation -----


def resize_image(image, width, height):
    """Resizes an image to the specified width and height.

    Parameters:
        image (ndarray): The image to resize.
        width (int): The width to resize the image to.
        height (int): The height to resize the image to.

    Returns:
        ndarray: The resized image.
    """
    # Resize the image
    resized_image = cv2.resize(image, (width, height))

    return resized_image


def contrast(image):
    """Enhances the contrast of an image using the Adaptive Histogram Equalization (CLAHE) method.

    Parameters:
        image (ndarray): The image to enhance.

    Returns:
        ndarray: The enhanced image.
    """
    # Convert the image to the LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    # Split the LAB image into its channels
    l_channel, a, b = cv2.split(lab)
    # Create a CLAHE object with a clip limit of 2.0 and a tile grid size of 8x8
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Enhance the contrast of the L channel using the CLAHE object
    cl = clahe.apply(l_channel)
    # Merge the enhanced L channel with the a and b channels
    limg = cv2.merge((cl, a, b))
    # Convert the enhanced image back to the BGR color space
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    # Return the enhanced image
    return enhanced_img


def sharpen(image):
    """Sharpens an image using a 3x3 kernel.

    Parameters:
        image (ndarray): The image to sharpen.

    Returns:
        ndarray: The sharpened image.
    """
    # Create a 3x3 kernel for sharpening
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

    # Apply the kernel to the image using the filter2D function
    sharpened_image = cv2.filter2D(image, -1, kernel)

    return sharpened_image


def reduce_noise(image, strength=20, h=10, hColor=7, templateWindowSize=21):
    """Reduces noise in an image using the Fast Non-Local Means Denoising algorithm.

    Parameters:
        image (ndarray): The image to denoise.
        strength (int, optional): The strength of the denoising. Default is 20.
        h (int, optional): The filter strength for the color component. Default is 10.
        hColor (int, optional): The filter strength for the coordinate component. Default is 7.
        templateWindowSize (int, optional): The size of the template patch that is used for weighted averaging. Default is 21.

    Returns:
        ndarray: The denoised image.
    """
    # Apply the Fast Non-Local Means Denoising algorithm to the image
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, strength, h, hColor, templateWindowSize)

    return denoised_image


def threshold(image, threshold_value):
    """
    Thresholds the given image by setting all pixels above the threshold value to white and all pixels below the threshold value to black.

    Parameters:
    image (numpy array): the image to threshold
    threshold_value (int): the threshold value

    Returns:
    numpy array: the thresholded image
    """
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to the image
    _, binary_image = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    return binary_image


# ----- Multiple Images Manipulation -----


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


if __name__ == '__main__':
    path = 'C:\\Users\\elorberb\\PycharmProjects\\BGU projects\\Guy-Lab\\trichomes_images\\camera_day2\\Camera2'
    imgs = read_images(path)
    print(imgs[0])
