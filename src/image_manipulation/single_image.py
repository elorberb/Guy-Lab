import imghdr
import pandas as pd
from PIL import Image
import numpy as np
import cv2
import os


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


def dilation(image, kernel_size):
    """
    Dilates the given image by adding pixels around lighter areas.

    Parameters:
    image (numpy array): the image to dilate
    kernel_size (int): the size of the kernel to use for dilation

    Returns:
    numpy array: the dilated image
    """
    # Create a kernel of the specified size
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    # Apply dilation to the image
    dilated_image = cv2.dilate(image, kernel)

    return dilated_image


def rgb_to_hsv(image):
    """
    Converts the given image from RGB to HSV color space.

    Parameters:
    image (numpy array): the image to convert

    Returns:
    numpy array: the image in HSV color space
    """
    # Convert the image from RGB to HSV
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    return hsv_image


def rgb_to_lab(image):
    """
    Converts the given image from RGB to LAB color space.

    Parameters:
    image (numpy array): the image to convert

    Returns:
    numpy array: the image in LAB color space
    """
    # Convert the image from RGB to LAB
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    return lab_image


def contrast_stretch(image):
    """
    Performs contrast stretching on the given image.

    Parameters:
    image (numpy array): the image to stretch

    Returns:
    numpy array: the contrast-stretched image
    """
    # Perform contrast stretching on the image
    stretched_image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    return stretched_image
