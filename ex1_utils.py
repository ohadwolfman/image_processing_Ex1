"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List

import numpy as np
import cv2
import matplotlib.pyplot as plt

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 316552496


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    # Read the image using OpenCV
    img = cv2.imread(filename)

    # Convert to grayscale if requested
    if representation == 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # Convert to RGB if requested
    elif representation == 2:
        # When the image file is read with the OpenCV function imread() the order of colors is BGR
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        raise ValueError("Wrong representation argument, 1 is for gray scale and 2 is for RGB")

    # Return the image as a numpy array
    return img / 255


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)
    print(img.shape)
    if representation == 1:
        plt.imshow(img, cmap='gray')  # cmap = color map

    elif representation == 2:
        plt.imshow(img)

    else:
        raise ValueError("Wrong representation argument, 1 is for gray scale and 2 is for RGB")

    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    transfer_matrix = np.array([[0.299, 0.587, 0.114],
                                [0.596, -0.275, -0.321],
                                [0.212, -0.523, 0.311]])
    OrigShape = imgRGB.shape
    newYIQ = np.dot(imgRGB.reshape(-1, 3), transfer_matrix.transpose()).reshape(OrigShape)
    #print(f"image shape: {newYIQ.shape}")
    return newYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    yiq_from_rgb = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgYIQ.shape
    return np.dot(imgYIQ.reshape(-1, 3), np.linalg.inv(yiq_from_rgb).transpose()).reshape(OrigShape)


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :ret
    """
    processingImg = np.copy(imgOrig)
    origSize = processingImg.shape

    if (imgOrig.ndim == 3):
        yiqImage = transformRGB2YIQ(imgOrig)
        processingImg = yiqImage[:, :, 0]
        origSize = processingImg.shape
        processingImg = ((processingImg.flatten()) * 255).astype(np.uint8)
        histOrg, bins = np.histogram(processingImg.flatten(), bins=256, range=[0, 256])

    else:  ##(imgOrig.ndim == 2):
        processingImg = ((processingImg.flatten()) * 255).astype(np.uint8)
        histOrg, bins = np.histogram(processingImg.flatten(), bins=256, range=[0, 256])

    cdf = np.cumsum(histOrg)
    cdf_normalized = cdf / cdf[-1]

    new_colors = (255 * cdf_normalized).astype(np.uint8)  # lookup table

    for pixel in range(processingImg.shape[0]):
        old_color = processingImg[pixel]
        processingImg[pixel] = new_colors[old_color]

    processingImg = processingImg.reshape(origSize)
    imEq = processingImg

    if (imgOrig.ndim == 3):
        yiqImage[:, :, 0] = processingImg
        imEq = ((transformYIQ2RGB(yiqImage)) * 255).astype(np.uint8) / 255

    histEQ, binsEq = np.histogram(imEq.flatten(), bins=256, range=[0, 256])
    imEq = imEq / 255
    plt.plot(histEQ, color='blue')
    plt.legend(('histEQ', 'imgEq'), loc='upper left')
    plt.show()
    return imEq, histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    pass
