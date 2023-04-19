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

    img = img.astype(np.float)
    norm_img = normalizeData(img)
    return norm_img


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
    reshapedImg = imgRGB.reshape(-1, 3)
    newYIQ = np.dot(reshapedImg, transfer_matrix.transpose()).reshape(OrigShape)
    #print(f"image shape: {newYIQ.shape}")
    return newYIQ


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    transfer_matrix = np.array([[0.299, 0.587, 0.114],
                             [0.59590059, -0.27455667, -0.32134392],
                             [0.21153661, -0.52273617, 0.31119955]])
    OrigShape = imgYIQ.shape
    reshapedImg = imgYIQ.reshape(-1, 3)
    inverseTransfer_Matrix = np.linalg.inv(transfer_matrix)
    newRgb = np.dot(reshapedImg, inverseTransfer_Matrix.transpose()).reshape(OrigShape)
    return newRgb


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Equalizes the histogram of an image
    param imgOrig: Original image: values in the range [0; 255]
    return: (imgEq, histOrg, histEQ)
            imEq - the new image with the equalized histogram
            histOrg - the histogram of the original image
            histEQ -the histogram of the imgEq
    """
    processingImg = np.copy(imgOrig)*255
    origSize = processingImg.shape

    if imgOrig.ndim == 3:
        yiqImage = transformRGB2YIQ(imgOrig)
        processingImg = yiqImage[:, :, 0]
        processingImg = cv2.normalize(processingImg, None, 0, 255, cv2.NORM_MINMAX)
        origSize = processingImg.shape
        processingImg = processingImg.astype('uint8')
        histOrg, bins = np.histogram(processingImg.flatten(), bins=256, range=[0, 256])

    elif imgOrig.ndim == 2:
        processingImg = processingImg.astype('uint8')
        histOrg, bins = np.histogram(processingImg.flatten(), bins=256, range=[0, 256])
    else:
        raise "Please try to run this function on gray-scale or RGB image"

    cdf = np.cumsum(histOrg)
    cdf_normalized = cdf / cdf[-1]

    new_colors = np.round(cdf_normalized * 255)  # lookup table
    print("processingImg.shape",processingImg.shape)

    for row in range(processingImg.shape[0]):
        for pixel in range(processingImg.shape[1]):
            old_color = processingImg[row][pixel]
            processingImg[row][pixel] = int(new_colors[old_color])

    processingImg = processingImg.reshape(origSize)
    imEq = processingImg

    if imgOrig.ndim == 2:
        histEQ, binsEq = np.histogram(imEq.flatten(), bins=256, range=[0, 256])
        imEq = imEq / 255

    if imgOrig.ndim == 3:
        yiqImage[:, :, 0] = processingImg/255
        imEq = transformYIQ2RGB(yiqImage)
        histEQ, binsEq = np.histogram((imEq*255).flatten(), bins=256, range=[0, 256])

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
    yiqImage = 0
    processingImg = np.copy(imOrig)
    if imOrig.ndim == 3:
        yiqImage = transformRGB2YIQ(imOrig)
        processingImg = yiqImage[:, :, 0]

    processingImg = cv2.normalize(processingImg, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    Orig_copy = processingImg.copy()

    histOrg = np.histogram(processingImg.ravel(), bins=256)[0]
    cdf = np.cumsum(histOrg)
    slices_size = cdf[-1] / nQuant
    slices = [0]
    curr_sum = 0
    curr_index = 0
    for i in range(1, nQuant + 1):
        while curr_sum < slices_size and curr_index < 256:
            curr_sum += histOrg[curr_index]
            curr_index = curr_index + 1
        if slices[-1] != curr_index - 1:
            curr_index = curr_index - 1
        slices.append(curr_index)
        curr_sum = 0

    slices.pop()
    slices.insert(nQuant, 255)

    images_list = []
    MSE_errorList = []
    for i in range(nIter):
        quantizeImg = np.zeros(processingImg.shape)
        intensityAvg = []
        for j in range(1, nQuant + 1):
            # print(f'j={j}, nQuant={nQuant}, slices={slices}, slices size={len(slices)}')
            try:
                sliceIntensities = np.array(range(slices[j-1], slices[j]))
                Pi = histOrg[slices[j-1]:slices[j]]  # Number of times those intensities levels appears in the image.
                avg = int((sliceIntensities * Pi).sum() / Pi.sum())  # The intensity level that is the average of this slice
                intensityAvg.append(avg)
            except RuntimeWarning:
                intensityAvg.append(0)
            except ValueError:
                intensityAvg.append(0)

        for k in range(nQuant):
            quantizeImg[processingImg > slices[k]] = intensityAvg[k]

        slices.clear()
        for k in range(1, nQuant):
            slices.append(int((intensityAvg[k - 1] + intensityAvg[k]) / 2))

        slices.insert(0, 0)
        slices.insert(nQuant, 255)

        MSE_errorList.append((np.sqrt((Orig_copy*255 - quantizeImg) ** 2)).mean())
        processingImg = quantizeImg
        images_list.append(quantizeImg / 255)
        if checkMSE(MSE_errorList, nIter):
            break

    if imOrig.ndim == 3:
        for i in range(len(MSE_errorList)):
            yiqImage[:, :, 0] = images_list[i]
            images_list[i] = transformYIQ2RGB(yiqImage)

    return images_list, MSE_errorList


def normalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


# Boolean function that checks if the equivalent of the last 5 values of the MSE_list
def checkMSE(MSE_list: List[float], nIter: int) -> bool:
    if len(MSE_list) > nIter / 10:
        for i in range(2, int(nIter / 10) + 1):
            if MSE_list[-1] != MSE_list[-i]:
                return False
    else:
        return False
    return True
