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
import cv2
import numpy as np
title_window = 'Gamma Correction'
trackbar_name = 'Slider'
gamma_slider_max_val = 255
max_pix = 255
isColor = False
img = 0

def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    try:
        global image
        if rep == 1:
            image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        else:
            image = cv2.imread(img_path)

        cv2.namedWindow(title_window)
        cv2.createTrackbar(trackbar_name, title_window, 100, 200, on_trackbar)
        on_trackbar(100)
        cv2.waitKey()
    except:
        print("error")

def on_trackbar(bright):
    gamma = float(bright) / 100

    invGamma = 0
    if gamma != 0:
        invGamma = 1.0 / gamma
    gammaTable = np.array([((i / float(255)) ** invGamma) * 255 for i in np.arange(0, 255 + 1)]).astype("uint8")
    # Lut for the image according to te gamma table
    img = cv2.LUT(image, gammaTable)
    cv2.imshow(title_window, img)

def main():
    gammaDisplay('dark.jpg', 1)


if __name__ == '__main__':
    main()


