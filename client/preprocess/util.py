import cv2
from PIL import Image, ImageEnhance
import numpy as np


def rm_hidden_letter(img: np.ndarray) -> np.ndarray:
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)


def rm_stamp(img: np.ndarray) -> np.ndarray:
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red = np.array([0, 50, 50])
    upper_red = np.array([30, 255, 255])
    mask0 = cv2.inRange(hsv_image, lower_red, upper_red)

    lower_red = np.array([160, 50, 50])
    upper_red = np.array([180, 255, 255])
    mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
    mask = mask0 + mask1

    img[mask > 0] = [255, 255, 255]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
