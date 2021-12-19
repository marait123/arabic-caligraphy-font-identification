import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def load_data():
    x = []
    y = []
    for classNum in range(1, 10):
        for filename in sorted(glob.glob(f'ACdata_base/{classNum}/*.jpg')):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            x.append(img)
            y.append(classNum)
    return np.asarray(x, dtype=object), np.asarray(y, dtype=object)

def binraization(img):
    blur = cv2.GaussianBlur(img,(3,3),0)
    _, binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binaryImage
