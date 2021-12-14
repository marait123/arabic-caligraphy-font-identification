import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

def load_data():
    x = []
    y = []
    for classNum in range(1, 10):
        for filename in sorted(glob.glob(f'ACdata_base/{classNum}/*.jpg')):
            img = cv2.imread(filename)
            x.append(img)
            y.append(classNum)
    return np.asarray(x, dtype=object), np.asarray(y, dtype=object)