import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize

def load_data():
    x = []
    y = []
    for classNum in range(1, 10):
        for filename in sorted(glob.glob(f'ACdata_base/{classNum}/*.jpg')):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            x.append(img)
            y.append(classNum)
    return np.asarray(x, dtype=object), np.asarray(y)

def binraization(img):
    blur = cv2.GaussianBlur(img,(3,3),0)
    _, binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binaryImage = invertBlackBackground(binaryImage)
    return binaryImage

def invertBlackBackground(binaryImage):
    # numOfZeros = binaryImage[binaryImage==0].size
    # numOfOnes = binaryImage.size - numOfZeros

    if determine_background(binaryImage) == 0:
        binaryImage = 255 - binaryImage

    return binaryImage        

def diacriticsSegmentationClustering(binaryImage):
    numLabels, labels, stats, _ = cv2.connectedComponentsWithStats(1-binaryImage, 8, cv2.CV_32S)
    areas = stats[:, cv2.CC_STAT_AREA]

    mue1 = np.mean(areas[1:])
    mue2 = mue1/4

    c1, c2 = [], []

    for i in range(1, numLabels):    
        area = stats[i, cv2.CC_STAT_AREA]
        d1 = np.abs(mue1-area)
        d2 = np.abs(mue2-area)
        if d1 < d2:
            c1.append(i)
        else:
            c2.append(i)

    textOnly = np.where(np.isin(labels, c1), 0, 1).astype(binaryImage.dtype)
    diacritics = np.where(np.isin(labels, c2), 0, 1).astype(binaryImage.dtype)
    return textOnly, diacritics

def diacriticsSegmentationFloodFill(binaryImage):
    diacritics = binaryImage.copy()
    
    baseline_idx = ((1-diacritics).sum(axis=1)).argmax()
    
    starts = np.array((diacritics[baseline_idx, :-1] != 0) & (diacritics[baseline_idx, 1:] == 0))
    seeds = np.where(starts)[0] + 1

    for seed in seeds:
        cv2.floodFill(diacritics, None, (seed, baseline_idx), 1)

    textOnly = binaryImage + (1-diacritics)

    return textOnly, diacritics

def extractImagesSet(binaryImage):
    edges = 1 - cv2.Canny(binaryImage*255, 50, 150)//255
    skeleton = 1-skeletonize(1-binaryImage)
    textOnly, diacritics = diacriticsSegmentationFloodFill(binaryImage)
    return edges, skeleton, textOnly, diacritics

def split_data(X, Y):
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, train_size=0.6)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, train_size=0.5)
    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


def determine_background(img):
    wMax = np.max(np.sum(img == 255, axis=0))
    bMax = np.max(np.sum(img == 0, axis=0))
    if wMax > bMax:
        return 1
    elif bMax > wMax:
        return 0
    else:
        return -1
    
    