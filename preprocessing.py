import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import os, shutil
from sklearn.model_selection import train_test_split
from skimage.morphology import skeletonize

def show_images(images,titles=None):
    #This function is used to show image(s) with titles by sending an array of images and an array of associated titles..
    # images[0] will be drawn with the title titles[0] if exists
    # You aren't required to understand this function, use it as-is.
    n_ims = len(images)
    if titles is None: titles = ['(%d)' % i for i in range(1,n_ims + 1)]
    fig = plt.figure()
    n = 1
    for image,title in zip(images,titles):
        a = fig.add_subplot(1,n_ims,n)
        if image.ndim == 2: 
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
        plt.axis('off')
        n += 1
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_ims)
    plt.show() 

def load_data():
    x = []
    y = []
    for classNum in range(1, 10):
        for filename in sorted(glob.glob(f'ACdata_base/{classNum}/*.jpg')):
            img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            x.append(img)
            y.append(classNum)
    return np.asarray(x, dtype=object), np.asarray(y)

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
    textOnly, diacritics = diacriticsSegmentationClustering(binaryImage)
    return edges, skeleton, textOnly, diacritics

def getAreaOfInterest(originalImage):
    img = originalImage.copy()
#     show_images([img],["original"])

    edges = cv2.Canny(img, 30, 150)
    # show_images([edges], ['edges'])


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilate = cv2.dilate(edges, kernel, iterations=1)//255
    # show_images([dilate], ['dilate'])

    threshold = 10


    hp = (dilate).sum(axis=1)
    vp = (dilate).sum(axis=0)

    h_occurances = np.where(hp>threshold)[0]
    v_occurances = np.where(vp>threshold)[0]

    rs, re = h_occurances[0], h_occurances[-1]
    cs, ce = v_occurances[0], v_occurances[-1]

    cropped = img[rs:re+1, cs:ce+1]
    
    return cropped

def copyFiles(X, Y, dir_name):
    os.mkdir(dir_name)
    for i in range(1, 10):
        os.mkdir(f'{dir_name}/{str(i)}')

    for i in range(Y.size):
        shutil.copy(X[i], f'{dir_name}/{Y[i]}')

def split_data(X, Y):
    X_train, X_rem, Y_train, Y_rem = train_test_split(X, Y, train_size=0.6)
    X_valid, X_test, Y_valid, Y_test = train_test_split(X_rem, Y_rem, train_size=0.5)

    try: 
        os.mkdir('data') 
    except: 
        shutil.rmtree('data')
        os.mkdir('data')

    copyFiles(X_train, Y_train, 'data/train')
    copyFiles(X_valid, Y_valid, 'data/valid')
    copyFiles(X_test, Y_test, 'data/test')


def determine_background(img):
    wMax = np.max(np.sum(img == 255, axis=0))
    bMax = np.max(np.sum(img == 0, axis=0))
    if wMax > bMax:
        return 1
    elif bMax > wMax:
        return 0
    else:
        return -1
    

def preprocessImage(img):
    blur = cv2.GaussianBlur(img,(3,3),0)
    
    _, binaryImage = cv2.threshold(blur,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    if determine_background(binaryImage) == 0:
        binaryImage = 255 - binaryImage
    
    return binaryImage