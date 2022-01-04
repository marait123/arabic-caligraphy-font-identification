import cv2
import numpy as np
from preprocessing import show_images

def horizontalProfileProjection(binaryImage, noOfBins):
    hpp = np.sum(1-binaryImage//255, axis=1)
    if hpp.size < noOfBins:
        noOfBins = hpp.size
    hpp = np.array([np.mean(bn) for bn in np.array_split(hpp, noOfBins)])
    hpp /= np.max(hpp)
    return hpp

def extract_hog_features(img,target_img_size=(256, 128)):
    
    img = cv2.resize(img, target_img_size)
    
    cell_size = (32, 32)
    block_size = (2, 2)
    nbins = 9  # Number of orientation bins
    hog = cv2.HOGDescriptor(_winSize=(img.shape[1] // cell_size[1] * cell_size[1],
                                      img.shape[0] // cell_size[0] * cell_size[0]),
                            _blockSize=(block_size[1] * cell_size[1],
                                        block_size[0] * cell_size[0]),
                            _blockStride=(cell_size[1], cell_size[0]),
                            _cellSize=(cell_size[1], cell_size[0]),
                            _nbins=nbins)

    h = hog.compute(img)
    h = h.flatten()
    return h.flatten()


def extract_EOH(img, bins = 360):
    dx, dy = cv2.spatialGradient(img)
    dx = dx.astype("float32")
    dy = dy.astype("float32")
    gradient_magnitude = cv2.magnitude(dy, dx)
    
    gradient_orientation = cv2.phase(dx, dy, angleInDegrees=True)
    eoh = np.histogram(gradient_orientation, bins=bins, range=(0, 360), weights=gradient_magnitude, density=True)
    return eoh[0]

