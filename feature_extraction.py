import cv2
import numpy as np

def horizontalProfileProjection(binaryImage, noOfBins):
    hpp = np.sum(1-binaryImage//255, axis=1)
    if hpp.size < noOfBins:
        noOfBins = hpp.size
    hpp = np.array([np.mean(bn) for bn in np.array_split(hpp, noOfBins)])
    return hpp

def extract_hog_features(img,target_img_size=(256, 128)):
    """
    You won't implement anything in this function. You just need to understand it 
    and understand its parameters (i.e win_size, cell_size, ... etc)
    """
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

