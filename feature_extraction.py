import cv2
import numpy as np
from scipy.signal import convolve2d

def horizontalProfileProjection(binaryImage, noOfBins):
    hpp = np.sum(1-binaryImage//255, axis=1)
    if hpp.size < noOfBins:
        noOfBins = hpp.size
    hpp = np.array([np.mean(bn) for bn in np.array_split(hpp, noOfBins)])
    hpp /= np.max(hpp)
    return hpp

def numberOfVerticalHorizontalLines(edges):
    horizontal = np.copy(1-edges)
    vertical = np.copy(1-edges)

    cols = horizontal.shape[1]
    horizontal_size = cols // 20
    horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.morphologyEx(horizontal, cv2.MORPH_OPEN, horizontalStructure)

    rows = vertical.shape[0]
    verticalsize = rows // 20
    verticalStructure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, verticalsize))
    vertical = cv2.morphologyEx(vertical, cv2.MORPH_OPEN, verticalStructure)

    # show_images([horizontal], ['horizontal lines'])
    # show_images([vertical], ['vertical lines'])

    HL = cv2.connectedComponents(horizontal)[0]-1
    VL = cv2.connectedComponents(vertical)[0]-1

    # print(f'number of horizontal lines:{HL}')
    # print(f'number of vertical lines:{VL}')

    return HL, VL

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

def localPhaseQuantization(img, winSize=3):
    
    STFTalpha=1/winSize

    img=np.float64(img)
    r=(winSize-1)/2
    x=np.arange(-r,r+1)[np.newaxis]

    w0=np.ones_like(x)
    w1=np.exp(-2*np.pi*x*STFTalpha*1j)
    w2=np.conj(w1)

    filterResp1=convolve2d(convolve2d(img,w0.T,'valid'), w1, 'valid')
    filterResp2=convolve2d(convolve2d(img,w1.T,'valid'), w0, 'valid')
    filterResp3=convolve2d(convolve2d(img,w1.T,'valid'), w1, 'valid')
    filterResp4=convolve2d(convolve2d(img,w1.T,'valid'), w2, 'valid')

    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)


   
    LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]
    
    LPQdesc=LPQdesc/LPQdesc.sum()

    return LPQdesc

def extractFeaturesFromImage(img):
    features = localPhaseQuantization(img)
    return features