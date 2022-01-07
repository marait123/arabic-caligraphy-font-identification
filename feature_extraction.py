import cv2
import numpy as np
from preprocessing import show_images
from scipy.signal import convolve2d

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

def lpq(img,winSize=3,freqestim=1,mode='nh'):
    # rho=0.90

    STFTalpha=1/winSize  # alpha in STFT approaches (for Gaussian derivative alpha=1)
    # sigmaS=(winSize-1)/4 # Sigma for STFT Gaussian window (applied if freqestim==2)
    # sigmaA=8/(winSize-1) # Sigma for Gaussian derivative quadrature filters (applied if freqestim==3)

    convmode='valid' # Compute descriptor responses only on part that have full neigborhood. Use 'same' if all pixels are included (extrapolates np.image with zeros).

    img=np.float64(img) # Convert np.image to double
    r=(winSize-1)/2 # Get radius from window size
    x=np.arange(-r,r+1)[np.newaxis] # Form spatial coordinates in window

    if freqestim==1:  #  STFT uniform window
        #  Basic STFT filters
        w0=np.ones_like(x)
        w1=np.exp(-2*np.pi*x*STFTalpha*1j)
        w2=np.conj(w1)

    ## Run filters to compute the frequency response in the four points. Store np.real and np.imaginary parts separately
    # Run first filter
    filterResp1=convolve2d(convolve2d(img,w0.T,convmode),w1,convmode)
    filterResp2=convolve2d(convolve2d(img,w1.T,convmode),w0,convmode)
    filterResp3=convolve2d(convolve2d(img,w1.T,convmode),w1,convmode)
    filterResp4=convolve2d(convolve2d(img,w1.T,convmode),w2,convmode)

    # Initilize frequency domain matrix for four frequency coordinates (np.real and np.imaginary parts for each frequency).
    freqResp=np.dstack([filterResp1.real, filterResp1.imag,
                        filterResp2.real, filterResp2.imag,
                        filterResp3.real, filterResp3.imag,
                        filterResp4.real, filterResp4.imag])

    ## Perform quantization and compute LPQ codewords
    inds = np.arange(freqResp.shape[2])[np.newaxis,np.newaxis,:]
    LPQdesc=((freqResp>0)*(2**inds)).sum(2)

    ## Switch format to uint8 if LPQ code np.image is required as output
    if mode=='im':
        LPQdesc=np.uint8(LPQdesc)

    ## Histogram if needed
    if mode=='nh' or mode=='h':
        LPQdesc=np.histogram(LPQdesc.flatten(),range(256))[0]

    ## Normalize histogram if needed
    if mode=='nh':
        LPQdesc=LPQdesc/LPQdesc.sum()

    return LPQdesc

def extractFeaturesFromImage(img):
    features = lpq(img)
    return features