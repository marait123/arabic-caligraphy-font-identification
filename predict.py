import sys
import os
import cv2
from torch.functional import Tensor
from preprocessing import preprocessImage
from feature_extraction import extractFeaturesFromImage
from classifier import getNNModelStructure, predict
import torch
import time
import numpy as np

featuresNo = 255

def process_data(input_path, output_path):
    try:
        files = os.listdir(input_path)
    except:
        print('ERROR: input path is not correct')
        sys.exit()
    

    try:
        resultsFile = open(f"{output_path}/results.txt", "w")
        timeFile = open(f"{output_path}/times.txt", "w")
    except:
        print("output path isn't correct")
        sys.exit()
    

    files.sort()

    model = getNNModelStructure(featuresNo)
    
    try:
        model.load_state_dict(torch.load('model.pth'))
    except:
        print("model isn't in the same path of predict.py")
        sys.exit()

    
    _ = predict(model, torch.zeros(1, featuresNo))

    for file in files:
        _, extension = os.path.splitext(file)
        # if extension.lower() != '.png':
        img = cv2.imread(f'{input_path}/{file}', cv2.IMREAD_GRAYSCALE)
        t1 = time.time()
        preprocessed = preprocessImage(img)
        features = extractFeaturesFromImage(preprocessed)
        features = features.reshape(1, features.size)
        features = torch.from_numpy(features)
        prediction = predict(model, features) + 1
        timeFile.write(f'{np.round(time.time()-t1, 2)}\n')
        resultsFile.write(f'{prediction.item()}\n')

    resultsFile.close()
    timeFile.close()

if __name__ == "__main__":
    arg_count = len(sys.argv) 
    
    if arg_count < 3:
        print('ERROR: command must be in format: "python predict.py $input_path $output_path"')
        sys.exit()

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    process_data(input_path, output_path)
    
