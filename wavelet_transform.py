# 
# wavelet_transform.py
# Author: Melissa Nardone
# Description: Performs feature extraction using the wavelet transform.
#

import os
import cv2
import pywt 

import numpy as np

WAVELET = 'haar'
SRC_DIR = '' 
DEST_DIR = ''

def wavelet_transform():
    print('Performing the discrete wavelet transform...')

    count = 0
    wavelet_type = WAVELET

    # loop through all mri images
    files = list()
    for (dirpath, _, filenames) in os.walk(SRC_DIR):
        files += [os.path.join(dirpath, file) for file in filenames]

    count = 0
    for file in files:
        img_data = cv2.imread(file, 0) # read in grayscale image
        img_resized = cv2.resize(img_data, (150, 150), interpolation = cv2.INTER_AREA)

        # wavelet transform [LH, HL, HH]
        coeffs = pywt.wavedec2(img_resized, level=2, wavelet=wavelet_type)

        classification = file.rsplit("\\")[-2]
        _, fname = os.path.split(file)
        fname = fname.replace('.jpg', '')

        # extract the approximation coefficients
        # path = os.path.join(DEST_DIR, 'LL', 'level2', classification, fname + '.jpg')
        # cv2.imwrite(path, coeffs[0])

        for level in range(1, 3):
            # form an image with the 3 channels representing the detail coefficients
            path = os.path.join(DEST_DIR, 'level' + str(3-level), classification, fname + '.jpg')
            data = np.array([coeffs[level][0], coeffs[level][1], coeffs[level][2]])
            cv2.imwrite(path, data.T)

        count = count + 1
    
    print('\tTotal images: ' + str(count))

wavelet_transform()