# 
# extract_slices.py
# Author: Melissa Nardone
# Description: Extracts the MRI slices from the ADNI dataset.
#

import os
import numpy as np
import nibabel as nib
import cv2

from matplotlib import pyplot as plt

IMAGE_DIR = 'E:/ADNI1_Complete_1Yr_1.5T/fall2022/labeled'

def extract_slices(base_dir):
    print('Extracting and saving slices...')
    total_slices = 0

    # loop through all mri images
    files = list()
    for (dirpath, dirnames, filenames) in os.walk(IMAGE_DIR):
        files += [os.path.join(dirpath, file) for file in filenames]

    for file in files:
        img = nib.load(file)
        data = img.get_fdata()

        # get filename
        dirname, fname = os.path.split(file)
        fname = fname.replace('.nii', '')

        # get diagnosis directory
        diagnosis = os.path.split(dirname)[1]

        # axial slices
        for idx in range(75, 176, 1):
            slice = data[idx, :, :]

            # normalize image data
            if (np.max(slice) != 0):
                slice = 255 * ((slice - np.min(slice)) / (np.max(slice) - np.min(slice)))

            # convert image to jpeg
            dest_dir = os.path.join('D:/ADNI1_Complete_1Yr_1.5T/sample/whole/', diagnosis, fname + '_axial_slice' + str(idx) + '.jpg')
            cv2.imwrite(dest_dir, slice)  
            total_slices += 1

        # sagittal slices
        for idx in range(33, 134, 1):
            slice = data[:, idx, :]

            # normalize image data
            if (np.max(slice) != 0):
                slice = 255 * ((slice - np.min(slice)) / (np.max(slice) - np.min(slice)))

            # convert image to jpeg
            dest_dir = os.path.join('D:/ADNI1_Complete_1Yr_1.5T/sample/whole/', diagnosis, fname + '_sagittal_slice' + str(idx) + '.jpg')
            cv2.imwrite(dest_dir, slice)  
            total_slices += 1

        # coronal slices
        for idx in range(75, 176, 1):
            slice = data[:, :, idx]

            # normalize image data
            if (np.max(slice) != 0):
                slice = 255 * ((slice - np.min(slice)) / (np.max(slice) - np.min(slice)))

            # convert image to jpeg
            dest_dir = os.path.join('D:/ADNI1_Complete_1Yr_1.5T/sample/whole/', diagnosis, fname + '_coronal_slice' + str(idx) + '.jpg')
            cv2.imwrite(dest_dir, slice)  
            total_slices += 1
    
    print('\tTotal slices: ' + str(total_slices))

extract_slices('')