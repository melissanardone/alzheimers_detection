# 
# extract_slices.py
# Author: Melissa Nardone
# Description: Extracts the MRI slices from the ADNI dataset.
#

import os
import shutil

SRC_DIR = 'ADNI/'
DEST_DIR = ''

def extract_source_files(base_dir):
    print('Extracting source files...')
    total_src_files = 0

    img_dir = os.path.join(base_dir, SRC_DIR)
    dst_dir = os.path.join(base_dir, DEST_DIR)

    # loop through all mri images
    files = list()
    for (dirpath, dirnames, filenames) in os.walk(img_dir):
        files += [os.path.join(dirpath, file) for file in filenames]

    for file in files:
        _, fname = os.path.split(file)
        dest_file = os.path.join(dst_dir, fname)
        shutil.copyfile(file, dest_file)
        total_src_files += 1
    
    print('\tTotal files: ' + str(total_src_files))