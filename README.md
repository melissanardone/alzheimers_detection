# Alzheimer's Detection with the DWT and CNNs

The provided scripts can be used for creating and training a model for Alzheimer's Detection.

## Description

The model included in the scripts incorporates the DWT and a CNN for feature extraction and classification. The detail coefficients from the second level of decomposition from the DWT are used as the input for CNN. The training and testing dataset selected was from ADNI1 dataset. The dataset includes MRI images that are organized into classes based on cognitive qualifications. Each script contains parameters which are noted that can be used for altering the model performance.

## Getting Started

### Dataset

Request access to the dataset by following the link provided below. Request access the the ADNI1 dataset, and, once access has been granted, download the MRI brain scans in the NIfTI format along with their classification labels. The MRI brain scans will require 80GB of storage.

[ADNI Data Use Agreement](https://ida.loni.usc.edu/collaboration/access/appLicense.jsp;jsessionid=885703F42AB44DAEFC824B4D1D034291)

### Dependencies

Install the following dependencies:
* Python
* TensorFlow
* Keras
* OpenCV Python
* NumPy
* Pandas
* Scikit-Learn
* NiBabel
* PyWavelets

### Setup

1. Download the ADNI1 dataset.
2. Note the parameters listed below before executing any of the scripts. 
3. Scripts do not autogenerate folders, and folders must be created before execution of the scripts.

### Execution

extract_source_files.py
* Description: Extracts the MRI slices from the ADNI dataset.
* Parameters: 
    * SRC_DIR - Source directory.
    * DEST_DIR - Destination directory.
```
py extract_source_files.py
```
extract_slices.py
* Description: Extracts the MRI slices from the ADNI dataset.
* Parameters:
    * SRC_DIR - Source directory.
    * DEST_DIR - Destination directory. Must include 3 folders corresponding to AD, MCI, and NL.
```
py extract_slices.py
```
wavelet_transform.py
* Description: Performs feature extraction using the wavelet transform.
* Parameters:
    * SRC_DIR - Source directory. 
    * DEST_DIR - Destination directory. Must include 3 folders corresponding to AD, MCI, and NL.
    * WAVELET - Wavelet type. Example: haar
```
py wavelet_transform.py
```
cnn_model.py
* Description: Creates and trains a CNN model for Alzheimer's detection.
* Parameters:
    * BALANCE DATA - Standardizes the amount of brain scans per classification. 
    * IMAGE_SHAPE - Corresponds to the size of the resulting DWT coefficients.
    * SLICE_COUNT - Corresponds to the number of slices used as input to the model per brain scan.
    * INDEX_OFFSET - Index offset of the intput slices.
    * BATCH_SIZE - Batch size for training the CNN.
    * EPOCHS - Epochs for training the CNN.
    * SRC_DIR - Source directory.
```
py cnn_model.py [test_id]
```

## Authors

Melissa Nardone

Email: [mnardone@calpoly.edu](mnardone@calpoly.edu)

Alternative Email: [melissa.n.nardone@gmail.com](melissa.n.nardone@gmail.com)


