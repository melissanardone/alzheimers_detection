#
# cnn_model.py
# Author: Melissa Nardone
# Description: Creates and trains a CNN model for Alzheimer's detection.
# Usage: cnn_model.py [test_id]
#

import os
import pandas as pd
import tensorflow as tf
import numpy as np
import imutils
import cv2
import time
import random
import sys

from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout, Concatenate, Input, BatchNormalization
from keras import backend as K
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import (ModelCheckpoint)
from matplotlib import pyplot as plt
from skimage.util import random_noise
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, ConfusionMatrixDisplay
from keras.callbacks import Callback
from sklearn import metrics

###### Model and training parameters #####
BALANCE_DATA = 1 # equalize the number of brain scans in each category

IMG_SHAPE = (38, 38, 3) # input coefficient size per slices 
SLICE_COUNT = 32 # slice count
BATCH_SIZE = 10
EPOCHS = 15 
INDEX_OFFSET = 90 # slice initial offset index

IMG_DIR = '' # image directory pointing to the DWT coefficients

# Define the classifications and classification labels
#label_dict = {'AD': 0, 'MCI': 1, 'NL': 2}
label_dict = {'AD': 0, 'NL': 1}

classification_labels = list(label_dict.keys())

###### Generate Model #####
inputs = []
outputs = []

# CNN Model 
for i in range(SLICE_COUNT):
    input = Input(IMG_SHAPE)
    inputs.append(input)
    conv1 =  Conv2D(8, (5, 5), activation='relu')(input)
    max_pool1 = MaxPooling2D((2, 2))(conv1)
    conv2 = Conv2D(16, (3, 3), activation='relu')(max_pool1)
    max_pool2 = MaxPooling2D((2, 2))(conv2)
    outputs.append(max_pool2)

if SLICE_COUNT == 1: 
    flatten = Flatten()(outputs[0]) 
else:
    comb = Concatenate()(outputs)
    flatten = Flatten()(comb)

dense1 = Dense(32, activation='relu')(flatten)
dense2 = Dense(16, activation='relu')(dense1)
dropout = Dropout(0.3)(dense2)

output = Dense(len(label_dict), activation='softmax')(dropout)

model = Model(inputs=[inputs], outputs=[output])

print(model.summary())
tf.keras.utils.plot_model(model, to_file="test.png", rankdir="TB", show_shapes=True)

##### Data Generator #####
def generate_dataframe():
    df = pd.DataFrame(columns=['img_id', 'classification'])

    # loop through all mri images
    for (dirpath, _, filenames) in os.walk(IMG_DIR):
        for file in filenames:
            data = file.rsplit('_')
            id = int(data[1])
            slice = data[2].rsplit('.')[-2]
            classification = os.path.split(dirpath)[1]

            if int(slice[5]) == 0:
                if classification in list(label_dict.keys()):
                    df = df.append({'img_id': id, 'classification': classification}, ignore_index=True)

    df = df.sort_values(by=['img_id'], ignore_index=True)

    return df

df = generate_dataframe()

img_ids = df['img_id'].unique()
img_ids.sort()
image_df = df

##### Remove dataset imbalances #####
if BALANCE_DATA:
    class_df_list = []
    class_counts = []

    # get the id count for each classification
    for label in list(label_dict.keys()):
        class_df = image_df[image_df['classification'] == label]
        class_df = class_df.reset_index(drop=True)
        class_df_list.append(class_df)
        class_counts.append(class_df.shape[0])

    min_class_count = min(class_counts)
    min_class_idx = class_counts.index(min_class_count)
    min_class_label = list(label_dict.keys())[min_class_idx]

    # generate dataframe with balanced classes
    balanced_image_df = pd.DataFrame(columns=['img_id', 'classification'])
    for df in class_df_list:
        if df.shape[0] > min_class_count:
            df = df.drop(range(min_class_count, df.shape[0])) # drop image ids to make all classes equivalent
        print(df['classification'][0] + ': ' + str(df.shape[0]))
        # add dataframe back
        balanced_image_df = balanced_image_df.append(df)

    balanced_image_df = balanced_image_df.reset_index(drop=True)
    print(balanced_image_df)

    image_df = balanced_image_df

# train/test/validation split, 70%, 20%, 10%
train_images, test_validate_images = train_test_split(image_df, test_size=0.3, random_state=42)
validation_images, test_images = train_test_split(test_validate_images, test_size=0.67, random_state=42)

print('\nTraining IDs: ', len(train_images))
print('Validation IDs: ', len(validation_images))
print('Test IDs: ', len(test_images))

# save training/validation/test ids and labels to CSV
train_images.to_csv('training_images.csv', index=False)
validation_images.to_csv('validation_images.csv', index=False)
test_images.to_csv('test_images.csv', index=False)

# loading images helper
def load_image(path):
    if IMG_SHAPE[2] == 1:
        im = cv2.imread(path, 0) # greyscale
    else:
        im = cv2.imread(path)

    im = np.array(im) / 255 # normalize

    return im

##### Data Generator #####
class DataGenerator(Sequence):
    def __init__(self, img_ids, labels, to_fit=True, batch_size=10, dim=(IMG_SHAPE[0], IMG_SHAPE[1]), n_channels=IMG_SHAPE[2], shuffle=True):
        self.img_ids = img_ids
        self.num_inputs = SLICE_COUNT
        self.num_classes = len(label_dict)
        self.labels = labels
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.img_ids) / self.batch_size))

    def __getitem__(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        
        # Locate the list of image _ids
        img_ids_temp = [self.img_ids[k] for k in indexes]

        # Generate data
        X = self._generate_X(img_ids_temp)

        if self.to_fit:
            y = self._generate_y(indexes)
            return X, y
        else:
            return X

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.img_ids))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def _generate_X(self, img_ids_temp):
        # list, with an array for each input
        X = [np.empty((self.batch_size, *self.dim, self.n_channels)) for i in range(self.num_inputs)]

        for i, img_id in enumerate(img_ids_temp):
            label_idx = np.where((self.img_ids == img_id))[0][0]
            classification = self.labels[label_idx]

            # load images
            for slice in range(SLICE_COUNT):
                path = os.path.join(IMG_DIR, classification, 'img_' + str(img_id) + '_slice' + str(slice+INDEX_OFFSET) + '.jpg')
                image = load_image(path)
                X[slice][i,] = image.reshape(IMG_SHAPE)

        return X

    def _generate_y(self, indexes):
        y = np.empty((self.batch_size, self.num_classes), dtype=int)
        # Generate data
        for i, index in enumerate(indexes):
            # Store sample
            y[i,] = label_dict[self.labels[index]]

        y = y.T

        # perform one-hot encoding
        return (keras.utils.to_categorical(y[0], num_classes=len(classification_labels)))

def split_classification_data(data):
    x = data['img_id'].to_numpy()
    y = data['classification'].to_numpy()
    
    return x, y

X_train, y_train = split_classification_data(train_images)
X_test, y_test = split_classification_data(test_images)
X_valid, y_valid = split_classification_data(validation_images)

train_generator = DataGenerator(X_train, y_train, batch_size=BATCH_SIZE)
validation_generator = DataGenerator(X_valid, y_valid, batch_size=BATCH_SIZE)

##### Train Model #####
training_metrics = ['categorical_accuracy']

# compile model
opt = tf.keras.optimizers.Adam()

model.compile(loss='categorical_crossentropy',
            optimizer=opt,
            metrics=training_metrics)

start = time.time()

# Test ID
test_id = sys.argv[1]

# for saving only the model with the best performance
model_path = 'model_' + test_id
checkpoint = ModelCheckpoint(model_path, monitor='categorical_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

print("Batch_size:", BATCH_SIZE, " \tStart time:", time.strftime("%H:%M:%S", time.gmtime(start)))

# train the model
H = model.fit(
    train_generator, 
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator, 
    validation_steps=len(validation_generator),
    epochs=EPOCHS,
    callbacks=callbacks_list)

end = time.time()
total_time = end-start
print("Training Complete. \tTotal Time: ", time.strftime("%H:%M:%S", time.gmtime(total_time)))

# Model testing
history = H
print(history.history.keys())

# summarize history for accuracy
plt.figure()
plt.plot(history.history['categorical_accuracy'], 'ok-')
plt.plot(history.history['val_categorical_accuracy'], 'ok--')
plt.title('Model Training Accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='upper left')
plt.savefig('model_accuracy_' + test_id + '.png')

# summarize history for loss
plt.figure()
plt.plot(history.history['loss'], 'ok-')
plt.plot(history.history['val_loss'], 'ok--')
plt.title('Model Training Loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training', 'validation'], loc='lower left')
plt.savefig('model_loss_' + test_id + '.png')

# Get the best saved model
model = tf.keras.models.load_model('model_' + test_id)

train_generator = DataGenerator(X_train, y_train, batch_size=BATCH_SIZE, shuffle=False)
validation_generator = DataGenerator(X_valid, y_valid, batch_size=BATCH_SIZE, shuffle=False)
test_generator = DataGenerator(X_test, y_test, batch_size=BATCH_SIZE, shuffle=False)

##### Model Testing #####

##### Generate Test Confusion Matrix #####
# generate predictions for test set for confusion matrix
y_pred = model.predict(test_generator)

y_pred = [classification_labels[y[0]] for y in (y_pred > 0.5).astype(int)]
metrics_dict = metrics.classification_report(y_test[0:len(y_pred)], y_pred, target_names=classification_labels, digits=4, output_dict=True)
print(metrics.classification_report(y_test[0:len(y_pred)], y_pred, target_names=classification_labels, digits=4))

matrix = confusion_matrix(y_test[0:len(y_pred)], y_pred, labels=classification_labels)

accuracy = metrics_dict['accuracy']
f1_score = metrics_dict['weighted avg']['f1-score']
precision = metrics_dict['weighted avg']['precision']
recall = metrics_dict['weighted avg']['recall']

# save confusion matrix
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classification_labels)
disp = disp.plot(cmap=plt.cm.Greys, colorbar=False)
plt.savefig('test_confusion_matrix_' + test_id + '.png')

##### Generate Validation Confusion Matrix #####
y_pred = model.predict(validation_generator)

y_pred = [classification_labels[y[0]] for y in (y_pred > 0.5).astype(int)]

matrix = confusion_matrix(y_valid[0:len(y_pred)], y_pred, labels=classification_labels)

# save confusion matrix
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classification_labels)
disp = disp.plot(cmap=plt.cm.Greys, colorbar=False)
plt.savefig('validation_confusion_matrix_' + test_id + '.png')

##### Generate Training Confusion Matrix #####
y_pred = model.predict(train_generator)

y_pred = [classification_labels[y[0]] for y in (y_pred > 0.5).astype(int)]

matrix = confusion_matrix(y_train[0:len(y_pred)], y_pred, labels=classification_labels)

# save confusion matrix
plt.figure()
disp = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=classification_labels)
disp = disp.plot(cmap=plt.cm.Greys, colorbar=False)
plt.savefig('training_confusion_matrix_' + test_id + '.png')

# save the test performance in a CSV
test_performance_dict = {'training time': [time.strftime("%H:%M:%S", time.gmtime(total_time))], 'accuracy': [accuracy], 'f1_score': [f1_score], 'precision': [precision], 'recall': [recall]}
test_peformance_df = pd.DataFrame(test_performance_dict)
test_peformance_df.to_csv('test_performance_' + test_id + '.csv')

print('TEST ID: ' + test_id)