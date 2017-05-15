#!/usr/bin/env python

import csv
import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D


CORRECTION = 0.1
DRIVING_LOGS = "./data/driving_log.csv"
IMG_PATH = "./data/IMG/{}"
INPUT_SHAPE = (160, 320, 3)
CROPPING = ((70, 25), (0, 0))
BATCH_SIZE=32


def read_image(path):
    path = path.split("/")[-1]
    image = cv2.imread(IMG_PATH.format(path))
    return image


def flip_image(image):
    return cv2.flip(image, 1)


def transform_data(samples, batch_size=BATCH_SIZE):
    """Returns a generator for X_train and y_train

    The size of the samples will be 5 times larger because of the
    augmentation.

    :param samples: lines from driving_logs.csv
    :param batch_size: number of samples to process each time
    """
    samples_size = len(samples)
    while True:
        shuffle(samples)
        for offset in range(0, samples_size, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                center_path, left_path, right_path = batch_sample[:3]
                clr_images = [read_image(elem) for elem in
                              [center_path, left_path, right_path]]
                images.extend(clr_images)

                center_measurement = float(batch_sample[3])
                clr_measurements = [
                    center_measurement,
                    center_measurement + CORRECTION,
                    center_measurement - CORRECTION,
                ]
                measurements.extend(clr_measurements)

                images.extend([flip_image(elem) for elem in clr_images])
                measurements.extend([-1.0 * elem for elem in clr_measurements])

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)


samples = []
with open(DRIVING_LOGS, "r") as f:
    reader = csv.reader(f)
    for line in reader:
        samples.append(line)

train_samples, validation_samples = train_test_split(samples, test_size=0.2)

train_generator = transform_data(train_samples)
validation_generator = transform_data(validation_samples)


model = Sequential()
# normalization
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
# get ride of the top portion of the image which has sky and trees
# and bottom portion which has the hood
# image shape becomes 65 x 320 x 3
model.add(Cropping2D(cropping=CROPPING, input_shape=INPUT_SHAPE))
# convolution with filter size 24, kernel size 5x5, output shape 31 x 158 x 24
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))

# output shape: 14x77x36
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))

# output shape: 5x37x48
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(0.2))

# output shape: 3x35x64
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))

# output shape: 1x33x64
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(0.2))

# output 2112
model.add(Flatten())

# fully connected 100
model.add(Dense(100))
model.add(Dropout(0.2))

# fully connected 50
model.add(Dense(50))
model.add(Dropout(0.2))

# fully connected 10
model.add(Dense(10))
model.add(Dropout(0.2))

# fully connected 1
model.add(Dense(1))

model.compile(loss="mse", optimizer="adam")
model.fit_generator(train_generator,
                    samples_per_epoch=len(train_samples) * 6,
                    validation_data=validation_generator,
                    nb_val_samples=len(validation_samples) * 6,
                    nb_epoch=10)

model.save("model.h5")
