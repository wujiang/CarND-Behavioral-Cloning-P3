#!/usr/bin/env python

import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


images = []
measurements = []
CORRECTION = 0.15
IMG_PATH = "./data/IMG/{}"


def read_image(path):
    path = path.split("/")[-1]
    image = cv2.imread(IMG_PATH.format(path))
    return image


def flip_image(image):
    return cv2.flip(image, 1)


with open("data/driving_log.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        center_path, left_path, right_path = line[:3]
        center_image = read_image(center_path)
        left_image = read_image(left_path)
        right_image = read_image(right_path)
        images.extend([center_image, left_image, right_image])

        center_measurement = float(line[3])
        left_measurement = center_measurement + CORRECTION
        right_measurement = center_measurement - CORRECTION
        measurements.extend([center_measurement, left_measurement, right_measurement])

        images.append(flip_image(center_image))
        measurements.append(-1.0 * center_measurement)
        images.append(flip_image(left_image))
        measurements.append(-1.0 * left_measurement)
        images.append(flip_image(right_image))
        measurements.append(-1.0 * right_measurement)

X_train = np.array(images)
y_train = np.array(measurements)

INPUT_SHAPE = (160, 320, 3)
CROPPING = ((70, 25), (0, 0))

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
model.add(Cropping2D(cropping=CROPPING, input_shape=INPUT_SHAPE))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save("model.h5")
