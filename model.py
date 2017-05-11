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
CORRECTION = 0.2
IMG_PATH = "./data/IMG/{}"


def read_image(path):
    path = path.split("/")[-1]
    image = cv2.imread(IMG_PATH.format(path))


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

        images.append(flip_image(image))
        measurements.append(-1.0 * measurement)

X_train = np.array(images)
y_train = np.array(measurements)


INPUT_SHAPE = (160, 320, 3)
CROPPING = ((70, 25), (0, 0))

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
model.add(Cropping2D(cropping=CROPPING))
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Convolution2D(6, 5, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))


model.compile(loss="mse", optimizer="adam")
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save("model.h5")
