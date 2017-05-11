#!/usr/bin/env python

import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D


images = []
measurements = []

with open("data/driving_log.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        source_path = line[0]
        filename = source_path.split("/")[-1]
        image = cv2.imread("data/IMG/{}".format(filename))
        images.append(image)
        measurement = float(line[3])
        measurements.append(measurement)
        images.append(cv2.flip(image, 1))
        measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)


INPUT_SHAPE = (160, 320, 3)

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=INPUT_SHAPE))
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
