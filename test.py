import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, \
                         Dropout, Activation

lines = []
data_dir = './data/data1/'

with open(data_dir + 'driving_log.csv') as csv_file:
    reader = csv.reader(csv_file)
    for line in reader:
        lines.append(line)

images = []
measurements = []
for line in lines:
    steering_bias = np.array()[0.0, 0.2, -0.2])
    for i in range(3):
        file_name = line[i].split('/')[-1]
        cur_path = data_dir + 'IMG/' + file_name
        image = cv2.imread(cur_path)
        images.append(image)
        images.append(np.fliplr(image))
        measurement = float(line[3] + steering_bias[i])
        measurements.append(measurement)
        measurements.append(-measurement)

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.3, shuffle=True, epochs=7)

model.save('model.h5')
