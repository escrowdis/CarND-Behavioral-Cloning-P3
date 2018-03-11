import sys
import csv
import cv2
import os
import numpy as np
from sklearn.utils import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, MaxPooling2D, \
                         Dropout, Activation, Cropping2D

####################################
def get_model_memory_usage(batch_size, model):
    # ref.: https://stackoverflow.com/questions/43137288/how-to-determine-needed-memory-of-keras-model
    import numpy as np
    from keras import backend as K

    shapes_mem_count = 0
    for l in model.layers:
        single_layer_mem = 1
        for s in l.output_shape:
            if s is None:
                continue
            single_layer_mem *= s
        shapes_mem_count += single_layer_mem

    trainable_count = np.sum([K.count_params(p) for p in set(model.trainable_weights)])
    non_trainable_count = np.sum([K.count_params(p) for p in set(model.non_trainable_weights)])

    total_memory = 4.0*batch_size*(shapes_mem_count + trainable_count + non_trainable_count)
    gbytes = np.round(total_memory / (1024.0 ** 3), 3)
    return gbytes
####################################

# print(sys.argv)
# 1: [data dir] / size
data_dir = sys.argv[1]
print('data_dir ' + data_dir)
# 2: network
net_id = 0
if len(sys.argv) > 2:
    net_id = sys.argv[2]

epoch = 7
if len(sys.argv) > 3:
    epoch = int(sys.argv[3])
batch_size = 32
if len(sys.argv) > 4:
    batch_size = int(sys.argv[4])

images = []
measurements = []
X_train = []
y_train = []

# os.chdir(data_dir)
# datas_dir = [d for d in os.listdir('.') if os.path.isdir(d)]
# print(datas_dir)
if 'size' != sys.argv[1]:
    lines = []
    with open(data_dir + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        fg_first = True
        for line in reader:
            if fg_first:
                fg_first = False
                continue
            lines.append(line)

    for line in lines:
        steering_bias = np.array([0.0, 0.2, -0.2])
        for i in range(3):
            file_name = line[i].split('/')[-1]
            cur_path = data_dir + 'IMG/' + file_name
            image = cv2.imread(cur_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
            images.append(image)
            images.append(np.fliplr(image))
            measurement = float(line[3]) + float(steering_bias[i])
            measurements.append(measurement)
            measurements.append(-measurement)

measurements, images = shuffle(measurements, images)
assert len(images) == len(measurements)
print('input data size', len(measurements))

model = Sequential()
model.add(Cropping2D(cropping=((55, 25), (0, 0)), input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x / 255.0 - 0.5))

if net_id == '0':
    # Traffic Sign
    model.add(Conv2D(64, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (5, 5), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.5))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(20, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1))
elif net_id == '1':
    # NVidia
    # Ref.: https://devblogs.nvidia.com/deep-learning-self-driving-cars/
    # input: YUV
    model.add(Conv2D(24, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

if 'size' == sys.argv[1]:
    model.summary()

    mem_gb = get_model_memory_usage(batch_size, model)
    print(mem_gb)
else:
    model.compile(loss='mse', optimizer='adam')

    y_train = np.array(measurements)
    X_train = np.array(images)

    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, \
              epochs=epoch, batch_size=batch_size)

    model.save('model.h5')
