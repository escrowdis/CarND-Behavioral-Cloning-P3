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
# Ref.: https://github.com/subodh-malgonde/behavioral-cloning/blob/master/model.py
# Ref.: https://keras.io/models/model/#fit_generator
def get_data_generator(data_dir, batch_size=32):
    lines = []
    with open(data_dir + 'driving_log.csv') as csv_file:
        reader = csv.reader(csv_file)
        fg_first = True
        for line in reader:
            if fg_first:
                fg_first = False
                continue
            lines.append(line)
    shuffle(lines)

    N = len(lines)
    batches_per_epoch = N // batch_size

    i = 0
    while(True):
        start = i * batch_size
        end = start + batch_size - 1

        X_batch = []
        y_batch = []

        for line in lines[start:end]:
            steering_bias = np.array([0.0, 0.2, -0.2])
            for i in range(3):
                file_name = line[i].split('/')[-1]
                cur_path = data_dir + 'IMG/' + file_name
                image = cv2.imread(cur_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
                X_batch.append(image)
                X_batch.append(np.fliplr(image))
                measurement = float(line[3]) + float(steering_bias[i])
                y_batch.append(measurement)
                y_batch.append(-measurement)

        i += 1
        if i == batches_per_epoch - 1:
            # reset the index so that we can cycle over the data_frame again
            i = 0

        X_batch = np.array(X_batch, np.float32)
        y_batch = np.array(y_batch, np.float32)
        yield X_batch, y_batch

#-----------------------------------------------

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

def GetModel(net_id):
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

    model.compile(loss='mse', optimizer='adam')

    return model

if 'size' == sys.argv[1]:
    model.summary()

    mem_gb = get_model_memory_usage(batch_size, model)
    print(mem_gb)
else:
    model = GetModel(net_id)

    training_generator = get_data_generator(data_dir, batch_size=batch_size)
    validation_data_generator = get_data_generator(data_dir, batch_size=batch_size)

    samples_per_epoch = (20000 // batch_size) * batch_size

    model.fit_generator(training_generator, validation_data=validation_data_generator,
                        samples_per_epoch=samples_per_epoch, nb_epoch=epoch, nb_val_samples=3000)

    model.save('model.h5')
