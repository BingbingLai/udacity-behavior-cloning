import cv2
import numpy as np
import csv
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
import keras
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import pooling
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import random


_CORRECTION_NUM = 0.02

# ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed']

def helper(parent_dir):
    lines = []
    csv_file = './{}/driving_log.csv'.format(parent_dir)
    image_file_base = './{}/IMG/'.format(parent_dir)

    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            lines.append(line)


    if parent_dir in [
        'data',
        # 'more-curves6',
    ]:
        lines = random.sample(lines, int(len(lines) * 0.4))

    return_images = []
    return_measurements = []

    lines = iter(lines)
    # remove headers
    _ = next(lines)

    for line in lines:
        # center
        source_path = line[0]
        filename = source_path.split('/')[-1]
        current_path = image_file_base + filename
        image = cv2.imread(current_path)

        if image is None:
            continue

        image = brightness(image)
        return_images.append(image)
        return_images.append(cv2.flip(image, 1))
        measurement = float(line[3])
        return_measurements.append(measurement)
        return_measurements.append(measurement * -1.0)


        angle = float(line[3])

        if angle < -0.15:

            # left turn for right image
            source_path = line[2]
            filename = source_path.split('/')[-1]
            current_path = image_file_base + filename
            image = cv2.imread(current_path)
            image = brightness(image)
            return_images.append(image)
            return_images.append(cv2.flip(image, 1))

            measurement = float(line[3])
            return_measurements.append(measurement-_CORRECTION_NUM)
            return_measurements.append((measurement-_CORRECTION_NUM) * -1)


        if angle > 0.15:
            # right
            source_path = line[1]
            filename = source_path.split('/')[-1]
            current_path = image_file_base + filename
            image = cv2.imread(current_path)
            image = brightness(image)
            return_images.append(image)
            return_images.append(cv2.flip(image, 1))

            measurement = float(line[3])
            return_measurements.append(measurement+_CORRECTION_NUM)
            return_measurements.append((measurement+_CORRECTION_NUM) * -1)


    return return_images, return_measurements



#random brightness of imgs
def brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    new_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return new_img



def train():
    parent_dirs = [
        # 'data',

        # trying recovery_data
        # seems helpful! -- helps stablized
        '1_recovery_data',


        'reverse-data',

        # NOTE: trying not using more-curves
        # it just bump into the bridge but it succesfully avoiding drifting into water
        # using just more-curves2 and more-curves3

        # 'more-curves',
        'more-curves2-good-data',
        'more-curves3-good-data',

        # use cuurves 4!
        'more-curves-4',

        # not so useful!
        # 'more-curves-6',

        # use curves-new-6!
        'more-curves-new-6',
        # curves-7 is the opposite direction of curve-new-6,so...
        'more-curves-7',

        # dataaa is the trainging data! train by ourlsef
        'dataaa',

        # 'more-curves-8',
        # does not need curves5
        # 'more-curves5',

    ]

    all_images = []
    all_measurements = []
    for the_dir in parent_dirs:
        _images, _measurements = helper(the_dir)
        print('dir: {}, images: {}'.format(the_dir, len(_images)))
        all_images.extend(_images)
        all_measurements.extend(_measurements)


    print('total images', len(all_images))
    print('total measurements', len(all_measurements))


    X_train = np.array(all_images)
    y_train = np.array(all_measurements)

    print(X_train.shape)
    print(y_train.shape)

    _DROPOUT_RATE = 0.2

    model = Sequential()
    model.add(Lambda(lambda x: x / 255.0 -0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,25), (0,0))))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Convolution2D(64,3,3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.add(Dropout(_DROPOUT_RATE))

    model.compile(optimizer=Adam(lr=0.001), loss='mse' , metrics=['accuracy'])
    print('Printing...')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save('model.h5')
    print('Done')

train()
