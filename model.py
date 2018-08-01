import os
import csv
import random

import cv2
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.layers import pooling
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


_CORRECTION = 0.02


_DROPOUT_RATE = 0.2


_SHARP_ANGEL_THESHOLD = 0.15

_TEST_SIZE = 0.2



def train():
    training_data_paths = [
        # dataaa is the trainging data! train by ourlsef
        'dataaa',
        # trying recovery_data
        # seems helpful! -- helps stablized
        '1_recovery_data',
        # driving in the opposite direction
        'reverse-data',
        # training around the curves
        'more-curves2-good-data',
        'more-curves3-good-data',
        # use cuurves 4!
        'more-curves-4',
        # use curves-new-6!
        'more-curves-new-6',
        # curves-7 is the opposite direction of curve-new-6,so...
        'more-curves-7',
    ]

    # majaor keeey variables
    all_images = []
    all_steerings = []
    csv_rows = _get_csv_rows(training_data_paths)
    train_samples, validation_samples = train_test_split(csv_rows, test_size=_TEST_SIZE)

    train_generator = _get_images_and_steerings(train_samples, batch_size=32)
    validation_generator = _get_images_and_steerings(validation_samples, batch_size=32)

    # the model
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
    print('Training...')
    model.fit_generator(
        train_generator,
        samples_per_epoch=len(train_samples),
        validation_data=validation_generator,
        nb_val_samples=len(validation_samples),
        nb_epoch=3,
    )
    model.save('model.h5')
    print('Done')


# where the driving log are stored
_CSV_FILE_BASE = './all_training_data/useful/{}/driving_log.csv'
# path to the corresponding image files that's recorded in the csvs
_IMAGE_FILE_BASE = './all_training_data/useful/{}/IMG/'


def _get_csv_rows(paths):
    def _get_csv_lines_helper(parent_dir):
        lines = []
        csv_file = _CSV_FILE_BASE.format(parent_dir)
        image_file_base = _IMAGE_FILE_BASE.format(parent_dir)
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        lines = iter(lines)
        _ = next(lines)
        return list(lines)

    to_return = []
    for p in paths:
        rows = _get_csv_lines_helper(p)
        print('{} has {} rows'.format(p, len(rows)))
        to_return.extend(_get_csv_lines_helper(p))

    return to_return


def _get_images_and_steerings(samples, batch_size=32):
    '''
    given a parent directory, parse all the images and the angle
    '''
    def _get_image_path(path):
        return path.split('/')[-1]

    def _left_turning(angle):
        return angle < (-1 * _SHARP_ANGEL_THESHOLD)

    def _right_turning(angle):
        return angle > 1 * _SHARP_ANGEL_THESHOLD

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            return_images = []
            return_steerings = []
            for row in batch_samples:
                # parse the csv rows
                center, left, right, steering, _, _, _ = row
                steering = float(steering)

                dirname = center.split('/')[-3]
                img_path_base = _IMAGE_FILE_BASE.format(dirname)

                center_img_fullpath = os.path.join(img_path_base, _get_image_path(center))
                left_img_fullpath = os.path.join(img_path_base, _get_image_path(left))
                right_img_fullpath = os.path.join(img_path_base, _get_image_path(right))

                # process center
                images, steerings = _process_img_and_steering(center_img_fullpath, steering)
                return_images.extend(images)
                return_steerings.extend(steerings)

                # process left
                if _left_turning(steering):
                    # right camera images to steer left a bit more
                    images, steerings = _process_img_and_steering(right_img_fullpath, steering-_CORRECTION)
                    return_images.extend(images)
                    return_steerings.extend(steerings)

                # process right
                if _right_turning(steering):
                    # left camera images to steer right a bit more
                    images, steerings = _process_img_and_steering(left_img_fullpath, steering + _CORRECTION)
                    return_images.extend(images)
                    return_steerings.extend(steerings)


                X_train = np.array(return_images)
                y_train = np.array(return_steerings)

            yield shuffle(X_train, y_train)


def _process_img_and_steering(image_full_path, steering):
    to_return_images, to_return_steerings = [], []
    image = cv2.imread(image_full_path)

    if image is None:
        return [], []

    image = _apply_random_brightness(image)

    # add the image and steering as it is
    to_return_images.append(image)
    to_return_steerings.append(steering)

    # as the flipped version of image and steering
    to_return_images.append(cv2.flip(image, 1))
    to_return_steerings.append(steering * -1.0)
    return to_return_images, to_return_steerings


def _apply_random_brightness(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    rand = random.uniform(0.3, 1.0)
    hsv[:,:,2] = rand*hsv[:,:,2]
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


if __name__ == '__main__':
    exit(train())
