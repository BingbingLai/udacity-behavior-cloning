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


_DROPOUT_RATE = 0.2


_SHARP_ANGEL_THESHOLD = 0.15



def train():
    parent_dirs = [
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

    all_images = []
    all_steerings = []
    for the_dir in parent_dirs:
        _images, _steerings = _get_images_and_steerings(the_dir)
        print('dir: {}, images: {}'.format(the_dir, len(_images)))
        all_images.extend(_images)
        all_steerings.extend(_steerings)
    print('total images', len(all_images))
    print('total steerings', len(all_steerings))
    X_train = np.array(all_images)
    y_train = np.array(all_steerings)

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
    print('Printing...')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)
    model.save('model.h5')
    print('Done')


def _get_images_and_steerings(parent_dir):
    '''
    given a parent directory, parse all the images and the angle
    '''
    # where the driving log are stored
    _CSV_FILE_BASE = './all_training_data/useful/{}/driving_log.csv'
    # path to the corresponding image files that's recorded in the csvs
    _IMAGE_FILE_BASE = './all_training_data/useful/{}/IMG/'

    def _get_image_path(path):
        return path.split('/')[-1]

    def _get_csv_lines(parent_dir):
        lines = []
        csv_file = _CSV_FILE_BASE.format(parent_dir)
        image_file_base = _IMAGE_FILE_BASE.format(parent_dir)
        with open(csv_file) as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)

        lines = iter(lines)
        _ = next(lines)
        return lines

    def _left_turning(angle):
        return angle < (-1 * _SHARP_ANGEL_THESHOLD)

    def _right_turning(angle):
        return angle > 1 * _SHARP_ANGEL_THESHOLD

    return_images = []
    return_steerings = []
    for line in _get_csv_lines(parent_dir):
        center, left, right, steering, _, _, _ = line
        img_path_base = _IMAGE_FILE_BASE.format(parent_dir)

        steering = float(steering)
        images, steerings = _process_img_and_steering(img_path_base + _get_image_path(center), steering)
        return_images.extend(images)
        return_steerings.extend(steerings)

        if _left_turning(steering):
            images, steerings = _process_img_and_steering(img_path_base + _get_image_path(right), steering-_CORRECTION_NUM)
            return_images.extend(images)
            return_steerings.extend(steerings)

        if _right_turning(steering):
            images, steerings = _process_img_and_steering(img_path_base + _get_image_path(left), steering + _CORRECTION_NUM)
            return_images.extend(images)
            return_steerings.extend(steerings)

    return return_images, return_steerings


def _process_img_and_steering(image_full_path, steering):
    to_return_images, to_return_steerings = [], []
    image = cv2.imread(image_full_path)
    assert image is not None
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
