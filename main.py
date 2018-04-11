# -*- coding: utf-8 -*-

import json
import glob
import os

import cv2

from keras import applications
from keras.models import Model

import numpy as np


def get_model(img_rows, img_cols, channel=3):
    vgg_model = applications.VGG16(weights='imagenet', include_top=True, input_shape=(img_rows, img_cols, channel))
    vgg_model.layers.pop()
    layer = vgg_model.layers[-1]
    model = Model(inputs=vgg_model.input, outputs=layer.output)
    model.summary(line_length=150)
    return model


def read_and_format_image(img_path):
    img = cv2.imread(img_path)
    resized_img = cv2.resize(img, (224, 224))
    return resized_img.reshape((1, 224, 224, 3))


def compute_feat_vectors(data_dir):
    list_images = glob.glob(data_dir + '/*')
    feat_vectors = {}
    model = get_model(224, 224, 3)
    i = 0
    for img_path in list_images:
        print(i)
        input = read_and_format_image(img_path)
        prediction = list(model.predict(input / 255).flatten().astype(float))
        file_name = os.path.basename(img_path)
        feat_vectors[file_name] = prediction
        i += 1
    # Saves the features
    with open('feat_vectors.json', 'w') as f:
        json.dump(feat_vectors, f)


def format_image_for_display(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (300, 300))
    return img


def compute_closest_images(img_path):
    input = read_and_format_image(img_path)
    model = get_model(224, 224, 3)
    query = list(model.predict(input / 255).flatten().astype(float))
    dataset = json.load(open('feat_vectors.json'))
    distances = []
    for key in list(dataset.keys()):
        dico = {}
        img = dataset[key]
        distance = np.linalg.norm(np.array(query) - np.array(img))
        dico['img_name'] = key
        dico['distance'] = distance
        distances.append(dico)
    distances.sort(key=lambda x: x['distance'])
    images = [d['img_name'] for d in distances]
    list_images = []

    for i in range(6):
        list_images.append(format_image_for_display('images/' + images[i]))
    print(len(list_images))

    stack_1 = np.hstack((list_images[0], list_images[1]))
    stack_1 = np.hstack((stack_1, list_images[2]))

    stack_2 = np.hstack((list_images[3], list_images[4]))
    stack_2 = np.hstack((stack_2, list_images[5]))

    stack = np.vstack((stack_1, stack_2))

    cv2.imshow('Retrieved images', stack)
    cv2.waitKey(0)

if __name__ == '__main__':
    # Directory with all the image database
    # Don't put a / at the end of the directory path
    data_dir = 'path/to/directory'

    # Computes the feature vectors of the image database and store them on a json file
    compute_feat_vectors('images')

    # Request image path
    img_path = 'path/to/image'

    # Returns the 6 closest images on the database
    compute_closest_images(img_path)
