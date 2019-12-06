#! /usr/bin/env python
import argparse
import csv
import math
import os
import sys
import time

import keras
import keras.backend as K
from tensorflow_gan.python.eval.classifier_metrics import diagonal_only_frechet_classifier_distance_from_activations
from tensorflow import convert_to_tensor
# import tensorflow.compat.v1.keras as keras
# import tensorflow.compat.v1.keras.backend as K
import numpy as np
from keras.models import Model
# from tensorflow.compat.v1.keras.models import Model
from scipy.special import expit
from skimage.io import imread_collection


def keras_extract_mnist_digits():

    (x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
    # xshape = x_train.shape
    return (
        x_train.reshape(x_train.shape[0], *x_train.shape[1:], 1),
        x_test.reshape(x_test.shape[0], *x_test.shape[1:], 1),
    )


def compute_real_fcd(X_real_train, classifier):
    np.random.shuffle(X_real_train)
    num_samples = X_real_train.shape[0]
    net_generated_data = X_real_train[: num_samples // 2]
    net_real_data = X_real_train[num_samples // 2 :]

    real_act = classifier.predict(net_real_data)
    gen_act = classifier.predict(net_generated_data)

    print("Calculating FCD for real data")
    fcd_tensor = diagonal_only_frechet_classifier_distance_from_activations(
        convert_to_tensor(real_act), convert_to_tensor(gen_act)
    )

    sess = K.get_session()

    fcd = sess.run(fcd_tensor)

    return fcd


def main(image_dir="./", net_loc="../cnn_mnist_10c.h5"):
    # imcollection = np.array(imread_collection(image_dir))[:, :, :, 0]
    imcollection = np.array(imread_collection(image_dir))

    net_generated_data = np.expand_dims(imcollection, 3)

    x_real_train, x_real_test = keras_extract_mnist_digits()
    num_samples = min(len(net_generated_data), len(x_real_test))

    x_real_train = x_real_train / 255
    x_real_test = x_real_test / 255
    net_generated_data = net_generated_data / 255

    np.random.shuffle(x_real_train)
    np.random.shuffle(x_real_test)
    np.random.shuffle(net_generated_data)

    x_real_train = x_real_train[:num_samples]
    x_real_test = x_real_test[:num_samples]

    full_classifier = keras.models.load_model(net_loc)
    req_layer = "flatten_1"
    classifier = Model(
        inputs=full_classifier.input,
        outputs=full_classifier.get_layer(req_layer).output,
    )

    print("Calculating FCD for train data")
    fcd_train = compute_real_fcd(x_real_train, classifier)
    print("Calculating FCD for test data")
    fcd_test = compute_real_fcd(x_real_test, classifier)

    print(
        f"samples = {num_samples} train fcd = {fcd_train:.3g} test fcd = {fcd_test:.3g}"
    )

    net_real_data = x_real_train

    assert len(net_generated_data) == len(net_real_data)
    print(
        np.max(net_generated_data),
        np.min(net_generated_data),
        f"{np.std(net_generated_data):.3f}",
        f"{np.mean(net_generated_data):.3f}",
    )
    print(
        np.max(net_real_data),
        np.min(net_real_data),
        f"{np.std(net_real_data):.3f}",
        f"{np.mean(net_real_data):.3f}",
    )
    real_act = classifier.predict(net_real_data)
    print(real_act.shape)
    gen_act = classifier.predict(net_generated_data)

    print("Calculating FCD for generated data")
    fcd_tensor = diagonal_only_frechet_classifier_distance_from_activations(
        convert_to_tensor(real_act), convert_to_tensor(gen_act)
    )

    sess = K.get_session()

    fcd = sess.run(fcd_tensor)
    print(f"fcd = {fcd:.3g}")
    sess.close()
    sys.exit(0)

    fcd_iters = 2

    gen_fcd_arr = []
    for fcd_i in range(fcd_iters):

        # inverse normalization due to tanh
        # net_generated_data = (net_generated_data + 1) / 2

        net_real_data = x_real_train

        assert len(net_generated_data) == len(net_real_data)
        print(
            np.max(net_generated_data),
            np.min(net_generated_data),
            f"{np.std(net_generated_data):.3f}",
            f"{np.mean(net_generated_data):.3f}",
        )
        print(
            np.max(net_real_data),
            np.min(net_real_data),
            f"{np.std(net_real_data):.3f}",
            f"{np.mean(net_real_data):.3f}",
        )

        np.random.shuffle(net_generated_data)
        np.random.shuffle(net_real_data)

        real_act = classifier.predict(net_real_data)
        gen_act = classifier.predict(net_generated_data)

        print("Calculating FCD for generated data")
        fcd_tensor = diagonal_only_frechet_classifier_distance_from_activations(
            convert_to_tensor(real_act), convert_to_tensor(gen_act)
        )

        sess = K.get_session()

        fcd = sess.run(fcd_tensor)
        gen_fcd_arr.append(fcd)


if __name__ == "__main__":
    import fire

    fire.Fire(main)
