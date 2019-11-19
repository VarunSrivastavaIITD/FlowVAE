#! /usr/bin/env python

import os
import scipy.misc
import tensorflow as tf
import fire


def save_images_from_event(fn, tag, output_dir="./"):
    os.makedirs(output_dir, exist_ok=True)

    image_str = tf.placeholder(tf.string)
    im_tf = tf.image.decode_image(image_str)

    sess = tf.InteractiveSession()
    with sess.as_default():
        count = 0
        for e in tf.train.summary_iterator(fn):
            for v in e.summary.value:
                if v.tag == tag:
                    im = im_tf.eval({image_str: v.image.encoded_image_string})
                    output_fn = os.path.realpath(
                        "{}/image_{:05d}.png".format(output_dir, count)
                    )
                    print("Saving '{}'".format(output_fn))
                    scipy.misc.imsave(output_fn, im)
                    count += 1


if __name__ == "__main__":
    fire.Fire({"save": save_images_from_event})

