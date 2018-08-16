from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import random
import logging
import sys

import tensorflow as tf

try:
    import cPickle as pickle
except:
    import pickle

logger = logging.getLogger('[Trainer][data]')
logger.setLevel(logging.INFO if int(os.environ.get('DEBUG', 0)) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


def _parse_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    return image, label


def _augment(image, label):
    """Helper for applying augmentation on an (image, label) pair"""
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_flip_left_right(image)

    # image = tf.contrib.image.rotate(image, tf.random_uniform([1], -0.34, 0.34), interpolation='bilinear')
    # image = tf.image.resize_images(image, tf.random_uniform([2], 204, 248, dtype=tf.int32))     # target = 224
    image = tf.contrib.image.rotate(image, tf.random_uniform([1], -0.17, 0.17), interpolation='bilinear')
    image = tf.image.resize_images(image, tf.random_uniform([2], 216, 236, dtype=tf.int32))  # target = 224
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    return image, label


def _no_augment(image, label):
    image = tf.image.resize_images(image, [224, 224], preserve_aspect_ratio=True)
    image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)
    return image, label


def read_jpg_vggface2(
        __data,
        __path='/data/public/rw/datasets/faces/vggface2_cropped',
        buffer_size=10000,
        num_epochs=None,
        shuffle=False,
        batch_size=128,
        prefetch_buffer_size=6,
        augmentation=False,
        cache_path='/data/private/deepface/resnet_train/filelist_'):
    cache_path = cache_path + __data + '.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)

        filelist = d['filelist']
        labels = d['labels']

        logger.info('Cache file loaded from (%s)' % cache_path)
    else:
        logger.info('Loading all the datafiles..')

        datapath = os.path.join(__path, __data, '*/*.jpg')
        filelist = glob.glob(datapath)
        labelpath = os.path.join(__path, __data, '*')
        labelist = glob.glob(labelpath)

        logger.info('Mapping all the class_id\'s to indices..')
        labels = []
        for f in filelist:
            label_txt = os.path.join(__path, __data, os.path.basename(os.path.dirname(f)))
            labels.append(labelist.index(label_txt))

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'filelist': filelist,
                'labels': labels
            }, f, protocol=2)
        logger.info('Mapping completed.')
        logger.info('Starting to train...')

    combined = list(zip(filelist, labels))
    random.shuffle(combined)
    filelist[:], labels[:] = zip(*combined)

    filelist = tf.constant(filelist)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filelist, labels))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)

    dataset = dataset.map(_parse_image, num_parallel_calls=40)
    if augmentation:
        dataset = dataset.map(_augment, num_parallel_calls=40)
    else:
        dataset = dataset.map(_no_augment, num_parallel_calls=20)

    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # prefetches data for next available GPU
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
