from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import csv
import random

import cv2
import tensorflow as tf

try:
    import cPickle as pickle
except:
    import pickle

from deepface.utils.common import get_roi
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet

with open('/data/private/deepface/resnet_train/file_bbox_0.pkl', 'rb') as f:
    pkl = pickle.load(f)
    file_bbox = pkl['bounding_box']

TRANSLATE_DELTA = 10
ROTATE_ANGLE = 5
CROPSIZE_DELTA = 10


def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _make_write_tfexample(filepath, metadata, writer):
    """Helper for creating a tf example object and writing to tfRecords file"""

    label_txt = os.path.basename(os.path.dirname(filepath))
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)

    # Detection
    detector = FaceDetectorDlib()
    faces = detector.detect(img)

    # roi crop - tightly cropped
    try:
        imgroi = get_roi(img, faces[0], roi_mode=FaceRecognizerResnet.NAME)
        imgstr = cv2.imencode('.jpg', imgroi)[1].tostring()

        # Make a record entry
        example = tf.train.Example(features=tf.train.Features(feature={
            'image/class/index': _int64_feature(metadata[label_txt]['index']),
            'image/class/identity': _bytes_feature(tf.compat.as_bytes(metadata[label_txt]['Name'])),
            'image/encoded': _bytes_feature(tf.compat.as_bytes(imgstr))
        }))
        writer.write(example.SerializeToString())
    except Exception as e:
        print('There was an error detecting face.')
        print("Exception in {}".format(str(e)))
        pass


def gen_tfrecord_vggface2(num_shards=1024):
    """Creates tfRecords file from directories of images
        TODO: implement sharding
    """

    __path = '/data/public/rw/datasets/faces/vggface2'
    # __path = '/data/private/dataset/minidemo'  # for testing purpose
    __train = 'train'
    __test = 'test'
    __meta = 'meta/identity_meta.csv'
    # __meta = '../meta/identity_meta.csv'  # for testing purpose
    __output_train = 'train.tfrecord'
    __output_test = 'test.tfrecord'

    # Read metadata
    metadata = {}
    metapath = os.path.join(__path, __meta)
    with open(metapath, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        idx_train = 0
        idx_test = 0
        for row in reader:
            if row['Flag'] == 1:
                row['index'] = idx_train
                idx_train += 1
            else:
                row['index'] = idx_test
                idx_test += 1
            metadata[row['Class_ID']] = row

    # Make *train* record file
    outputpath = os.path.join(__path, __output_train)
    writer = tf.python_io.TFRecordWriter(outputpath)

    trainpath = os.path.join(__path, __train, '*/*.jpg')
    filelist = glob.glob(trainpath)

    count = 0
    for filepath in filelist:
        _make_write_tfexample(filepath, metadata, writer)
        if count % 500 == 0:
            print("%d of %d completed" % (count, len(filelist)))
        count += 1
    writer.close()

    # Make *test* record file
    outputpath = os.path.join(__path, __output_test)
    writer = tf.python_io.TFRecordWriter(outputpath)

    testpath = os.path.join(__path, __test, '*/*.jpg')
    filelist = glob.glob(testpath)
    count = 0
    for filepath in filelist:
        _make_write_tfexample(filepath, metadata, writer)
        if count % 1000 == 0:
            print("%d of %d completed" % (count, len(filelist)))
        count += 1
    writer.close()


def _parse_example(example):
    """Helper for parsing an example protocol and returning a (features, label) pair"""

    features = {'image/class/index': tf.FixedLenFeature([], tf.int64),
                'image/class/identity': tf.FixedLenFeature([], tf.string),
                'image/encoded': tf.FixedLenFeature([], tf.string)}
    parsed_features = tf.parse_single_example(example, features)

    image_decoded = tf.image.decode_image(parsed_features['image/encoded'], channels=3)
    image_resized = tf.image.resize_image_with_crop_or_pad(image_decoded, 224, 224)
    image = tf.image.convert_image_dtype(image_resized, tf.float32)

    return image, parsed_features['image/class/index']


def _parse_image(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)
    image_resized = tf.image.resize_images(image, [224, 224])
    return image_resized, label


def _batch_normalize(tensor_in, label, epsilon=0.0001):
    """Helper for applying batch normalization on input tensor"""
    mean, variance = tf.nn.moments(tensor_in, axes=[0])
    tensor_out = (tensor_in - mean) / (variance + epsilon)
    return tensor_out, label


def _augment(image, label):
    """Helper for applying augmentation on an (image, label) pair"""
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_flip_left_right(image)
    return image, label


def _parse_and_augment(filename, label, x, y, w, h):
    # parsing image
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.cast(image_decoded, tf.float32)

    # augmentation
    image = tf.image.crop_to_bounding_box(image, y, x, h, w)
    image = tf.contrib.image.rotate(image,
                                    angles=int(360 - (ROTATE_ANGLE / 2) + (ROTATE_ANGLE * random.uniform(0, 1))))
    # image = tf.image.crop_to_bounding_box(image,
    #                                       offset_width=tf.add(
    #                                           y, tf.constant(
    #                                               int(TRANSLATE_DELTA * random.uniform(0, 1)))),
    #                                       offset_height=tf.add(
    #                                           x, tf.constant(
    #                                               int(TRANSLATE_DELTA * random.uniform(0, 1)))),
    #                                       target_width=tf.add(
    #                                           h, tf.constant(
    #                                               int(CROPSIZE_DELTA * random.uniform(0, 1)))),
    #                                       target_height=tf.add(
    #                                           w, tf.constant(
    #                                               int(CROPSIZE_DELTA * random.uniform(0, 1)))))

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_flip_left_right(image)

    image_resized = tf.image.resize_images(image, [224, 224])

    return image_resized, label


def read_tfrecord_vggface2(filename,
                           buffer_size=1000,
                           num_epochs=None,
                           shuffle=False,
                           batch_size=32):
    """Reads from tfRecords file and returns an (image, label) pair"""

    # __path = '/data/public/rw/datasets/faces/vggface2'
    __path = '/data/private/dataset/minidemo'

    # Read
    filepath = os.path.join(__path, filename)
    dataset = tf.data.TFRecordDataset(filepath)

    # Preprocessing data
    dataset = dataset.map(_parse_example)
    dataset = dataset.map(_augment)

    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def read_jpg_vggface2(
        __data,
        __path='/data/public/rw/datasets/faces/',
        buffer_size=10000,
        num_epochs=None,
        shuffle=False,
        batch_size=128,
        prefetch_buffer_size=6,
        cache_path='/data/private/deepface/resnet_train/filelist_'):
    cache_path = cache_path + __data + '.pkl'
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)

        filtered_filelist = d['filelist']
        labels = d['labels']
        x = d['x']
        y = d['y']
        w = d['w']
        h = d['h']

        from resnet_train.train_and_evaluate import logger
        logger.info('Cache file loaded from (%s)' % cache_path)
    else:
        from resnet_train.train_and_evaluate import logger
        logger.info('Loading all the datafiles..')

        datapath = os.path.join(__path, __data, '*/*.jpg')
        filelist = glob.glob(datapath)
        labelpath = os.path.join(__path, __data, '*')
        labelist = glob.glob(labelpath)

        logger.info('Mapping all the class_id\'s to indices..')
        labels = []
        x = []
        y = []
        w = []
        h = []
        filtered_filelist = []
        for file in filelist:
            try:
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                width = img.shape[1]
                height = img.shape[0]

                bounding_box = file_bbox[os.path.join(os.path.basename(os.path.dirname(file)), os.path.basename(file))]
                x.append(bounding_box['x'])
                y.append(bounding_box['y'])
                w.append(bounding_box['w'])
                h.append(bounding_box['h'])

                if height < bounding_box['y'] + bounding_box['h']:
                    print('HERES THE ERROR!!!')
                elif width < bounding_box['x'] + bounding_box['w']:
                    print("HERES ANOTHER ERROR")

                label_txt = os.path.join(__path, __data, os.path.basename(os.path.dirname(file)))
                labels.append(labelist.index(label_txt))

                filtered_filelist.append(file)
            except Exception as e:
                print(str(e))
                continue

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'filelist': filtered_filelist,
                'labels': labels,
                'x': x,
                'y': y,
                'w': w,
                'h': h
            }, f, protocol=2)
        logger.info('Mapping completed.')
        logger.info('Starting to train...')

    for i in range(len(filtered_filelist)):
        img = cv2.imread(filtered_filelist[i], cv2.IMREAD_COLOR)
        if img.shape[0] < y[i] + h[i]:
            print('%s [%d < %d + %d]' % (filtered_filelist[i], img.shape[0], y[i], h[i]))
        elif img.shape[1] < x[i] + w[i]:
            print('wow heres another error!')

    combined = list(zip(filtered_filelist, labels, x, y, w, h))
    random.shuffle(combined)
    filelist[:], labels[:], x[:], y[:], w[:], h[:] = zip(*combined)

    filelist = tf.constant(filelist)
    labels = tf.constant(labels)
    x = tf.constant(x, dtype=tf.int32)
    y = tf.constant(y, dtype=tf.int32)
    w = tf.constant(w, dtype=tf.int32)
    h = tf.constant(h, dtype=tf.int32)
    dataset = tf.data.Dataset.from_tensor_slices((filelist, labels, x, y, w, h))
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.repeat(num_epochs)

    # dataset = dataset.map(_parse_image, num_parallel_calls=40)
    # dataset = dataset.map(_augment, num_parallel_calls=40)

    dataset = dataset.map(_parse_and_augment, num_parallel_calls=40)

    dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))

    # prefetches data for next available GPU
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
