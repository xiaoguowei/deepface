from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import csv

import cv2
import tensorflow as tf

try:
    import cPickle as pickle
except:
    import pickle

from deepface.utils.common import get_roi
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet


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
    
    return image, label


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


def read_jpg_vggface2(__data,
                      __path='/data/public/rw/datasets/faces/vggface2_cropped',
                      buffer_size=2000,
                      num_epochs=None,
                      shuffle=False,
                      batch_size=128,
                      prefetch_buffer_size=1,
                      cache_path='/data/private/deepface/resnet_train/filelist_cropped.pkl'):
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            d = pickle.load(f)

        filelist = d['filelist']
        labels = d['labels']

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
        for file in filelist:
            label_txt = os.path.join(__path, __data, os.path.basename(os.path.dirname(file)))
            labels.append(labelist.index(label_txt))

        with open(cache_path, 'wb') as f:
            pickle.dump({
                'filelist': filelist,
                'labels': labels
            }, f, protocol=2)
        logger.info('Mapping completed.')
        logger.info('Starting to train...')

    filelist = tf.constant(filelist)
    labels = tf.constant(labels)
    dataset = tf.data.Dataset.from_tensor_slices((filelist, labels))

    dataset = dataset.map(_parse_image, num_parallel_calls=10)
    dataset.repeat = dataset.repeat(num_epochs)
    if shuffle:
        dataset = dataset.shuffle(buffer_size)
    dataset = dataset.batch(batch_size)

    # This one liner parallelizes input pipeline
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()
