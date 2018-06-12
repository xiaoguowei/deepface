import os
import h5py

import cv2
import numpy as np
import tensorflow as tf
import pickle

from confs.conf import DeepFaceConfs
from recognizers.recognizer_base import FaceRecognizer
from utils.common import grouper, rotate_dot


def conv_block(input_tensor, filters, stage, block, strides=(2, 2), bias=False):
    layer_name = 'conv' + str(stage) + '_' + str(block)
    l = tf.layers.conv2d(input_tensor, filters[0], 1, strides=strides, use_bias=bias, name=layer_name + '_1x1_reduce')
    l = tf.layers.batch_normalization(l, axis=3, name=layer_name + '_1x1_reduce/bn')
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[1], 3, padding='SAME', use_bias=bias, name=layer_name + '_3x3')
    l = tf.layers.batch_normalization(l, axis=3, name=layer_name + '_3x3/bn')
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[2], 1, name=layer_name + '_1x1_increase')
    l = tf.layers.batch_normalization(l, axis=3, name=layer_name + '_1x1_increase/bn')

    m = tf.layers.conv2d(input_tensor, filters[2], 1, strides=strides, use_bias=bias, name=layer_name + '_1x1_proj')
    m = tf.layers.batch_normalization(m, axis=3, name=layer_name + '_1x1_proj/bn')

    l = tf.add(l, m)
    l = tf.nn.relu(l)
    return l


def identity_block(input_tensor, filters, stage, block, bias=False):
    layer_name = 'conv' + str(stage) + '_' + str(block)
    l = tf.layers.conv2d(input_tensor, filters[0], 1, use_bias=bias, name=layer_name + '_1x1_reduce')
    l = tf.layers.batch_normalization(l, axis=3, name=layer_name + '_1x1_reduce/bn')
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[1], 3, padding='SAME', use_bias=bias, name=layer_name + '_3x3')
    l = tf.layers.batch_normalization(l, name=layer_name + '_3x3/bn')
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[2], 1, use_bias=bias, name=layer_name + '_1x1_increase')
    l = tf.layers.batch_normalization(l, name=layer_name + '_1x1_increase/bn')

    l = tf.add(l, input_tensor)
    l = tf.nn.relu(l)
    return l


def get_layer_type(layer):
    layer_type = ''
    if layer.find('bn') > 0:
        layer_type = 'BatchNormalization'
    elif layer[0:4] == 'conv':
        layer_type = 'Conv2D'
    elif layer[0:4] == 'clas':
        layer_type = 'Classifier'
    return layer_type


class FaceRecognizerResnet(FaceRecognizer):
    NAME = 'recognizer_resnet'

    def __init__(self):
        self.batch_size = 1
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vggface2_resnet')
        filename = 'weight.h5'
        filepath = os.path.join(dir_path, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError('Weight file not found, path=%s' % filepath)

        self.input_node = tf.placeholder(tf.float32, shape=(None, 224, 224, 3), name='image')
        current = self.input_node

        # save for detect:
        network = {}

        # Building the cnn architecture:
        # First block:
        l = tf.layers.conv2d(current, 64, (7, 7), strides=(2, 2), padding='SAME', use_bias=False, name='conv1/7x7_s2')
        l = tf.layers.batch_normalization(l, axis=3, name='conv1/7x7_s2/bn')
        l = tf.nn.relu(l)
        l = tf.layers.max_pooling2d(l, 3, 2)

        # Second block:
        l = conv_block(l, [64, 64, 256], stage=2, block=1, strides=(1, 1))
        l = identity_block(l, [64, 64, 256], stage=2, block=2)
        l = identity_block(l, [64, 64, 256], stage=2, block=3)

        # Third block:
        l = conv_block(l, [128, 128, 512], stage=3, block=1)
        l = identity_block(l, [128, 128, 512], stage=3, block=2)
        l = identity_block(l, [128, 128, 512], stage=3, block=3)
        l = identity_block(l, [128, 128, 512], stage=3, block=4)

        # Fourth block:
        l = conv_block(l, [256, 256, 1024], stage=4, block=1)
        l = identity_block(l, [256, 256, 1024], stage=4, block=2)
        l = identity_block(l, [256, 256, 1024], stage=4, block=3)
        l = identity_block(l, [256, 256, 1024], stage=4, block=4)
        l = identity_block(l, [256, 256, 1024], stage=4, block=5)
        l = identity_block(l, [256, 256, 1024], stage=4, block=6)

        # Fifth block:
        l = conv_block(l, [512, 512, 2048], stage=5, block=1)
        l = identity_block(l, [512, 512, 2048], stage=5, block=2)
        l = identity_block(l, [512, 512, 2048], stage=5, block=3)

        # Final stage:
        l = tf.layers.average_pooling2d(l, 7, 1)
        l = tf.layers.flatten(l)
        network['feat'] = l
        output = tf.layers.dense(l, 8631, activation=tf.nn.softmax, name='classifier')  # 8631 classes
        network['out'] = output

        # Assign weights:
        assign_list = []
        with h5py.File(filepath, mode='r') as f:
            layers = f.attrs['layer_names']
            for layer in layers:
                g = f[layer]

                if isinstance(layer, bytes):
                    layer = layer.decode('utf-8')
                layer_type = get_layer_type(layer)
                if layer_type == 'Conv2D':
                    with tf.variable_scope('', reuse=True):
                        conv = tf.get_variable(layer + '/kernel')
                        w = np.asarray(g[layer + '/kernel:0'])
                        assign_op = conv.assign(tf.constant(w))
                        assign_list.append(assign_op)

                elif layer_type == 'BatchNormalization':
                    with tf.variable_scope('', reuse=True):
                        beta = tf.get_variable(layer + '/beta')
                        gamma = tf.get_variable(layer + '/gamma')
                        mean = tf.get_variable(layer + '/moving_mean')
                        var = tf.get_variable(layer + '/moving_variance')
                        w = np.asarray(g[layer + '/beta:0'])
                        assign_op = beta.assign(tf.constant(w))
                        assign_list.append(assign_op)
                        w = np.asarray(g[layer + '/gamma:0'])
                        assign_op = gamma.assign(tf.constant(w))
                        assign_list.append(assign_op)
                        w = np.asarray(g[layer + '/moving_mean:0'])
                        assign_op = mean.assign(tf.constant(w))
                        assign_list.append(assign_op)
                        w = np.asarray(g[layer + '/moving_variance:0'])
                        assign_op = var.assign(tf.constant(w))
                        assign_list.append(assign_op)

                elif layer_type == 'Classifier':
                    with tf.variable_scope('', reuse=True):
                        bias = tf.get_variable(layer + '/bias')
                        kernel = tf.get_variable(layer + '/kernel')
                        w = np.asarray(g[layer + '/bias:0'])
                        assign_op = bias.assign(tf.constant(w))
                        assign_list.append(assign_op)
                        w = np.asarray(g[layer + '/kernel:0'])
                        assign_op = kernel.assign(tf.constant(w))
                        assign_list.append(assign_op)

        # Create session:
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        init = tf.global_variables_initializer()
        sess = tf.Session()
        self.persistent_sess = sess

        # Warm-up:
        self.persistent_sess.run(init, feed_dict={
            self.input_node: np.zeros((self.batch_size, 224, 224, 3), dtype=np.uint8)
        })
        self.persistent_sess.run([assign_list, update_ops], feed_dict={
            self.input_node: np.zeros((self.batch_size, 224, 224, 3), dtype=np.uint8)
        })

        self.network = network

        self.db = None
        db_path = DeepFaceConfs.get()['recognizer']['vgg'].get('db', '')
        if db_path:
            with open(os.path.join(dir_path, db_path), 'rb') as f:
                self.db = pickle.load(f)

    def name(self):
        return FaceRecognizerResnet.NAME

    def detect(self, rois):
        new_rois = []
        for roi in rois:
            if roi.shape[0] != 224 or roi.shape[1] != 224:
                new_roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                new_rois.append(new_roi)
            else:
                new_rois.append(roi)

        probs = []
        feats = []
        names = []
        for roi_chunk in grouper(new_rois, self.batch_size,
                                 fillvalue=np.zeros((self.batch_size, 224, 224, 3), dtype=np.uint8)):
            prob, feat = self.persistent_sess.run([self.network['out'], self.network['feat']],
                                                  feed_dict={self.input_node: roi_chunk})
            feat = [np.squeeze(x) for x in feat]
            probs.append(prob)
            feats.append(feat)
            names.append('test')
        probs = np.vstack(probs)[:len(rois)]
        feats = np.vstack(feats)[:len(rois)]

        # if self.db is None:
        #     names = [[(self.class_names[idx], prop[idx]) for idx in
        #               prop.argsort()[-DeepFaceConfs.get()['recognizer']['topk']:][::-1]] for prop in probs]
        # else:
        #     # TODO
        #     names = []
        #     for feat in feats:
        #         scores = []
        #         for db_name, db_feature in self.db.items():
        #             similarity = np.dot(feat / np.linalg.norm(feat, 2), db_feature / np.linalg.norm(db_feature, 2))
        #             scores.append((db_name, similarity))
        #         scores.sort(key=lambda x: x[1], reverse=True)
        #         names.append(scores)

        return {
            'output': probs,
            'feature': feats,
            'name': names
        }
