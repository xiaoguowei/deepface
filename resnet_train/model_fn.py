from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

LEARNING_RATE = 1e-3
TRAIN_SIZE = 3e6
MOMENTUM = 0.99


def conv_block(input_tensor, filters, stage, block, mode, strides=(2, 2), bias=False):
    """Helper function for building the convolution block"""

    layer_name = 'conv_b%d_%d' % (stage, block)
    l = tf.layers.conv2d(input_tensor, filters[0], 1, strides=strides, use_bias=bias,
                         name=layer_name + '_1x1_reduce')
    l = tf.layers.batch_normalization(l, axis=3, momentum=MOMENTUM, name=layer_name + '_1x1_reduce/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[1], 3, padding='SAME', use_bias=bias, name=layer_name + '_3x3')
    l = tf.layers.batch_normalization(l, axis=3, momentum=MOMENTUM, name=layer_name + '_3x3/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[2], 1, name=layer_name + '_1x1_increase')
    l = tf.layers.batch_normalization(l, axis=3, momentum=MOMENTUM, name=layer_name + '_1x1_increase/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)

    m = tf.layers.conv2d(input_tensor, filters[2], 1, strides=strides, use_bias=bias, name=layer_name + '_1x1_proj')
    m = tf.layers.batch_normalization(m, axis=3, momentum=MOMENTUM, name=layer_name + '_1x1_proj/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)

    l = tf.add(l, m)
    l = tf.nn.relu(l)
    return l


def identity_block(input_tensor, filters, stage, block, mode, bias=False):
    """Helper function for building the identity block"""

    layer_name = 'conv_i%d_%d' % (stage, block)
    l = tf.layers.conv2d(input_tensor, filters[0], 1, use_bias=bias, name=layer_name + '_1x1_reduce')
    l = tf.layers.batch_normalization(l, axis=3, momentum=MOMENTUM, name=layer_name + '_1x1_reduce/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)
    l = tf.nn.relu(l)

    l = tf.layers.conv2d(l, filters[1], 3, padding='SAME', use_bias=bias, name=layer_name + '_3x3')
    l = tf.layers.batch_normalization(l, momentum=MOMENTUM, name=layer_name + '_3x3/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)
    l = tf.nn.relu(l)
    l = tf.layers.conv2d(l, filters[2], 1, use_bias=bias, name=layer_name + '_1x1_increase')
    l = tf.layers.batch_normalization(l, momentum=MOMENTUM, name=layer_name + '_1x1_increase/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)

    l = tf.add(l, input_tensor)
    l = tf.nn.relu(l)
    return l


def model_fn_resnet(features, labels, mode):
    """Model function for ResNet architecture"""
    if isinstance(features, dict):
        feature = features['feature']
    else:
        feature = features
    input_layer = tf.reshape(feature, [-1, 224, 224, 3])

    # normalize input image:
    input_layer = tf.divide(input_layer, tf.Variable(128.0, tf.float32))
    input_layer = tf.subtract(input_layer, tf.Variable(1.0, tf.float32))

    # Building hidden layers (ResNet architecture)
    # First block:
    l = tf.layers.conv2d(input_layer, 64, (7, 7), strides=(2, 2), padding='SAME', use_bias=False, name='conv1/7x7_s2')
    l = tf.layers.batch_normalization(l, axis=3, momentum=MOMENTUM, name='conv1/7x7_s2/bn',
                                      training=mode == tf.estimator.ModeKeys.TRAIN)

    l = tf.nn.relu(l)
    l = tf.layers.max_pooling2d(l, 3, 2)

    # Second block:
    l = conv_block(l, [64, 64, 256], stage=2, block=1, mode=mode, strides=(1, 1))
    l = identity_block(l, [64, 64, 256], stage=2, block=2, mode=mode)
    l = identity_block(l, [64, 64, 256], stage=2, block=3, mode=mode)

    # Third block:
    l = conv_block(l, [128, 128, 512], stage=3, block=1, mode=mode)
    l = identity_block(l, [128, 128, 512], stage=3, block=2, mode=mode)
    l = identity_block(l, [128, 128, 512], stage=3, block=3, mode=mode)
    l = identity_block(l, [128, 128, 512], stage=3, block=4, mode=mode)

    # Fourth block:
    l = conv_block(l, [256, 256, 1024], stage=4, block=1, mode=mode)
    l = identity_block(l, [256, 256, 1024], stage=4, block=2, mode=mode)
    l = identity_block(l, [256, 256, 1024], stage=4, block=3, mode=mode)
    l = identity_block(l, [256, 256, 1024], stage=4, block=4, mode=mode)
    l = identity_block(l, [256, 256, 1024], stage=4, block=5, mode=mode)
    l = identity_block(l, [256, 256, 1024], stage=4, block=6, mode=mode)

    # Fifth block:
    l = conv_block(l, [512, 512, 2048], stage=5, block=1, mode=mode)
    l = identity_block(l, [512, 512, 2048], stage=5, block=2, mode=mode)
    l = identity_block(l, [512, 512, 2048], stage=5, block=3, mode=mode)

    # Final stage:
    l = tf.layers.average_pooling2d(l, 7, 1)
    l = tf.layers.flatten(l)

    # Dropout layer (Prevent overfitting)
    l = tf.layers.dropout(l, rate=0.5)

    # Output layer
    logits = tf.layers.dense(l, units=8631, name='logits')

    learning_rate = tf.train.piecewise_constant(
        tf.train.get_global_step(),
        [80000, 100000, 120000, 140000, 160000, 200000, 300000],
        [0.1, 0.05, 0.025, 0.01, 0.005, 0.001, 0.0001, 0.00005]
    )
    return after_logits(feature, labels, mode, logits, learning_rate, weight_decay=0.0001, optname='sgd')


def after_logits(feature, labels, mode, logits, learning_rate, weight_decay, optname='sgd'):
    # Predictions
    predictions = {
        "classes": tf.argmax(logits, axis=1, name='prediction_tensor'),
        "probabilities": tf.nn.softmax(logits, name='softmax_tensor')
    }

    accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions["classes"], name='accuracy_op')
    accuracy5 = tf.cast(tf.nn.in_top_k(predictions=predictions['probabilities'], targets=labels, k=5, name='accuracy5_op'), tf.float32)
    accuracy5 = tf.reduce_mean(accuracy5, name='accuracy5_mean_op')

    # Calculate Loss - weight decay of 0.0001
    cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    l2_loss = weight_decay * tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in tf.trainable_variables()])
    loss = cross_entropy + l2_loss

    tf.identity(loss, 'cross_entropy')
    tf.identity(accuracy[1], name='acc_top1')
    tf.identity(accuracy5, name='acc_top5')
    # tf.identity(labels, 'true_labels')

    predictions['accuracy'] = accuracy[1]
    predictions['accuracy5'] = accuracy5

    # Logging tensor hook
    tf.summary.scalar('accuracy/top1', accuracy[1])
    tf.summary.scalar('accuracy/top5', accuracy5)
    tf.summary.image('input_image', feature)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Configure the Training Op
    if mode == tf.estimator.ModeKeys.TRAIN:
        # Piecewise constant learning rate:
        tf.identity(learning_rate, 'learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)

        if optname == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif optname == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif optname == 'rmsprop':
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate, momentum=0.9)
        else:
            raise Exception('invalid optname=%s' % optname)
        optimizer = tf.contrib.estimator.TowerOptimizer(optimizer)

        # Save update_ops for batch_normalization
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics
    eval_metric_ops = {"accuracy/top1": accuracy}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops, predictions=predictions)


def model_fn_mnasnet(features, labels, mode):
    """
    Model function for MnasNet architecture : MnasNet: Platform-Aware Neural Architecture Search for Mobile
    """
    if isinstance(features, dict):
        feature = features['feature']
    else:
        feature = features
    input_layer = tf.reshape(feature, [-1, 224, 224, 3])

    # normalize input image:
    input_layer = tf.divide(input_layer, tf.Variable(128.0, tf.float32))
    input_layer = tf.subtract(input_layer, tf.Variable(1.0, tf.float32))

    width_mult = 1.
    input_ch = int(32 * width_mult)
    training = (mode == tf.estimator.ModeKeys.TRAIN)
    bn_momentum = 0.9997

    # first conv3x3
    l = tf.layers.conv2d(input_layer, input_ch, (3, 3), strides=(2, 2), padding='SAME', use_bias=False, name='conv1/conv')
    l = tf.layers.batch_normalization(l, axis=3, momentum=bn_momentum, name='conv1/bn', training=training)
    l = tf.nn.relu6(l)

    # second sepconv3x3
    l = tf.layers.separable_conv2d(l, input_ch // 2, (3, 3), padding='SAME', use_bias=False, name='conv2/conv')
    l = tf.layers.batch_normalization(l, axis=3, momentum=bn_momentum, name='conv2/bn', training=training)
    l = tf.nn.relu6(l)

    # inverted residual blocks
    inverted_residual_setting = [
        # t, c, n, s, k
        [3, 24, 3, 1, 3],  # -> 56x56
        [3, 40, 3, 2, 5],  # -> 28x28
        [6, 80, 3, 2, 5],  # -> 14x14
        [6, 96, 2, 2, 3],  # -> 14x14
        [6, 192, 4, 1, 5],  # -> 7x7
        [6, 320, 1, 2, 3],  # -> 7x7
    ]

    def inverted_residual(layer_in, out_ch, stride, expand_ratio, kernel, name):
        in_ch = layer_in.shape[-1]        # TODO
        l = tf.layers.conv2d(layer_in, in_ch * expand_ratio, (1, 1), strides=(1, 1), padding='SAME', use_bias=False,
                             name=name + '/conv_expand')
        l = tf.layers.batch_normalization(l, axis=3, momentum=bn_momentum,
                                          name=name + '/bn', training=training)
        l = tf.nn.relu6(l)

        l = tf.layers.separable_conv2d(l, out_ch, (kernel, kernel), strides=(stride, stride),
                                       padding='SAME', use_bias=False,
                                       name=name + '/dw')
        l = tf.layers.batch_normalization(l, axis=3, momentum=bn_momentum,
                                          name=name + '/bn2', training=training)

        # skip-conn
        if in_ch == out_ch and stride == 1:
            l = l + layer_in

        l = tf.nn.relu6(l)
        return l

    l_cnt = 0
    for t, c, n, s, k in inverted_residual_setting:
        out_ch = int(c * width_mult)
        for i in range(n):
            l = inverted_residual(l, out_ch, s, t, k, name='inverted%d_%d' % (l_cnt, i))
            s = 1   # set stride = 1
        l_cnt += 1
        print(l_cnt, l)

    last_ch = max(2048, int(2048 * width_mult))
    l = tf.layers.conv2d(l, last_ch, (1, 1), strides=(1, 1), padding='SAME', use_bias=False,
                         name='last/conv1x1')

    # Final stage:
    l = tf.layers.average_pooling2d(l, 7, 1)
    l = tf.layers.flatten(l)

    # Dropout layer (Prevent overfitting)
    l = tf.layers.dropout(l, rate=0.5)

    # Output layer
    logits = tf.layers.dense(l, units=8631, name='logits')

    learning_rate = tf.train.piecewise_constant(
        tf.train.get_global_step(),
        [80000, 100000, 120000, 140000, 160000, 200000, 300000],
        [0.0001, 0.00005, 0.00001, 0.000005, 0.0000001, 0.00000005, 0.0001, 0.00005]
    )
    return after_logits(feature, labels, mode, logits, learning_rate, weight_decay=0.00001, optname='rmsprop')
