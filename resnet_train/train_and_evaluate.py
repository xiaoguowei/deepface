from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys

import fire
import tensorflow as tf

from resnet_train.model_fn import resnet_model_fn
from resnet_train.process_data import read_jpg_vggface2

# python debug logger:
logger = logging.getLogger('ResNet_training')
logger.setLevel(logging.INFO if int(os.environ.get('DEBUG', 0)) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

os.environ['GLOG_logtostderr'] = '1'

class ResNetRunner:
    def __init__(self):
        # tensorflow training logger:
        tf.logging.set_verbosity(tf.logging.INFO)

        self.classifier = tf.estimator.Estimator(
            model_fn=resnet_model_fn, model_dir='/data/public/rw/workspace-annie/resnet-model8')

        logger.info('Custom estimator has been created.')

        self.tensors_to_log = {'probabilities': 'softmax_tensor', 'predictions': 'prediction_tensor'}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log, every_n_iter=50)

    def train(self, batch_size=128, num_epochs=250, steps=None):
        logger.info('Starting to train...')
        self.classifier.train(
            input_fn=lambda: read_jpg_vggface2('train',
                                               num_epochs=num_epochs,
                                               shuffle=True,
                                               batch_size=batch_size),
            steps=steps,
            hooks=[self.logging_hook]
        )
        return

    def evaluate(self, num_epochs=1):
        eval_results = self.classifier.evaluate(
            input_fn=lambda: read_jpg_vggface2('train', num_epochs=num_epochs))
        print(eval_results)
        return

    def predict(self):
        # TODO configure input source
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={})
        predict_result = self.classifier.predict(predict_input_fn)
        print(predict_result)
        return


if __name__ == '__main__':
    fire.Fire(ResNetRunner)
