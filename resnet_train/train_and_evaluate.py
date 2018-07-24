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
        run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000,
                                            keep_checkpoint_max=3)
        self.estimator = tf.estimator.Estimator(
            # multi gpu setup:
            model_fn=tf.contrib.estimator.replicate_model_fn(resnet_model_fn),
            model_dir='/data/public/rw/workspace-annie/0722_SGD_after_25_40epochs_learning_rate_0.15',
            config=run_config
        )
        logger.info('Custom estimator has been created.')

        # tensorflow training logger:
        tf.logging.set_verbosity(tf.logging.INFO)
        self.tensors_to_log = {
            'train_accuracy': 'train_accuracy',
            'learning_rate': 'learning_rate'}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log,
            every_n_iter=50
        )

    def train(self, batch_size=256, num_epochs=200, max_steps=2400000):
        self.estimator.train(
            input_fn=lambda: read_jpg_vggface2('train',
                                               num_epochs=num_epochs,
                                               shuffle=True,
                                               batch_size=batch_size),
            max_steps=max_steps,
            hooks=[self.logging_hook]
        )
        return

    def evaluate(self, batch_size=256, num_epochs=1):
        eval_results = self.estimator.evaluate(
            input_fn=lambda: read_jpg_vggface2('train', num_epochs=num_epochs, batch_size=batch_size))
        print(eval_results)
        return

    def predict(self):
        # TODO configure input source
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={})
        predict_result = self.estimator.predict(predict_input_fn)
        print(predict_result)
        return


if __name__ == '__main__':
    fire.Fire(ResNetRunner)
