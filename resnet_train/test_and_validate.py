from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import sys

import fire
import tensorflow as tf

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

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


class ResNetTester:
    def __init__(self):
        self.run_name = '0813_bbox'  # FIXME
        run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000,
                                            keep_checkpoint_max=3)
        self.estimator = tf.estimator.Estimator(
            # multi gpu setup (this has been deprecated after tf v1.8):
            model_fn=tf.contrib.estimator.replicate_model_fn(resnet_model_fn),
            model_dir='/data/public/rw/workspace-annie/' + self.run_name,
            config=run_config
        )
        logger.info('Custom estimator has been created.')

        # tensorflow training logger:
        tf.logging.set_verbosity(tf.logging.INFO)
        self.tensors_to_log = {'train_accuracy': 'train_accuracy'}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log,
            every_n_iter=100
        )


if __name__ == '__main__':
    fire.Fire(ResNetTester)
