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


class ResNetRunner:
    def __init__(self):
        self.run_name = '0813_tester'    #FIXME
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
        self.tensors_to_log = {
            'train_accuracy': 'train_accuracy'
            # 'predictions': 'prediction_tensor',
            # 'true_labels': 'true_labels'
            # 'probabilities': 'softmax_tensor'
        }
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log,
            every_n_iter=100
        )

    def train(self, batch_size=256, num_epochs=50, max_steps=100000000):
        self.estimator.train(
            input_fn=lambda: read_jpg_vggface2(
                mode=tf.estimator.ModeKeys.TRAIN,
                name=self.run_name,
                num_epochs=num_epochs,
                shuffle=True,
                batch_size=batch_size),
            max_steps=max_steps,
            hooks=[self.logging_hook]
        )
        return

    def evaluate(self, batch_size=256, num_epochs=1):
        eval_results = self.estimator.evaluate(
            input_fn=lambda: read_jpg_vggface2(
                mode=tf.estimator.ModeKeys.EVAL,
                name=self.run_name,
                num_epochs=num_epochs,
                shuffle=True,
                batch_size=batch_size),
            steps=1000,
            hooks=[self.logging_hook])
        print(eval_results)
        return

    def train_and_eval(self, batch_size=256):
        while True:
            self.estimator.train(
                input_fn=lambda: read_jpg_vggface2(
                    mode=tf.estimator.ModeKeys.TRAIN,
                    name=self.run_name,
                    num_epochs=2,
                    shuffle=True,
                    batch_size=batch_size),
                max_steps=100000000000,
                hooks=[self.logging_hook]
            )
            eval_results = self.estimator.evaluate(
                input_fn=lambda: read_jpg_vggface2(
                    mode=tf.estimator.ModeKeys.EVAL,
                    name=self.run_name,
                    num_epochs=1,
                    shuffle=True,
                    batch_size=batch_size),
                steps=1000,
                hooks=[self.logging_hook]
            )
            print(eval_results)
        return

    # !!DO NOT USE!! USE WITH TF VERSION 1.10.0
    def train_and_evaluate(self, batch_size=256, max_steps=1200000):
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: read_jpg_vggface2(
                mode=tf.estimator.ModeKeys.TRAIN,
                name=self.run_name,
                num_epochs=20,
                shuffle=True,
                batch_size=batch_size),
            max_steps=max_steps,
            hooks=[self.logging_hook]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: read_jpg_vggface2(
                mode=tf.estimator.ModeKeys.EVAL,
                name=self.run_name,
                num_epochs=1,
                shuffle=True,
                batch_size=batch_size),
            steps=100,
            hooks=[self.logging_hook],
            throttle_secs=60 * 60 * 1  # every 1 hour
        )
        tf.estimator.train_and_evaluate(self.estimator, train_spec, eval_spec)

    def predict(self):
        def predict_input_fn(path):
            image_string = tf.read_file(path)
            image_decoded = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.cast(image_decoded, tf.float32)
            image_resized = tf.image.resize_images(image, [224, 224])
            return image_resized

        predict_results = self.estimator.predict(
            input_fn=predict_input_fn('/data/public/rw/datasets/faces/debug/train/n005380/0034_01.jpg'))

        for result in predict_results:
            print(result)
        return


if __name__ == '__main__':
    fire.Fire(ResNetRunner)
