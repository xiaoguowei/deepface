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

from resnet_train.model_fn import model_fn_resnet, model_fn_mnasnet
from resnet_train.data import read_jpg_vggface2

# python debug logger:
logger = logging.getLogger('[Trainer]')
logger.setLevel(logging.INFO if int(os.environ.get('DEBUG', 0)) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)

os.environ['GLOG_logtostderr'] = '1'
tf.logging.set_verbosity(tf.logging.INFO)


class TrainerRecognition:
    def __init__(self):
        pass

    def _set_estimator(self):
        model_arch = 'mnasnet'
        model_name = 'aug3'  # fixme
        model_path = '/data/private/deepface-models/%s_%s' % (model_arch, model_name)
        run_config = tf.estimator.RunConfig(save_checkpoints_steps=1000, keep_checkpoint_max=30)

        model_fn = model_fn_resnet
        if model_arch == 'mnasnet':
            model_fn = model_fn_mnasnet

        self.estimator = tf.estimator.Estimator(
            # multi gpu setup (this has been deprecated after tf v1.8):
            model_fn=tf.contrib.estimator.replicate_model_fn(model_fn),
            model_dir=model_path,
            config=run_config
        )
        logger.info('Custom estimator has been created. %s' % model_path)

        self.logging_hook = tf.train.LoggingTensorHook(
            tensors={
                'accuracy': 'acc_top1',
                'accuracy5': 'acc_top5'
            },
            every_n_iter=100
        )

    def evaluate(self, batch_size=64, num_epochs=1):
        self._set_estimator()
        eval_results = self.estimator.evaluate(
            input_fn=lambda: read_jpg_vggface2(
                'validation_split',
                num_epochs=num_epochs,
                shuffle=True,
                augmentation=False,
                batch_size=batch_size),
            steps=None,
            hooks=[self.logging_hook])
        print(eval_results)
        return

    def train(self, batch_size=256, max_steps=4000000):
        self._set_estimator()
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: read_jpg_vggface2(
                'train_split',
                num_epochs=50,
                shuffle=True,
                augmentation=True,
                batch_size=batch_size),
            max_steps=max_steps,
            hooks=[self.logging_hook]
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: read_jpg_vggface2(
                'validation_split',
                num_epochs=1,
                shuffle=True,
                augmentation=False,
                batch_size=batch_size),
            steps=100,
            # hooks=[self.logging_hook],
            start_delay_secs=60*60*2,   # two hours
            throttle_secs=60*60*1   # every 1 hour
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
    fire.Fire(TrainerRecognition)
