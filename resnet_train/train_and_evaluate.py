from __future__ import absolute_import

import fire
import tensorflow as tf

from resnet_train.model_fn import resnet_model_fn
from resnet_train.process_data import read_jpg_vggface2


class ResNetRunner:
    def __init__(self):
        self.classifier = tf.estimator.Estimator(
            model_fn=resnet_model_fn, model_dir='/data/public/rw/workspace-annie/resnet-model')

        self.tensors_to_log = {'probabilities': 'softmax_tensor'}
        self.logging_hook = tf.train.LoggingTensorHook(
            tensors=self.tensors_to_log, every_n_iter=10)

    def train(self, batch_size=128, num_epochs=None, steps=2000):
        self.classifier.train(
            input_fn=lambda: read_jpg_vggface2('test',
                                               num_epochs=num_epochs,
                                               shuffle=True,
                                               batch_size=batch_size),
            steps=steps,
            hooks=[self.logging_hook]
        )
        return

    def evaluate(self, num_epochs=1):
        eval_results = self.classifier.evaluate(
            input_fn=lambda: read_jpg_vggface2('test', num_epochs=num_epochs))
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
