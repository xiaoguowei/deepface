r"""Convert raw WIDER dataset to TFRecord for face_detection.

Example usage:
python create_wider_tf_record.py \
    --wider_output_dir=... \
    --wider_dir=... \
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from scripts.commons import *

flags = tf.app.flags
tf.flags.DEFINE_string('wider_output_dir',
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                       'Output data directory.')
tf.flags.DEFINE_string('wider_dir', '/data/public/rw/datasets/faces/wider_face', 'Dataset directory')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)

WIDER_PATHS = {
    'train': 'WIDER_train',
    'valid': 'WIDER_val',
    'test': 'WIDER_test',
}


class WiderData(FaceData):
    def __init__(self, file, setname, annotations):
        super(WiderData, self).__init__()
        self.file = file
        self.setname = setname
        self.faces = [WiderAnnotation(line) for line in annotations]

    def setpath(self):
        return WIDER_PATHS[self.setname]

    def filepath(self):
        return os.path.join(FLAGS.wider_dir, self.setpath(), 'images', self.file)


class WiderAnnotation(FaceAnnotation):
    def __init__(self, annotation_line):
        elms = list(map(int, annotation_line.split()))
        self.x, self.y, self.w, self.h = elms[:4]
        self.blur, self.expression, self.illumination, self.invalid, self.occlusion, self.pose = elms[4:]

    def rect(self, width, height):
        return [self.x, self.x + self.w, self.y, self.y + self.h]


def gen_data_wider(setname='train'):
    if setname == 'train':
        gtpath = 'wider_face_train_bbx_gt.txt'
    elif setname == 'valid':
        gtpath = 'wider_face_val_bbx_gt.txt'
    else:
        raise Exception('invalid setname=%s' % setname)

    tf.logging.debug('gen_data_wider %s=%s' % (setname, gtpath))
    path = os.path.join(FLAGS.wider_dir, 'wider_face_split', gtpath)
    with open(path) as f:
        lines = f.readlines()

    while len(lines) > 0:
        filename = lines.pop(0).strip()
        facenum = int(lines.pop(0))
        annotations = [lines.pop(0) for _ in range(facenum)]
        data = WiderData(filename, setname, annotations)
        yield data


def main(_):
    if not os.path.exists(FLAGS.wider_dir):
        tf.logging.error('not found db: --wider_dir=%s' % FLAGS.wider_dir)
        exit(-1)
    if not all([os.path.exists(os.path.join(FLAGS.wider_dir, v)) for _, v in WIDER_PATHS.items()]):
        tf.logging.error('%s should be located in --wider_dir=%s' % ([v for _, v in WIDER_PATHS.items()], FLAGS.wider_dir))
        exit(-1)
    if not os.path.exists(FLAGS.wider_output_dir):
        tf.logging.warning('not found output directory: --output_dir%s' % FLAGS.wider_output_dir)
        os.makedirs(FLAGS.wider_output_dir)

    train_output_path = os.path.join(FLAGS.wider_output_dir, 'wider_train.record')
    valid_output_path = os.path.join(FLAGS.wider_output_dir, 'wider_valid.record')

    # save trainset / validset
    save_facedb('WIDER trainset', train_output_path, gen_data_wider(setname='train'))
    save_facedb('WIDER validset', valid_output_path, gen_data_wider(setname='valid'))


if __name__ == '__main__':
    tf.app.run()
