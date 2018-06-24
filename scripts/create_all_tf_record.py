r"""Convert raw FDDB/WIDER dataset to TFRecord for face_detection.

Example usage:
python create_all_tf_record.py
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from scripts.commons import *
from scripts.create_fddb_tf_record import gen_data_fddb, fddb_train_folds, fddb_valid_folds
from scripts.create_wider_tf_record import gen_data_wider

flags = tf.app.flags
tf.flags.DEFINE_string('all_output_dir',
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                       'Output data directory.')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


def gen_data_all(setname='train'):
    # merged one from multiple dataset
    db_fddb = gen_data_fddb(fddb_train_folds if setname == 'train' else fddb_valid_folds)
    for data in db_fddb:
        yield data

    db_wider = gen_data_wider(setname)
    for data in db_wider:
        yield data


def main(_):
    if not os.path.exists(FLAGS.all_output_dir):
        tf.logging.warning('not found output directory: --all_output_dir%s' % FLAGS.all_output_dir)
        os.makedirs(FLAGS.all_output_dir)

    train_output_path = os.path.join(FLAGS.all_output_dir, 'all_train.record')
    valid_output_path = os.path.join(FLAGS.all_output_dir, 'all_valid.record')

    # save trainset / validset
    save_facedb('ALL trainset', train_output_path, gen_data_all(setname='train'))
    save_facedb('ALL validset', valid_output_path, gen_data_all(setname='valid'))


if __name__ == '__main__':
    tf.app.run()
