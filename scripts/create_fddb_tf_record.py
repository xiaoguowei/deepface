r"""Convert raw FDDB dataset to TFRecord for face_detection.

Example usage:
    python create_fddb_tf_record.py \
      --fddb_output_dir="${IMAGE_DIR}" \
      --fddb_annotation_dir="${ANNOTATIONS_FILE}" \
      --fddb_output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from scripts.commons import *

flags = tf.app.flags
tf.flags.DEFINE_string('fddb_output_dir',
                       os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
                       'Output data directory.')
tf.flags.DEFINE_string('fddb_annotation_dir', '/data/public/rw/datasets/faces/fddb/FDDB-folds/', 'Annotation of image directory.')
tf.flags.DEFINE_string('fddb_image_dir', '/data/public/rw/datasets/faces/fddb/originalPics/', 'Image directory')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


fddb_train_folds = {0, 1, 2, 3, 4, 5, 6, 7, 8}
fddb_valid_folds = set(range(10)) - fddb_train_folds


class FDDBData(FaceData):
    __slots__ = ['file', 'faces']

    def __init__(self, file, annotations):
        super(FDDBData, self).__init__()
        self.file = file
        self.faces = [FDDBAnnotation(line) for line in annotations]

    def filepath(self):
        return os.path.join(FLAGS.fddb_image_dir, self.file + '.jpg')


class FDDBAnnotation(FaceAnnotation):
    __slots__ = ['major_radius', 'minor_radius', 'angle', 'center_x', 'center_y']

    def __init__(self, annotation_line):
        elms = [float(x) for x in annotation_line.split()]
        self.major_radius, self.minor_radius, self.angle, self.center_x, self.center_y = elms[:5]

    def rect(self, width, height):
        # note coords is in format - (<major_axis_radius minor_axis_radius angle center_x center_y 1>)
        rad_M = self.major_radius
        rad_m = self.minor_radius
        rad_avg = float(rad_M + rad_m) / 2
        cx = self.center_x
        cy = self.center_y + rad_m / 2.5
        xmin = max(cx - rad_m, 0)
        xmax = min(cx + rad_m, width - 1)
        ymin = max(cy - rad_m, 0)
        ymax = min(cy + rad_m, height - 1)
        return list(map(int, [xmin, xmax, ymin, ymax]))


def gen_data_fddb(folds=None):
    """

    :param folds: (optional) set or list containing fold numbers, eg. [1,2,3,4]
    :return: generator for FDDBData
    """
    image_dir = FLAGS.fddb_image_dir
    annotation_dir = FLAGS.fddb_annotation_dir

    list_of_annotation_files = ["FDDB-fold-%02d-ellipseList.txt" % (i + 1) for i in range(10)]

    for idx, annotation_path in enumerate(list_of_annotation_files):
        f = open(os.path.join(annotation_dir, annotation_path))
        tf.logging.debug('process %s...' % annotation_path)

        lines = f.readlines()
        f.close()

        while len(lines) > 0:
            imgfile = lines.pop(0).strip()
            facecnt = int(lines.pop(0))
            annotations = []
            for _ in range(facecnt):
                anno = lines.pop(0)
                annotations.append(anno)

            if folds is not None and idx % 10 not in folds:
                continue
            data = FDDBData(imgfile, annotations)
            yield data


def main(_):
    if not os.path.exists(FLAGS.fddb_image_dir):
        tf.logging.error('not found images: --image_dir=%s' % FLAGS.fddb_image_dir)
        exit(-1)
    if not os.path.exists(FLAGS.fddb_annotation_dir):
        tf.logging.error('not found annotations: --annotation_dir=%s' % FLAGS.fddb_annotation_dir)
        exit(-1)
    if not os.path.exists(FLAGS.fddb_output_dir):
        tf.logging.warning('not found output directory: --output_dir%s' % FLAGS.fddb_output_dir)
        os.makedirs(FLAGS.fddb_output_dir)

    train_output_path = os.path.join(FLAGS.fddb_output_dir, 'fddb_train.record')
    valid_output_path = os.path.join(FLAGS.fddb_output_dir, 'fddb_valid.record')
    total_output_path = os.path.join(FLAGS.fddb_output_dir, 'fddb_total.record')

    # save trainset / validset
    save_facedb('FDDB trainset', train_output_path, gen_data_fddb(folds=fddb_train_folds))
    save_facedb('FDDB validset', valid_output_path, gen_data_fddb(folds=fddb_valid_folds))
    save_facedb('FDDB totalset', total_output_path, gen_data_fddb(folds=None))


if __name__ == '__main__':
    tf.app.run()
