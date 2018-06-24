r"""

Example usage:
    python tester.py \
      --db=fddb \
      --fddb_annotation_dir="${ANNOTATIONS_FILE}" \
      --fddb_output_dir="${OUTPUT_DIR}"
"""
import sys

import tensorflow as tf
import cv2
from scripts.create_fddb_tf_record import gen_data_fddb
from scripts.create_wider_tf_record import gen_data_wider

flags = tf.app.flags
tf.flags.DEFINE_enum('db', 'fddb', ['fddb', 'wider'], help='DB Name')
FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)


if __name__ == '__main__':
    tf.logging.info('DB: %s' % FLAGS.db)
    tf.logging.info('---- PRESS q or ESC to terminate ----')

    if FLAGS.db == 'fddb':
        gen = gen_data_fddb()
    elif FLAGS.db == 'wider':
        gen = gen_data_wider()
    else:
        raise NotImplemented('dbname=%s' % FLAGS.db)

    for data in gen:
        cv2.imshow(data.filename(), data.visualize())
        k = cv2.waitKey(0)
        if k in [27, 113]:
            sys.exit(0)
        cv2.destroyAllWindows()
