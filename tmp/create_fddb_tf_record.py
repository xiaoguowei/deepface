# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Convert raw FDDB dataset to TFRecord for face_detection.

Example usage:
    python create_fddb_tf_record.py --logtostderr \
      --train_image_dir="${TRAIN_IMAGE_DIR}" \
      --val_image_dir="${VAL_IMAGE_DIR}" \
      --test_image_dir="${TEST_IMAGE_DIR}" \
      --train_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --val_annotations_file="${VAL_ANNOTATIONS_FILE}" \
      --testdev_annotations_file="${TESTDEV_ANNOTATIONS_FILE}" \
      --output_dir="${OUTPUT_DIR}"
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import os
import numpy as np
import PIL.Image


import tensorflow as tf

from object_detection.utils import dataset_util



'''Files are already defined, format is already defined, need to separate the file into 3 parts
1. Train
2. Validation
3. Test

However for the purpose of this dataset we will only have train and validation set
8 folds - train
2 folds - validation

FLAGS NEEDED - none
'''

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', False,
                        'Whether to include instance segmentations masks '
                        '(PNG encoded) in the result. default: False.')


tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_string('annotation_dir', '', 'Annotation of image directory.')
tf.flags.DEFINE_string('image_dir', '', 'Image directory')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)



def main(_):


  FLAGS.output_dir = "/Users/KkaKkoong/tensorflow/models/research/object_detection/data"
  FLAGS.image_dir = "/Users/KkaKkoong/tensorflow/data_raw/originalPics"
  FLAGS.annotation_dir = "/Users/KkaKkoong/tensorflow/data_raw/FDDB-folds"

  train_output_path = os.path.join(FLAGS.output_dir, 'fddb_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'fddb_val.record')


  [train_all_info, val_all_info] = load_all_info()


  write_TFRecord(train_output_path, train_all_info)
  write_TFRecord(val_output_path, val_all_info)


def load_all_info():
  """We will divide the dataset into 8 / 2 meaning 8 folds will be used for training,
     2 folds will be used for validation"""
  image_dir = FLAGS.image_dir
  annotation_dir = FLAGS.annotation_dir

  list_of_annotation_files = ["FDDB-fold-01-ellipseList.txt",
                                "FDDB-fold-02-ellipseList.txt",
                                "FDDB-fold-03-ellipseList.txt",
                                "FDDB-fold-04-ellipseList.txt",
                                "FDDB-fold-05-ellipseList.txt",
                                "FDDB-fold-06-ellipseList.txt",
                                "FDDB-fold-07-ellipseList.txt",
                                "FDDB-fold-08-ellipseList.txt",
                                "FDDB-fold-09-ellipseList.txt",
                                "FDDB-fold-10-ellipseList.txt"
                                ]

  train_all_info = [] #this will be list of dictionaries
  val_all_info = []

  for i in range(len(list_of_annotation_files)):
    with open(os.path.join(annotation_dir, list_of_annotation_files[i])) as f:
      print(os.path.join(annotation_dir, list_of_annotation_files[i]))
      print("-------")
      lines = f.read().splitlines()
      index = 0
      line_i = 0
      while line_i < len(lines):

        if index % 3 == 0:
          element = dict()
          face_num = 0
          element['filename'] = os.path.join(image_dir, str(lines[line_i]))
          element['filename'] += '.jpg'
          image = PIL.Image.open(element['filename'])
          element['width'], element['height'] = image.size
          element['format'] = 'jpeg'.encode('utf8')
          with tf.gfile.GFile(element['filename'], 'rb') as fid:
            element['bytes'] = fid.read()
          index += 1
          line_i += 1
        elif index % 3 == 1:
          face_num = int(lines[line_i])
          index += 1
          line_i += 1
        elif index % 3 == 2:

          element['xmins'] = []
          element['xmaxs'] = []
          element['ymins'] = []
          element['ymaxs'] = []
          for _ in range(face_num):
            coords = lines[line_i].split(" ")
            #print(coords)
            coords = list(map(float, coords[:5]))
            [xmin, xmax, ymin, ymax] = convert_ellipse2rect(coords, element['width'], element['height'])
            element['xmins'].append(xmin)
            element['xmaxs'].append(xmax)
            element['ymins'].append(ymin)
            element['ymaxs'].append(ymax)
            line_i += 1
          index += 1
        if i < 8: #0,1,2,3,4,5,6,7
          train_all_info.append(element)
        else:
          val_all_info.append(element)
      print("looped through ", index, " many lines")
      print("train_all_info instances count ", len(train_all_info))
      print("val_all_info instances count ", len(val_all_info))




  return [train_all_info, val_all_info]


def convert_ellipse2rect(coords, width, height):
    #note coords is in format - (<major_axis_radius minor_axis_radius angle center_x center_y 1>)
    rad = coords[0]
    cx = coords[3]
    cy = coords[4]
    xmin = max(cx - rad, 0)
    xmax = min(cx + rad, width - 1)
    ymin = max(cy - rad, 0)
    ymax = min(cy + rad, height - 1)
    return [xmin, xmax, ymin, ymax]


def write_TFRecord(output_path, all_info):
  writer = tf.python_io.TFRecordWriter(output_path)

  for data_and_label_info in all_info:
    tf_example = create_tf_example(data_and_label_info)
    writer.write(tf_example.SerializeToString())

  writer.close()

def create_tf_example(label_and_data_info):
  height = label_and_data_info['height'] # Image height
  width = label_and_data_info['width'] # Image width
  filename = label_and_data_info['filename'] # Filename of the image. Empty if image is not from file
  encoded_image_data = label_and_data_info['bytes'] # Encoded image bytes
  image_format = label_and_data_info['format'] # b'jpeg' or b'png'

  xmins = label_and_data_info['xmins'] # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = label_and_data_info['xmaxs'] # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = label_and_data_info['ymins'] # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = label_and_data_info['ymaxs'] # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['face'] * len(ymins) # List of string class name of bounding box (1 per box)
  classes = [1] * len(ymins) # List of integer class id of bounding box (1 per box)
  tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature([x.encode('utf8') for x in classes_text]),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_label_and_data





if __name__ == '__main__':
  tf.app.run()
