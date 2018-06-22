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

r"""Convert raw WIDER dataset to TFRecord for face_detection.

Example usage:
	python create_wider_tf_record.py

	If your paths are different, make sure to modify them using tf.FLAGS
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
train - wider_face_train_bbx_gt.txt
val - wider_face_val_bbx_gt.txt

FLAGS NEEDED - none
'''

flags = tf.app.flags
tf.flags.DEFINE_boolean('include_masks', False,
						'Whether to include instance segmentations masks '
						'(PNG encoded) in the result. default: False.')


tf.flags.DEFINE_string('output_dir', '/tmp/', 'Output data directory.')
tf.flags.DEFINE_string('annotation_dir', '', 'Annotation of image directory.')
tf.flags.DEFINE_string('image_train_dir', '', 'Image train directory')
tf.flags.DEFINE_string('image_val_dir', '', 'Image validataion directory')

FLAGS = flags.FLAGS

tf.logging.set_verbosity(tf.logging.INFO)



def main(_):


  FLAGS.output_dir = "/Users/KkaKkoong/tensorflow/models/research/object_detection/data"
  FLAGS.image_train_dir = "/Users/KkaKkoong/tensorflow/data_raw/wider_train/images"
  FLAGS.image_val_dir = "/Users/KkaKkoong/tensorflow/data_raw/wider_val/images"
  FLAGS.annotation_dir = "/Users/KkaKkoong/tensorflow/data_raw/wider_face_split"

  train_output_path = os.path.join(FLAGS.output_dir, 'wider_train.record')
  val_output_path = os.path.join(FLAGS.output_dir, 'wider_val.record')


  [train_all_info, val_all_info] = load_all_info()


  write_TFRecord(train_output_path, train_all_info)
  write_TFRecord(val_output_path, val_all_info)


def load_all_info():

  image_train_dir = FLAGS.image_train_dir
  image_val_dir = FLAGS.image_val_dir
  annotation_dir = FLAGS.annotation_dir

  train_annotation_file = os.path.join(annotation_dir, "wider_face_train_bbx_gt.txt")
  val_annotation_file = os.path.join(annotation_dir, "wider_face_val_bbx_gt.txt")

  annotation_files = {"train": train_annotation_file, "val": val_annotation_file}
  image_dirs = {"train": image_train_dir, "val": image_val_dir}

  train_all_info = [] #this will be list of dictionaries
  val_all_info = []

  '''make train.record, val.record'''
  for key in annotation_files:
    with open(annotation_files[key]) as f:
      print(annotation_files[key])
      print("-------")
      lines = f.read().splitlines()
      index = 0
      line_i = 0
      while line_i < len(lines):

        if index % 3 == 0:
          element = dict()
          face_num = 0
          element['filename'] = os.path.join(image_dirs[key], str(lines[line_i]))
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
            [xmin, xmax, ymin, ymax] = convert_(coords)

            element['xmins'].append(xmin)
            element['xmaxs'].append(xmax)
            element['ymins'].append(ymin)
            element['ymaxs'].append(ymax)
            line_i += 1
          index += 1
        if key == "train":
          train_all_info.append(element)
        else: #key == "val"
          val_all_info.append(element)

  print("train_all_info instances count ", len(train_all_info))
  print("val_all_info instances count ", len(val_all_info))
  return [train_all_info, val_all_info]


def convert_(coords):
	#note coords is in format - (x1, y1, w, h, blur)
	xmin = coords[0]
	ymin = coords[1]
	xmax = xmin + coords[2]
	ymax = ymin + coords[3]

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
	  'image/object/bbox/xmin': dataset_util.float_list_feature([float(x) / width for x in xmins]),
	  'image/object/bbox/xmax': dataset_util.float_list_feature([float(x) / width for x in xmaxs]),
	  'image/object/bbox/ymin': dataset_util.float_list_feature([float(y) / height for y in ymins]),
	  'image/object/bbox/ymax': dataset_util.float_list_feature([float(y) / height for y in ymaxs]),
	  'image/object/class/text': dataset_util.bytes_list_feature([x.encode('utf8') for x in classes_text]),
	  'image/object/class/label': dataset_util.int64_list_feature(classes),
  }))
  return tf_label_and_data





if __name__ == '__main__':
  tf.app.run()
