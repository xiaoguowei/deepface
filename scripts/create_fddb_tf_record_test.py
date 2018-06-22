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
"""Test for create_fddb_tf_record.py."""

import io
import os

import numpy as np
import PIL.Image
import PIL.ImageDraw
import tensorflow as tf



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



def draw_box(image, xmins, xmaxs, ymins, ymaxs, count):
    draw = PIL.ImageDraw.Draw(image)
    for i in range(len(xmins)):
        draw.line((xmins[i], ymins[i], xmaxs[i], ymins[i]), fill=128)
        draw.line((xmaxs[i], ymins[i], xmaxs[i], ymaxs[i]), fill=128)
        draw.line((xmaxs[i], ymaxs[i], xmins[i], ymaxs[i]), fill=128)
        draw.line((xmins[i], ymaxs[i], xmins[i], ymins[i]), fill=128)
    del draw
    image.save("/Users/KkaKkoong/deepface/tmp/sample_faces/face" + str(count) + ".jpg")


image_dir = "/Users/KkaKkoong/tensorflow/data_raw/originalPics"
annotation_dir = "/Users/KkaKkoong/tensorflow/data_raw/FDDB-folds"
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

test_annotation_dir = os.path.join(annotation_dir, list_of_annotation_files[0])

with open(test_annotation_dir) as f:
      print(test_annotation_dir)
      print("-------")
      lines = f.read().splitlines()
      index = 0
      line_i = 0
      while index < 30:

        if index % 3 == 0:
          element = dict()
          face_num = 0
          element['filename'] = os.path.join(image_dir, str(lines[line_i]))
          element['filename'] += '.jpg'
          image = PIL.Image.open(element['filename'])
          element['width'], element['height'] = image.size
          element['format'] = 'jpeg'.encode('utf-8')
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
          draw_box(image, element['xmins'], element['xmaxs'], element['ymins'], element['ymaxs'], index / 3)

          index += 1



