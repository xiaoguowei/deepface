import os
import sys

import dlib
import tensorflow as tf
import numpy as np
import cv2

from confs.conf import DeepFaceConfs
from detectors.detector_base import FaceDetector
from utils.bbox import BoundingBox


class FaceDetectorSsd(FaceDetector):
    """
    reference : https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/
    """
    NAME = 'detector_ssd'

    def __init__(self):
        super(FaceDetectorSsd, self).__init__()
        graph_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector']['ssd']['frozen_graph']
        )
        self.detector = self.load_graph(graph_path)

        predictor_path = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            DeepFaceConfs.get()['detector']['dlib']['landmark_detector']
        )
        self.predictor = dlib.shape_predictor(predictor_path)


    def load_graph(self, graph_path):
        # Here we will assume the frozen graph is in detectors/ssd/frozen_inference_graph.pb
        # We load the protobuf file from the disk and parse it to retrieve the
        # unserialized graph_def
        frozen_graph_filename = graph_path
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and return it
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name = "prefix")
        return graph


    def name(self):
        return FaceDetectorSsd.NAME


    def evaluate(self, npimg):


        #for op in self.detector.get_operations():
        #    print(op)

        # We access the input and output nodes
        x = self.detector.get_tensor_by_name('prefix/image_tensor:0')
        y1 = self.detector.get_tensor_by_name('prefix/detection_boxes:0')
        y2 = self.detector.get_tensor_by_name('prefix/detection_scores:0')
        y3 = self.detector.get_tensor_by_name('prefix/detection_classes:0')

        # We launch a Session
        with tf.Session(graph=self.detector) as sess:
            # Note: we don't need to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants
            dets,scores,classes = sess.run([y1,y2,y3], feed_dict={
                x: [npimg]
            })


        return dets[0], scores[0]

    def detect(self, npimg):
        dets, scores = self.evaluate(npimg)
        height, width, channels = npimg.shape

        faces = []
        for det, score in zip(dets, scores):
            if score < DeepFaceConfs.get()['detector']['ssd']['score_th']:
                continue



            y = int(max(det[0], 0) * height)
            x = int(max(det[1], 0) * width)
            h = int((det[2] - det[0]) * height)
            w = int((det[3] - det[1]) * width)

            if w <= 1 or h <= 1:
                continue

            bbox = BoundingBox(x, y, w, h, score)

            rect = dlib.rectangle(left=x,top=y,right=x+w,bottom=y+h)

            # find landmark
            shape = self.predictor(npimg, rect)
            coords = np.zeros((68, 2), dtype=np.int)

            # loop over the 68 facial landmarks and convert them
            # to a 2-tuple of (x, y)-coordinates
            for i in range(0, 68):
                coords[i] = (shape.part(i).x, shape.part(i).y)
            bbox.face_landmark = coords

            faces.append(bbox)

        faces = sorted(faces, key=lambda x: x.score, reverse=True)



        return faces

