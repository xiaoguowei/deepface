import logging
import os
import sys
import cv2
import numpy as np

import fire

from confs.conf import DeepFaceConfs
from detectors.detector_dlib import FaceDetectorDlib
from recognizers.recognizer_vgg import FaceRecognizerVGG
from utils.visualization import draw_bboxs

logger = logging.getLogger('DeepFace')
logger.setLevel(logging.INFO if os.environ.get('DEBUG', 0) == 0 else logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.handlers = []
logger.addHandler(ch)


class DeepFace:
    def __init__(self):
        self.detector = None
        self.recognizer = None

    def set_detector(self, detector):
        if self.detector is not None and self.detector.name() == detector:
            return
        logger.debug('set_detector old=%s new=%s' % (self.detector, detector))
        if detector == FaceDetectorDlib.NAME:
            self.detector = FaceDetectorDlib()

    def set_recognizer(self, recognizer):
        if self.recognizer is not None and self.recognizer.name() == recognizer:
            return
        logger.debug('set_recognizer old=%s new=%s' % (self.recognizer, recognizer))
        if recognizer == FaceRecognizerVGG.NAME:
            self.recognizer = FaceRecognizerVGG()

    def run(self, detector=FaceDetectorDlib.NAME, recognizer=FaceRecognizerVGG.NAME, image='./samples/ak.jpg', visualize=False):
        self.set_detector(detector)
        self.set_recognizer(recognizer)

        if isinstance(image, str):
            logger.debug('read image, path=%s' % image)
            npimg = cv2.imread(image, cv2.IMREAD_COLOR)
        elif isinstance(image, np.ndarray):
            npimg = image
        else:
            logger.error('Argument image should be str or ndarray. image=%s' % str(image))
            sys.exit(-1)

        if npimg is None:
            logger.error('image can not be read, path=%s' % image)
            sys.exit(-1)

        logger.debug('run face detection+ %dx%d' % (npimg.shape[1], npimg.shape[0]))
        faces = self.detector.detect(npimg)
        logger.debug('run face detection-')

        if recognizer:
            rois = []
            for face in faces:
                roi = npimg[face.y:face.y+face.h, face.x:face.x+face.w, :]
                rois.append(roi)

            logger.debug('run face recognition+')
            result = self.recognizer.detect(rois)
            logger.debug('run face recognition-')
            print(result['name'])
            for face_idx, face in enumerate(faces):
                face.face_feature = result['feature'][face_idx]
                name, score = result['name'][face_idx][0]
                if score < DeepFaceConfs.get()['recognizer']['score_th']:
                    continue
                face.face_name = name
                face.face_score = score

        img = draw_bboxs(np.copy(npimg), faces)
        cv2.imwrite('result.jpg', img)
        if visualize and visualize not in ['false', 'False']:
            cv2.imshow('DeepFace', img)
            cv2.waitKey(0)

        return faces

    def save_features(self):
        name_paths = [
            ('jisoo', './samples/blackpink/blackpink_js1.jpg'),
            ('jennie', './samples/blackpink/blackpink_jn1.jpg'),
            ('lisa', './samples/blackpink/blackpink_lisa1.jpg'),
            ('rose', './samples/blackpink/blackpink_rose1.jpg'),
        ]
        features = {}
        for name, path in name_paths:
            faces = self.run(image=path)

            features[name] = faces[0].face_feature

        import pickle
        with open('db.pkl', 'wb') as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)


if __name__ == '__main__':
    fire.Fire(DeepFace)
