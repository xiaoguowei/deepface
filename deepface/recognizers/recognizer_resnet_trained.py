import os
import cv2
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.contrib import predictor

from ..confs.conf import DeepFaceConfs
from ..recognizers.recognizer_base import FaceRecognizer
from ..utils.common import feat_distance_cosine, faces_to_rois
from ..recognizers.recognizer_resnet import FaceRecognizerResnet


class FaceRecognizerResnetTrained(FaceRecognizer):
    NAME = 'recognizer_resnet_trained'

    def __init__(self, custom_db=None):
        __model_dir = '/data/public/rw/workspace-annie/savedModels/0816'
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'vggface2_resnet')

        if not os.path.exists(__model_dir):
            raise FileNotFoundError('Model not found, path=%s' % __model_dir)
        self.predict_fn = predictor.from_saved_model(__model_dir)

        self.db = None

        db_path = ''
        if custom_db:
            db_path = custom_db
        elif DeepFaceConfs.get()['recognizer']['resnet152'].get('db', ''):
            db_path = DeepFaceConfs.get()['recognizer']['resnet152'].get('db', '')
            db_path = os.path.join(dir_path, db_path)

        if db_path:
            with open(db_path, 'rb') as f:
                self.db = pickle.load(f)

        return

    def name(self):
        return FaceRecognizerResnetTrained.NAME

    def get_new_rois(self, rois):
        new_rois = []
        for roi in rois:
            if roi.shape[0] != 224 or roi.shape[1] != 224:
                new_roi = cv2.resize(roi, (224, 224), interpolation=cv2.INTER_AREA)
                new_rois.append(new_roi)
            else:
                new_rois.append(roi)
        return new_rois

    def extract_features(self, npimg=None, rois=None, faces=None):
        if not rois and faces:
            rois = faces_to_rois(npimg=npimg,
                                 faces=faces,
                                 roi_mode=FaceRecognizerResnet.NAME)

        if rois:
            new_rois = self.get_new_rois(rois=rois)

        probs = []
        feats = []
        for roi in new_rois:
            predictions = self.predict_fn({'input': [roi]})
            feat = predictions['features']
            prob = predictions['probabilities']
            feat = [np.squeeze(x) for x in feat]
            feats.append(feat)
            probs.append(prob)
        feats = np.vstack(feats)[:len(rois)]
        probs = np.vstack(probs)[:len(rois)]
        return probs, feats

    def detect(self, rois=None, npimg=None, faces=None):
        probs, feats = self.extract_features(npimg=npimg,
                                             rois=rois,
                                             faces=faces)

        if self.db is None:
            names = [[('', 0.0)]] * len(feats)
        else:
            names = []
            for feat in feats:
                scores = []
                for db_name, db_feature in self.db.items():
                    similarity = feat_distance_cosine(db_feature, feat)
                    scores.append((db_name, similarity))
                scores.sort(key=lambda x: x[1], reverse=True)
                names.append(scores)

        return {
            'output': probs,
            'feature': feats,
            'name': names
        }

    def get_threshold(self):
        return DeepFaceConfs.get()['recognizer']['resnet']['score_th']

