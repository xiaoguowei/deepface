import os
from glob import glob

import cv2
from tqdm import tqdm

from deepface.confs.conf import DeepFaceConfs
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.recognizers.recognizer_vgg import FaceRecognizerVGG


def get_detector(name='dlib'):
    """

    :type name: str
    :param name: 
    :return:
    """

    if name == 'dlib':
        return FaceDetectorDlib()

    return None


def get_recognizer(name='vgg', db=None):
    """

    :param db:
    :param name:
    :return:
    """
    if name == 'vgg':
        return FaceRecognizerVGG(custom_db=db)

    return None


def save_features(img_folder_path, output_path=None):
    """

    :param path: folder contain images("./samples/faces/")
    :return:
    """
    name_paths = [(os.path.basename(img_path)[:-4], img_path)
                  for img_path in glob(os.path.join(img_folder_path, "*.jpg"))]

    detector = get_detector()
    recognizer = get_recognizer()

    features = {}
    for name, path in tqdm(name_paths):
        npimg = cv2.imread(path, cv2.IMREAD_COLOR)
        faces = detector.detect(npimg)
        _, feats = recognizer.extract_features(npimg=npimg, faces=faces)
        features[name] = feats[0]

    import pickle
    if not output_path:
        output_path = os.path.join("recognizers/vggface", os.path.basename(img_folder_path) + ".pkl")
    with open(output_path, 'wb') as f:
        pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
