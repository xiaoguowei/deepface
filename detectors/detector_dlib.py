import dlib

from confs.conf import DeepFaceConfs
from detectors.detector_base import FaceDetector
from utils.bbox import BoundingBox


class FaceDetectorDlib(FaceDetector):
    NAME = 'detector_dlib'

    def __init__(self):
        super().__init__()
        self.detector = dlib.get_frontal_face_detector()
        self.upsample_scale = DeepFaceConfs.get()['detector']['dlib']['scale']

    def name(self):
        return FaceDetectorDlib.NAME

    def detect(self, npimg):
        dets, scores, idx = self.detector.run(npimg, self.upsample_scale, -1)
        faces = []
        for det, score in zip(dets, scores):
            if score < DeepFaceConfs.get()['detector']['dlib']['score_th']:
                continue

            x = det.left()
            y = det.top()
            w = det.right() - det.left()
            h = det.bottom() - det.top()
            bbox = BoundingBox(x, y, w, h, score)
            faces.append(bbox)
        return faces
