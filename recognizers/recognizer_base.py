import abc


class FaceRecognizer:
    @abc.abstractmethod
    def name(self):
        return 'recognizer'

    @abc.abstractmethod
    def detect(self, rois):
        pass