import abc


class FaceRecognizer:
    def __str__(self):
        return self.name()

    @abc.abstractmethod
    def name(self):
        return 'recognizer'

    @abc.abstractmethod
    def detect(self, rois):
        pass
