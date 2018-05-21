class BoundingBox:
    __slots__ = ['x', 'y', 'w', 'h', 'score', 'face_name', 'face_score', 'face_feature']

    def __init__(self, x=0, y=0, w=0, h=0, score=0.0):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.score = score

        self.face_name = ''
        self.face_score = ''
        self.face_feature = None
