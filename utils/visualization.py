import cv2

from utils.colors import get_random_color


def draw_bbox(npimg, bbox, color=(0, 255, 0)):
    cv2.rectangle(npimg, (bbox.x, bbox.y), (bbox.x + bbox.w, bbox.y + bbox.h), color, 2)
    if bbox.score > 0.0:
        cv2.putText(npimg, "%s %.2f" % (('%s(%.2f):' % (bbox.face_name, bbox.face_score)) if bbox.face_name else '', bbox.score), (bbox.x, bbox.y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness=2)


def draw_bboxs(npimg, bboxs):
    for i, bbox in enumerate(bboxs):
        draw_bbox(npimg, bbox, color=get_random_color(i).tuple())
    return npimg
