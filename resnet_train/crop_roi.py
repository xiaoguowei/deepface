import os
import glob
import numpy as np

import cv2
import fire
import pickle

from deepface.utils.common import get_roi
from deepface.detectors.detector_dlib import FaceDetectorDlib
from deepface.recognizers.recognizer_resnet import FaceRecognizerResnet


def read_detect_write(detector, f, __path, __data):
    img = cv2.imread(f, cv2.IMREAD_COLOR)
    faces = detector.detect(img)

    try:
        imgroi = get_roi(img, faces[0], roi_mode=FaceRecognizerResnet.NAME)
        imgname = os.path.basename(f)

        directory = os.path.join(__path, __data, os.path.basename(os.path.dirname(f)))
        if not os.path.exists(directory):
            os.makedirs(directory)

        filename = os.path.join(directory, imgname)
        cv2.imwrite(filename, imgroi)
    except Exception as e:
        print('Exception in {}'.format(str(e)))


class RunCrop:
    def __init__(self, ind=0, numworkers=1):
        self.detector = FaceDetectorDlib()

        self.__path = '/data/public/rw/datasets/faces/augment_debugger'
        self.__mode = ''

        cachepath = '/data/private/deepface/resnet_train/0813_debugger2.pkl'
        if os.path.exists(cachepath):
            with open(cachepath, 'rb') as f:
                pkl = pickle.load(f)
            print('loaded from cache file.')
            self.filelist = pkl['filelist']
        else:
            trainpath = os.path.join(self.__path, self.__mode, '*/*.jpg')
            print('loading file list...')
            self.filelist = glob.glob(trainpath)
            print('loading completed.')
            with open(cachepath, 'wb') as f:
                pickle.dump({
                    'filelist': self.filelist
                }, f, protocol=2)

        self.ind = ind
        self.numworkers = numworkers
        self.seg_start = ind * int(len(self.filelist) / numworkers)
        self.seg_end = (ind + 1) * int(len(self.filelist) / numworkers)

    def saveCoordinates(self):
        # label, img_name, x, y, w, h
        file_bbox = {}
        count = 0
        for f in self.filelist[self.seg_start:self.seg_end]:
            img = cv2.imread(f, cv2.IMREAD_COLOR)
            faces = self.detector.detect(img)
            width = img.shape[1]
            height = img.shape[0]

            label_imgname = os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))
            try:
                # Use the face in the most center:
                deltas = []
                for face in faces:
                    x = face.w / 2 + face.x
                    y = face.h / 2 + face.y
                    deltas.append(((width / 2 - x) ** 2 + (height / 2 - y) ** 2) ** 0.5)
                ind = np.argmin(deltas)

                # Discard faces outside the image
                if width < faces[ind].x + faces[ind].w or height < faces[ind].y + faces[ind].h:
                    print("wow! here's the error")
                file_bbox[label_imgname] = {'x': int(max(0, faces[ind].x)),
                                            'y': int(max(0, faces[ind].y)),
                                            'w': int(min(width - x, faces[ind].w)),
                                            'h': int(min(height - y, faces[ind].h))}
                if width < file_bbox[label_imgname]['x'] + file_bbox[label_imgname]['w'] or height < \
                        file_bbox[label_imgname]['h'] + file_bbox[label_imgname]['y']:
                    print("wow! it still hasn't been fixed!")
            except Exception as e:
                pass

            if count % 500 == 0:
                print('[Worker_%d]: %d of %d completed.' % (self.ind + 1, count, len(self.filelist) / self.numworkers))
            count += 1

        with open('/data/private/deepface/resnet_train/file_bbox_' + str(self.ind) + '.pkl', 'wb') as f:
            pickle.dump({
                'bounding_box': file_bbox
            }, f, protocol=2)

    def mergeAllBboxPkl(self):
        file_bbox = {}
        for i in range(self.numworkers):
            with open('/data/private/deepface/resnet_train/file_bbox_' + i + '.pkl', 'rb') as f:
                d = pickle.load(f)
            file_bbox.update(d['bounding_box'])

        with open('/data/private/deepface/resnet_train/file_bbox.pkl', 'wb') as f:
            pickle.dump({
                'bounding_box': file_bbox
            }, f, protocol=2)
        print('Saved %d bounding box coordinates' % len(file_bbox))

    def run(self):
        __save_path = '/data/public/rw/datasets/faces/vggface2_cropped_debug'

        count = 0

        print('Start detection.')
        for f in self.filelist[self.seg_start:self.seg_end]:
            read_detect_write(self.detector, f, __save_path, self.__mode)

            if count % 500 == 0:
                print('[Worker_%d]: %d of %d completed.' % (self.ind + 1, count, len(self.filelist) / self.numworkers))
            count += 1


if __name__ == '__main__':
    fire.Fire(RunCrop)
