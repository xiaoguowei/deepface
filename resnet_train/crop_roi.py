import os
import glob

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
    def __init__(self):
        pass

    def run(self, ind, numworkers=1):

        d = FaceDetectorDlib()

        __path = '/data/public/rw/datasets/faces/vggface2'
        __save_path = '/data/public/rw/datasets/faces/vggface2_cropped_debug'
        __train = 'train'
        __test = 'train'

        cachepath = '/data/private/deepface/resnet_train/crop_filelist.pkl'
        if os.path.exists(cachepath):
            with open(cachepath, 'rb') as f:
                pkl = pickle.load(f)
            print('loaded from cache file.')
            todo_filelist = pkl['filelist']
            # saved_filelist = pkl['saved_filelist']
        else:
            trainpath = os.path.join(__path, __test, '*/*.jpg')
            savedtrainpath = os.path.join(__save_path, __test, '*/*.jpg')
            print('loading file list...')
            todo_filelist = glob.glob(trainpath)
            saved_filelist = glob.glob(savedtrainpath)
            print('loading completed.')
            with open(cachepath, 'wb') as f:
                pickle.dump({
                    'todo_filelist': todo_filelist,
                    'saved_filelist': saved_filelist
                }, f, protocol=2)

        count = 0
        seg_start = ind * int(len(todo_filelist) / numworkers)
        seg_end = (ind + 1) * int(len(todo_filelist) / numworkers)

        print('Start detection.')
        for f in todo_filelist[seg_start:seg_end]:
            indx = f.find('/test/')
            check = f[:indx] + '_cropped' + f[indx:]
            read_detect_write(d, f, __save_path, __test)

            # if check in saved_filelist:
            #     print('%s exists.' % check)
            #     continue
            # else:
            #     read_detect_write(d, f, __save_path, __test)

            if count % 500 == 0:
                print('[Worker_%d]: %d of %d completed.' % (ind + 1, count, len(todo_filelist) / numworkers))
            count += 1


if __name__ == '__main__':
    fire.Fire(RunCrop)
