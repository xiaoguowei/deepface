import os
import h5py

import numpy as np

from recognizers.recognizer_base import FaceRecognizer


class FaceRecognizerResnet(FaceRecognizer):
    NAME = 'recognizer_resnet'

    def __init__(self):
        self.batch_size = 4
        dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'resnet')
        filename = 'weight.h5'
        filepath = os.path.join(dir_path, filename)

        # if not os.path.exists(filepath):
        #     raise FileNotFoundError('Weight file not found, path=%s' % filepath)

        # Opens and loads all layer weights from a h5 file
        with h5py.File(filepath, mode='r') as f:
            layers = f.attrs['layer_names']

            for layer in layers:
                g = f[layer]
                if 'weight_names' in g.attrs:
                    weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]

                    for weight_name in weight_names:
                        print(weight_name)
                        weight = np.asarray(g[weight_name])
                        print(weight)

    def name(self):
        return FaceRecognizerResnet.NAME

    def detect(self, rois):
        return {
            'output': [],
            'feature': [],
            'name': []
        }
