import abc

import cv2
import tensorflow as tf
from tqdm import tqdm


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


class FaceData:
    def __init__(self):
        self.file = ''
        self.faces = []

    @abc.abstractmethod
    def filepath(self):
        pass

    def filename(self):
        return self.file

    def image(self):
        return cv2.imread(self.filepath(), cv2.IMREAD_COLOR)

    def visualize(self):
        img = self.image()
        h, w = img.shape[:2]
        for face in self.faces:
            x, x2, y, y2 = face.rect(w, h)
            cv2.rectangle(img, (x, y), (x2, y2), (0, 255, 0), 2)
        return img

    def tfrecord(self):
        filename = self.filepath()
        image = self.image()
        height, width = image.shape[:2]
        with tf.gfile.GFile(filename, 'rb') as fid:
            bytes = fid.read()

        encoded_image_data = bytes  # Encoded image bytes
        image_format = 'jpeg'.encode('utf8')  # b'jpeg' or b'png'

        rects = [face.rect(width, height) for face in self.faces]
        tf_label_and_data = tf.train.Example(features=tf.train.Features(feature={
            'image/height': int64_feature(height),
            'image/width': int64_feature(width),
            'image/filename': bytes_feature(filename.encode('utf8')),
            'image/source_id': bytes_feature(filename.encode('utf8')),
            'image/encoded': bytes_feature(encoded_image_data),
            'image/format': bytes_feature(image_format),
            'image/object/bbox/xmin': float_list_feature([float(x[0]) / width for x in rects]),
            'image/object/bbox/xmax': float_list_feature([float(x[1]) / width for x in rects]),
            'image/object/bbox/ymin': float_list_feature([float(x[2]) / height for x in rects]),
            'image/object/bbox/ymax': float_list_feature([float(x[3]) / height for x in rects]),
            'image/object/class/text': bytes_list_feature(['face'.encode('utf8')] * len(self.faces)),
            'image/object/class/label': int64_list_feature([1] * len(self.faces)),
        }))
        return tf_label_and_data


class FaceAnnotation:
    @abc.abstractmethod
    def rect(self, width, height):
        pass


def save_facedb(setname, path, facedata_gen):
    cnt = 0
    writer = tf.python_io.TFRecordWriter(path)
    for data in tqdm(facedata_gen, desc='saving %s' % setname):
        writer.write(data.tfrecord().SerializeToString())
        cnt += 1
    writer.close()
    tf.logging.info('%s=%d saved. %s' % (setname, cnt, path))
