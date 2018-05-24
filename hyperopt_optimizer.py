from confs.conf import DeepFaceConfs
from deepface import DeepFace
from hyperopt import STATUS_OK


def objective(args):
    if 'crop_y_ratio' in args.keys():
        print('---------- crop_y_ratio set', args['crop_y_ratio'])
        DeepFaceConfs.get()['roi']['crop_y_ratio'] = args['crop_y_ratio']
    if 'size_ratio' in args.keys():
        print('---------- size_ratio set', args['size_ratio'])
        DeepFaceConfs.get()['roi']['size_ratio'] = args['size_ratio']

    t = DeepFace()
    try:
        score = t.test_lfw(visualize=False)
    except Exception as e:
        print('--------- error...')
        print(e)
        return 100
    print('---------- score=', score)

    return -score
