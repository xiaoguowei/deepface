from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import setuptools

_VERSION = '0.1.0'

cwd = os.path.dirname(os.path.abspath(__file__))
subprocess.check_output(["bash", "datas/residual_180601/download.sh"], cwd=cwd)

# 'opencv-python >= 3.3.1'
REQUIRED_PACKAGES = [
    'imageio >= 2.3.0',
    'natsort >= 5.3.2',
    'numpy >= 1.14.3',
    'scipy >= 1.1.0',
    'tensorflow >= 1.7.0',
    'tensorpack >= 0.8.5',
    'xmltodict >= 0.11.0',
]

DEPENDENCY_LINKS = [
]

setuptools.setup(
    name='deepface',
    version=_VERSION,
    description=
    'Deep Learning Models for Face Detection/Recognition/Alignments, implemented in Tensorflow',
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url='https://github.com/ildoonet/deepface',
    license='Apache License 2.0',
    packages=setuptools.find_packages(exclude=['tests']),
    data_files=[('datas/residual_180601', ['datas/residual_180601/model.ckpt-160000.meta',
                                           'datas/residual_180601/model.ckpt-160000.index',
                                           'datas/residual_180601/model.ckpt-160000.data-00000-of-00001'
                                           ]),
                ],
    zip_safe=False)
